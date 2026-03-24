import pandas as pd
import numpy as np
import ast

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn, optim
from torchvision import datasets, utils, models
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import torch.optim as optim
from torchvision.transforms import v2
import multiprocessing

from sklearn.model_selection import GroupShuffleSplit, train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import os
import json
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

#########################################################################
# Constants
#########################################################################

SUBSET = False
IMAGE_SIZE = 384
BATCH_SIZE = 16
LEARNING_RATE = 0.00004

LOAD_CHECKPOINT = False
checkpoint_version = 1
checkpoint_epoch = 20

HEAD_WARMUP_EPOCHS = 4
HEAD_WARMUP_LR = 1e-3
FINETUNE_EPOCHS = 18
EARLY_STOPPING_PATIENCE = 5

SAVE_CHECKPOINT = True
SAVE_AS_VERSION = "15"

#########################################################################
# Class & function definitions (importable by worker processes)
#########################################################################


class PythonGeneDataset(Dataset):
    def __init__(self, labels_df, img_dir, indices=None, transform=None):
        self.labels_df = labels_df
        if indices is not None:
            self.labels_df = self.labels_df.iloc[indices]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.labels_df.iloc[idx, 0]}.png")
        image = Image.open(img_name)
        labels = torch.tensor(self.labels_df.iloc[idx, 7:].astype('float32').values)
        if self.transform:
            image = self.transform(image)
        return image, labels


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            F_loss = alpha * F_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


def compute_mAP(model, data_loader, device, num_labels):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    APs = []
    for i in range(num_labels):
        class_preds = all_preds[:, i]
        class_targets = all_targets[:, i]
        if torch.sum(class_targets) == 0:
            continue
        sorted_indices = torch.argsort(class_preds, descending=True)
        sorted_targets = class_targets[sorted_indices]
        true_positives = torch.cumsum(sorted_targets, dim=0).float()
        false_positives = torch.cumsum((sorted_targets == 0), dim=0).float()
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (torch.sum(class_targets) + 1e-10)
        precision = torch.cat([torch.tensor([1.0]), precision])
        recall = torch.cat([torch.tensor([0.0]), recall])
        for j in range(len(precision) - 1, 0, -1):
            precision[j-1] = max(precision[j-1], precision[j])
        AP = 0.0
        for j in range(len(precision) - 1):
            AP += (recall[j+1] - recall[j]) * precision[j+1]
        APs.append(AP.item())

    if len(APs) == 0:
        return 0.0, []
    mAP = sum(APs) / len(APs)
    return mAP, APs


#########################################################################
# Main — everything below only runs in the main process
#########################################################################

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Data loading ---
    labels_extended = pd.read_csv('data/selected_gene_df.csv')
    if SUBSET:
        labels_extended = labels_extended.sample(frac=0.01, random_state=123)

    clean_possible_genes = labels_extended.columns.to_list()[7:]
    num_labels = len(clean_possible_genes)
    print(f'Number of labels: {num_labels}')

    # --- Group-based split to prevent data leakage ---
    listing_ids = labels_extended['index'].astype(str).str.split('-').str[0]
    gss = GroupShuffleSplit(n_splits=1, train_size=0.85, random_state=123)
    train_idx, test_idx = next(gss.split(labels_extended, groups=listing_ids))

    train_df = labels_extended.iloc[train_idx].reset_index(drop=True)
    test_df = labels_extended.iloc[test_idx].reset_index(drop=True)
    print(f'Shape of trainset: {train_df.shape}')
    print(f'Shape of testset: {test_df.shape}')

    # --- Transforms ---
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.6007, 0.5679, 0.5206], std=[0.2411, 0.2392, 0.2479]),
        v2.RandomErasing(p=0.2),
    ])

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.6007, 0.5679, 0.5206], std=[0.2411, 0.2392, 0.2479]),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')

    # --- Group-based train/val split ---
    train_listing_ids = train_df['index'].astype(str).str.split('-').str[0]
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_sub_idx, valid_sub_idx = next(gss_val.split(train_df, groups=train_listing_ids))

    train_dataset = PythonGeneDataset(labels_df=train_df, img_dir='data/img/',
                                      indices=train_sub_idx, transform=train_transform)
    valid_dataset = PythonGeneDataset(labels_df=train_df, img_dir='data/img/',
                                      indices=valid_sub_idx, transform=eval_transform)
    print(f'Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}')

    num_workers = multiprocessing.cpu_count()
    print(f'Number of workers: {num_workers}')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f'Saving as version: {SAVE_AS_VERSION}')

    # --- Model ---
    efficientnet = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')
    num_labels = len(clean_possible_genes)

    new_layers = nn.Sequential(
        nn.LazyLinear(2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.LazyLinear(num_labels)
    )
    efficientnet.classifier = new_layers

    # --- Per-class alpha from inverse frequency ---
    pos_counts = train_df.iloc[:, 7:].sum(axis=0).values.astype(np.float32)
    pos_counts = np.clip(pos_counts, 1.0, None)
    inv_freq = 1.0 / pos_counts
    alpha_per_class = torch.tensor(inv_freq / inv_freq.sum() * len(inv_freq), dtype=torch.float32)
    print(f'Per-class alpha range: [{alpha_per_class.min():.3f}, {alpha_per_class.max():.3f}]')

    efficientnet.to(device)
    criterion = FocalLoss(alpha=alpha_per_class)

    # =====================================================
    # Phase 1: Freeze backbone, train classifier head only
    # =====================================================
    print('\n=== Phase 1: Training classifier head (backbone frozen) ===')
    for param in efficientnet.features.parameters():
        param.requires_grad = False
    for param in efficientnet.classifier.parameters():
        param.requires_grad = True

    head_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, efficientnet.parameters()),
        lr=HEAD_WARMUP_LR, weight_decay=0.01
    )
    head_scaler = GradScaler('cuda')

    result = {
        "epoch": [], "train_loss": [], "valid_loss": [],
        "valid_mAP": [], "lr": [], "phase": []
    }

    best_valid_mAP = 0.0
    patience_counter = 0

    for epoch in range(1, HEAD_WARMUP_EPOCHS + 1):
        efficientnet.train()
        train_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Phase 1 - Epoch {epoch}/{HEAD_WARMUP_EPOCHS}")
                inputs, labels = inputs.to(device), labels.to(device)

                head_optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', dtype=torch.float16):
                    outputs = efficientnet(inputs)
                    loss = criterion(outputs, labels)

                head_scaler.scale(loss).backward()
                head_scaler.unscale_(head_optimizer)
                torch.nn.utils.clip_grad_norm_(efficientnet.parameters(), max_norm=1.0)
                head_scaler.step(head_optimizer)
                head_scaler.update()

                train_loss += loss.item() * inputs.size(0)
            train_loss = train_loss / len(train_loader.dataset)

        # Validation
        efficientnet.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast('cuda', dtype=torch.float16):
                    outputs = efficientnet(inputs)
                    loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
        valid_loss = valid_loss / len(valid_loader.dataset)

        valid_mAP, _ = compute_mAP(efficientnet, valid_loader, device, num_labels)

        result["epoch"].append(epoch)
        result["train_loss"].append(train_loss)
        result["valid_loss"].append(valid_loss)
        result["valid_mAP"].append(valid_mAP)
        result["lr"].append(HEAD_WARMUP_LR)
        result["phase"].append(1)

        print(f'Phase 1 Epoch {epoch}/{HEAD_WARMUP_EPOCHS}, Train Loss: {train_loss:.6f}, '
              f'Valid Loss: {valid_loss:.6f}, Valid mAP: {valid_mAP:.4f}')

        if SAVE_CHECKPOINT:
            torch.save({
                'epoch': epoch, 'model_state_dict': efficientnet.state_dict(),
                'optimizer_state_dict': head_optimizer.state_dict(),
                'image_size': IMAGE_SIZE, 'batch_size': BATCH_SIZE,
                'train_loss': train_loss, 'valid_loss': valid_loss,
                'valid_mAP': valid_mAP, 'phase': 1,
            }, f'model/model_v{SAVE_AS_VERSION}_epoch{epoch}.pt')

        if valid_mAP > best_valid_mAP:
            best_valid_mAP = valid_mAP
            patience_counter = 0
        else:
            patience_counter += 1

    # =====================================================
    # Phase 2: Unfreeze all, discriminative learning rates
    # =====================================================
    print('\n=== Phase 2: Fine-tuning full model (discriminative LRs) ===')
    for param in efficientnet.parameters():
        param.requires_grad = True

    param_groups = [
        {'params': efficientnet.features[:4].parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': efficientnet.features[4:].parameters(), 'lr': LEARNING_RATE * 0.5},
        {'params': efficientnet.classifier.parameters(), 'lr': LEARNING_RATE * 2.5},
    ]

    finetune_optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    warmup_scheduler = LinearLR(finetune_optimizer, start_factor=0.1, total_iters=2)
    cosine_scheduler = CosineAnnealingLR(finetune_optimizer, T_max=FINETUNE_EPOCHS - 2)
    scheduler = SequentialLR(finetune_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[2])

    finetune_scaler = GradScaler('cuda')
    patience_counter = 0

    for epoch_offset in range(1, FINETUNE_EPOCHS + 1):
        epoch = HEAD_WARMUP_EPOCHS + epoch_offset
        efficientnet.train()
        train_loss = 0.0

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Phase 2 - Epoch {epoch}/{HEAD_WARMUP_EPOCHS + FINETUNE_EPOCHS}")
                inputs, labels = inputs.to(device), labels.to(device)

                finetune_optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', dtype=torch.float16):
                    outputs = efficientnet(inputs)
                    loss = criterion(outputs, labels)

                finetune_scaler.scale(loss).backward()
                finetune_scaler.unscale_(finetune_optimizer)
                torch.nn.utils.clip_grad_norm_(efficientnet.parameters(), max_norm=1.0)
                finetune_scaler.step(finetune_optimizer)
                finetune_scaler.update()

                train_loss += loss.item() * inputs.size(0)
            train_loss = train_loss / len(train_loader.dataset)

        # Validation
        efficientnet.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast('cuda', dtype=torch.float16):
                    outputs = efficientnet(inputs)
                    loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
        valid_loss = valid_loss / len(valid_loader.dataset)

        valid_mAP, _ = compute_mAP(efficientnet, valid_loader, device, num_labels)
        scheduler.step()

        current_lrs = [pg['lr'] for pg in finetune_optimizer.param_groups]
        result["epoch"].append(epoch)
        result["train_loss"].append(train_loss)
        result["valid_loss"].append(valid_loss)
        result["valid_mAP"].append(valid_mAP)
        result["lr"].append(current_lrs[1])
        result["phase"].append(2)

        print(f'Phase 2 Epoch {epoch}/{HEAD_WARMUP_EPOCHS + FINETUNE_EPOCHS}, '
              f'Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}, '
              f'Valid mAP: {valid_mAP:.4f}, LRs: {[f"{lr:.2e}" for lr in current_lrs]}')

        if SAVE_CHECKPOINT:
            torch.save({
                'epoch': epoch, 'model_state_dict': efficientnet.state_dict(),
                'optimizer_state_dict': finetune_optimizer.state_dict(),
                'image_size': IMAGE_SIZE, 'batch_size': BATCH_SIZE,
                'train_loss': train_loss, 'valid_loss': valid_loss,
                'valid_mAP': valid_mAP, 'phase': 2,
            }, f'model/model_v{SAVE_AS_VERSION}_epoch{epoch}.pt')

        if valid_mAP > best_valid_mAP:
            best_valid_mAP = valid_mAP
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without mAP improvement.')
                break

        pd.DataFrame(result).to_csv(f'results/version{SAVE_AS_VERSION}_ImageSize{IMAGE_SIZE}_LR{LEARNING_RATE}.csv', index=False)

    # =====================================================
    # Evaluate on test set using best model (by val mAP)
    # =====================================================
    best_epoch_idx = np.argmax(result["valid_mAP"])
    best_model_epoch = result["epoch"][best_epoch_idx]
    best_model_path = f'model/model_v{SAVE_AS_VERSION}_epoch{best_model_epoch}.pt'
    print(f'\nBest model: epoch {best_model_epoch} with valid mAP: {result["valid_mAP"][best_epoch_idx]:.4f}')

    best_checkpoint = torch.load(best_model_path, map_location=device)

    efficientnet = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')
    efficientnet.classifier = nn.Sequential(
        nn.LazyLinear(2048), nn.BatchNorm1d(2048), nn.ReLU(),
        nn.Dropout(0.5), nn.LazyLinear(num_labels)
    )
    efficientnet.load_state_dict(best_checkpoint['model_state_dict'])
    efficientnet.to(device).eval()

    test_dataset = PythonGeneDataset(labels_df=test_df, img_dir='data/img/', transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    test_mAP, test_class_APs = compute_mAP(efficientnet, test_loader, device, num_labels)
    print(f'\nTest set mAP: {test_mAP:.4f}')
    print('\nPer-class Average Precision:')
    for gene, ap in zip(clean_possible_genes, test_class_APs):
        print(f'  {gene}: {ap:.4f}')

    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast('cuda', dtype=torch.float16):
                outputs = efficientnet(inputs)
                loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.6f}')

    # --- Per-class threshold optimization on validation set ---
    print('\n=== Optimizing per-class thresholds on validation set ===')
    from sklearn.metrics import f1_score, classification_report

    efficientnet.eval()
    all_val_preds, all_val_targets = [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            outputs = efficientnet(inputs)
            all_val_preds.append(torch.sigmoid(outputs).cpu())
            all_val_targets.append(targets)

    all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
    all_val_targets = torch.cat(all_val_targets, dim=0).numpy()

    optimal_thresholds = []
    for i in range(num_labels):
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.1, 0.91, 0.05):
            preds_binary = (all_val_preds[:, i] >= thresh).astype(float)
            f1 = f1_score(all_val_targets[:, i], preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)
        print(f'  {clean_possible_genes[i]}: threshold={best_thresh:.2f}, val F1={best_f1:.4f}')

    thresholds_dict = {gene: thresh for gene, thresh in zip(clean_possible_genes, optimal_thresholds)}
    with open(f'results/optimal_thresholds_v{SAVE_AS_VERSION}.json', 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f'Optimal thresholds saved to results/optimal_thresholds_v{SAVE_AS_VERSION}.json')

    # Evaluate test set with optimal thresholds
    all_test_preds, all_test_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = efficientnet(inputs)
            all_test_preds.append(torch.sigmoid(outputs).cpu())
            all_test_targets.append(targets)

    all_test_preds = torch.cat(all_test_preds, dim=0).numpy()
    all_test_targets = torch.cat(all_test_targets, dim=0).numpy()

    preds_optimized = np.zeros_like(all_test_preds)
    for i, thresh in enumerate(optimal_thresholds):
        preds_optimized[:, i] = (all_test_preds[:, i] >= thresh).astype(float)

    print('\n=== Test set classification report (optimized thresholds) ===')
    print(classification_report(all_test_targets, preds_optimized,
                                target_names=clean_possible_genes, zero_division=0))

    preds_fixed = (all_test_preds >= 0.5).astype(float)
    print('=== Test set classification report (fixed 0.5 threshold) ===')
    print(classification_report(all_test_targets, preds_fixed,
                                target_names=clean_possible_genes, zero_division=0))
