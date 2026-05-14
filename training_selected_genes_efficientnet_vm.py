import pandas as pd
import numpy as np
import ast

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn, optim
from torchvision import datasets, utils, models
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import torch.optim as optim
from torchvision.transforms import v2
import multiprocessing

from copy import deepcopy
from torch.utils.checkpoint import checkpoint_sequential
from torch.optim.swa_utils import AveragedModel, update_bn

from sklearn.model_selection import GroupShuffleSplit, train_test_split
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import os
import json
from tqdm import tqdm

import gcsfs

ImageFile.LOAD_TRUNCATED_IMAGES = True

#########################################################################
# Constants
#########################################################################

SUBSET = False
IMAGE_SIZE = 480
BATCH_SIZE = 32
LEARNING_RATE = 0.00004

LOAD_CHECKPOINT = False
checkpoint_version = 1
checkpoint_epoch = 20

HEAD_WARMUP_EPOCHS = 4
HEAD_WARMUP_LR = 1e-3
FINETUNE_EPOCHS = 24
EARLY_STOPPING_PATIENCE = 5

SAVE_CHECKPOINT = True
SAVE_AS_VERSION = "2"

GCP_PROJECT = 'beaming-opus-494700-q4'
GCS_MODELS = 'gs://bp-scraper-images/models'
GCS_RESULTS = 'gs://bp-scraper-images/results'
DATA_ROOT = '/mnt/data'

# Tier 1 / Tier 2 knobs
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
USE_BF16 = True
USE_CHANNELS_LAST = True
USE_GRAD_CKPT = True
GRAD_CKPT_SEGMENTS = 4
USE_COMPILE = True
COMPILE_MODE = 'default'
EMA_DECAY = 0.9998
MIXUP_PROB = 0.5
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
TTA_HFLIP_FINAL = True
AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

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
        image = Image.open(img_name).convert('RGB')
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


class AsymmetricLoss(nn.Module):
    """Ridnik et al. 2021. Multi-label classification."""

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05,
                 eps=1e-8, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        x_sigmoid = torch.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1.0 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            one_sided_w = one_sided_w.detach()
            loss = loss * one_sided_w

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = alpha * loss

        loss = -loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def _rand_bbox(size, lam):
    _, _, H, W = size
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_or_cutmix(images, targets, p=0.5, mixup_alpha=0.2, cutmix_alpha=1.0):
    """Multi-label-safe mixup/cutmix. With prob p, apply EITHER mixup OR cutmix (50/50)."""
    if np.random.rand() > p:
        return images, targets
    perm = torch.randperm(images.size(0), device=images.device)
    if np.random.rand() < 0.5:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        images = lam * images + (1.0 - lam) * images[perm]
        targets = lam * targets + (1.0 - lam) * targets[perm]
    else:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        x1, y1, x2, y2 = _rand_bbox(images.size(), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        actual_lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        targets = actual_lam * targets + (1.0 - actual_lam) * targets[perm]
    return images, targets


def build_param_groups(named_params, lr, weight_decay=0.01):
    """Split params into decay (weight tensors with ndim>=2) vs no-decay (BN affine + biases)."""
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay,    'lr': lr, 'weight_decay': weight_decay},
        {'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
    ]


class _GradCkptForwardMixin:
    """Mixin that overrides EfficientNet._forward_impl to run `features` through
    checkpoint_sequential when training and at least one feature param needs grad.
    Implemented as a class mixin (rather than a closure-based monkey patch) so that
    AveragedModel's deepcopy produces an EMA module whose forward correctly resolves
    `self.features`/`avgpool`/`classifier` to its own deep-copied submodules."""
    _grad_ckpt_segments = 4

    def _forward_impl(self, x):
        use_ckpt = (
            self.training
            and USE_GRAD_CKPT
            and any(p.requires_grad for p in self.features.parameters())
        )
        if use_ckpt:
            x = checkpoint_sequential(
                self.features, self._grad_ckpt_segments, x, use_reentrant=False
            )
        else:
            x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _enable_grad_ckpt(model, segments=4):
    """Enable grad checkpointing on `model.features` via class-level mixin.
    Mutates `model.__class__` to a dynamic subclass that inherits from the mixin."""
    parent = type(model)
    new_cls = type(parent.__name__ + 'GradCkpt', (_GradCkptForwardMixin, parent), {})
    model.__class__ = new_cls
    model._grad_ckpt_segments = segments
    return model


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


def evaluate(model, data_loader, device, num_labels, criterion,
             autocast_dtype=torch.bfloat16, tta_hflip=False,
             channels_last=True):
    """Single-pass eval: returns (avg_loss, mAP, per_class_AP)."""
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    n_seen = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            if channels_last:
                inputs = inputs.to(device, non_blocking=True,
                                   memory_format=torch.channels_last)
            else:
                inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast('cuda', dtype=autocast_dtype):
                logits = model(inputs)
                loss = criterion(logits, targets)
                if tta_hflip:
                    logits_flip = model(torch.flip(inputs, dims=[3]))
                    probs = 0.5 * (torch.sigmoid(logits) + torch.sigmoid(logits_flip))
                else:
                    probs = torch.sigmoid(logits)
            total_loss += loss.item() * inputs.size(0)
            n_seen += inputs.size(0)
            all_preds.append(probs.float().cpu())
            all_targets.append(targets.float().cpu())

    avg_loss = total_loss / max(n_seen, 1)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    APs = []
    for i in range(num_labels):
        cp = all_preds[:, i]
        ct = all_targets[:, i]
        if torch.sum(ct) == 0:
            continue
        order = torch.argsort(cp, descending=True)
        st = ct[order]
        tp = torch.cumsum(st, 0).float()
        fp = torch.cumsum((st == 0), 0).float()
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (torch.sum(ct) + 1e-10)
        precision = torch.cat([torch.tensor([1.0]), precision])
        recall = torch.cat([torch.tensor([0.0]), recall])
        for j in range(len(precision) - 1, 0, -1):
            precision[j-1] = max(precision[j-1], precision[j])
        AP = 0.0
        for j in range(len(precision) - 1):
            AP += (recall[j+1] - recall[j]) * precision[j+1]
        APs.append(AP.item())

    mAP = sum(APs) / max(len(APs), 1) if APs else 0.0
    return avg_loss, mAP, APs


def strip_compile_prefix(state_dict):
    """torch.compile saves state dict keys with `_orig_mod.` prefix; strip for portability."""
    return {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}


#########################################################################
# Main — everything below only runs in the main process
#########################################################################

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    fs = gcsfs.GCSFileSystem(project=GCP_PROJECT)

    # --- Data loading ---
    labels_extended = pd.read_csv(f'{DATA_ROOT}/top37_gene_df.csv')
    if SUBSET:
        labels_extended = labels_extended.sample(frac=0.005, random_state=123)

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
        v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.6007, 0.5679, 0.5206], std=[0.2411, 0.2392, 0.2479]),
        v2.RandomErasing(p=0.2),
    ])

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(int(IMAGE_SIZE * 1.14), antialias=True),
        v2.CenterCrop(IMAGE_SIZE),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.6007, 0.5679, 0.5206], std=[0.2411, 0.2392, 0.2479]),
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')

    # --- Group-based train/val split ---
    train_listing_ids = train_df['index'].astype(str).str.split('-').str[0]
    gss_val = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=123)
    train_sub_idx, valid_sub_idx = next(gss_val.split(train_df, groups=train_listing_ids))

    train_dataset = PythonGeneDataset(labels_df=train_df, img_dir=f'{DATA_ROOT}/img/',
                                      indices=train_sub_idx, transform=train_transform)
    valid_dataset = PythonGeneDataset(labels_df=train_df, img_dir=f'{DATA_ROOT}/img/',
                                      indices=valid_sub_idx, transform=eval_transform)
    print(f'Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}')

    print(f'Number of workers: {NUM_WORKERS}')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)

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
    efficientnet.to(device)

    # Materialize LazyLinear params (required before AveragedModel deepcopy + torch.compile)
    efficientnet.eval()
    with torch.no_grad():
        _dummy = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        _ = efficientnet(_dummy)
    print(f'Lazy materialized -- classifier[0].weight: {efficientnet.classifier[0].weight.shape}')

    if USE_CHANNELS_LAST:
        efficientnet = efficientnet.to(memory_format=torch.channels_last)

    if USE_GRAD_CKPT:
        efficientnet = _enable_grad_ckpt(efficientnet, segments=GRAD_CKPT_SEGMENTS)

    # EMA wraps the uncompiled model so AveragedModel.deepcopy is clean.
    # The class-level _forward_impl on _GradCkptForwardMixin survives deepcopy:
    # ema_model.module.__class__ inherits the mixin, so its forward correctly
    # resolves self.features / self.avgpool / self.classifier to its own submodules.
    ema_model = AveragedModel(
        efficientnet,
        avg_fn=lambda avg, new, num: EMA_DECAY * avg + (1.0 - EMA_DECAY) * new,
    )

    if USE_COMPILE:
        efficientnet = torch.compile(efficientnet, mode=COMPILE_MODE)

    # --- Per-class alpha from inverse frequency ---
    pos_counts = train_df.iloc[:, 7:].sum(axis=0).values.astype(np.float32)
    pos_counts = np.clip(pos_counts, 1.0, None)
    inv_freq = 1.0 / pos_counts
    alpha_per_class = torch.tensor(inv_freq / inv_freq.sum() * len(inv_freq), dtype=torch.float32)
    print(f'Per-class alpha range: [{alpha_per_class.min():.3f}, {alpha_per_class.max():.3f}]')

    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0, clip=0.05, alpha=alpha_per_class)

    # =====================================================
    # Phase 1: Freeze backbone, train classifier head only
    # =====================================================
    print('\n=== Phase 1: Training classifier head (backbone frozen) ===')
    underlying = efficientnet._orig_mod if USE_COMPILE else efficientnet
    for param in underlying.features.parameters():
        param.requires_grad = False
    for param in underlying.classifier.parameters():
        param.requires_grad = True

    head_param_groups = build_param_groups(
        underlying.classifier.named_parameters(),
        lr=HEAD_WARMUP_LR, weight_decay=0.01,
    )
    head_optimizer = torch.optim.AdamW(head_param_groups)

    result = {
        "epoch": [], "train_loss": [],
        "valid_loss": [], "valid_mAP": [],
        "valid_loss_ema": [], "valid_mAP_ema": [],
        "lr": [], "phase": [],
    }

    best_live_mAP = 0.0
    best_ema_mAP = 0.0
    best_ema_epoch = -1
    patience_counter = 0

    for epoch in range(1, HEAD_WARMUP_EPOCHS + 1):
        efficientnet.train()
        train_loss = 0.0
        n_seen = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Phase 1 - Epoch {epoch}/{HEAD_WARMUP_EPOCHS}")
                if USE_CHANNELS_LAST:
                    inputs = inputs.to(device, non_blocking=True,
                                       memory_format=torch.channels_last)
                else:
                    inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                head_optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', dtype=AUTOCAST_DTYPE):
                    outputs = efficientnet(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(efficientnet.parameters(), max_norm=1.0)
                head_optimizer.step()

                ema_model.update_parameters(efficientnet)

                train_loss += loss.item() * inputs.size(0)
                n_seen += inputs.size(0)

        train_loss = train_loss / max(n_seen, 1)

        # Validation -- live + EMA
        valid_loss, valid_mAP, _ = evaluate(
            efficientnet, valid_loader, device, num_labels, criterion,
            autocast_dtype=AUTOCAST_DTYPE, tta_hflip=False,
            channels_last=USE_CHANNELS_LAST,
        )
        ema_valid_loss, ema_valid_mAP, _ = evaluate(
            ema_model, valid_loader, device, num_labels, criterion,
            autocast_dtype=AUTOCAST_DTYPE, tta_hflip=False,
            channels_last=USE_CHANNELS_LAST,
        )

        result["epoch"].append(epoch)
        result["train_loss"].append(train_loss)
        result["valid_loss"].append(valid_loss)
        result["valid_mAP"].append(valid_mAP)
        result["valid_loss_ema"].append(ema_valid_loss)
        result["valid_mAP_ema"].append(ema_valid_mAP)
        result["lr"].append(HEAD_WARMUP_LR)
        result["phase"].append(1)

        print(f'Phase 1 Epoch {epoch}/{HEAD_WARMUP_EPOCHS}, '
              f'Train Loss: {train_loss:.6f}, '
              f'Valid Loss: {valid_loss:.6f}, Valid mAP: {valid_mAP:.4f}, '
              f'EMA Loss: {ema_valid_loss:.6f}, EMA mAP: {ema_valid_mAP:.4f}')

        if SAVE_CHECKPOINT:
            live_sd = strip_compile_prefix(efficientnet.state_dict())
            ema_sd = strip_compile_prefix(ema_model.module.state_dict())
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': live_sd,
                'ema_state_dict': ema_sd,
                'optimizer_state_dict': head_optimizer.state_dict(),
                'image_size': IMAGE_SIZE, 'batch_size': BATCH_SIZE,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 'valid_mAP': valid_mAP,
                'ema_valid_loss': ema_valid_loss, 'ema_valid_mAP': ema_valid_mAP,
                'phase': 1,
            }
            with fs.open(f'{GCS_MODELS}/model_v{SAVE_AS_VERSION}_epoch{epoch}.pt', 'wb') as f:
                torch.save(checkpoint, f)

        if ema_valid_mAP > best_ema_mAP:
            best_ema_mAP = ema_valid_mAP
            best_ema_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if valid_mAP > best_live_mAP:
            best_live_mAP = valid_mAP

    # =====================================================
    # Phase 2: Unfreeze all, discriminative learning rates
    # =====================================================
    print('\n=== Phase 2: Fine-tuning full model (discriminative LRs) ===')
    underlying = efficientnet._orig_mod if USE_COMPILE else efficientnet
    for param in underlying.parameters():
        param.requires_grad = True

    p_low_groups = build_param_groups(
        underlying.features[:4].named_parameters(),
        lr=LEARNING_RATE * 0.1,
    )
    p_mid_groups = build_param_groups(
        underlying.features[4:].named_parameters(),
        lr=LEARNING_RATE * 0.5,
    )
    p_head_groups = build_param_groups(
        underlying.classifier.named_parameters(),
        lr=LEARNING_RATE * 2.5,
    )
    param_groups = p_low_groups + p_mid_groups + p_head_groups

    finetune_optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    warmup_scheduler = LinearLR(finetune_optimizer, start_factor=0.1, total_iters=2)
    cosine_scheduler = CosineAnnealingLR(finetune_optimizer, T_max=FINETUNE_EPOCHS - 2)
    scheduler = SequentialLR(finetune_optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[2])

    patience_counter = 0

    for epoch_offset in range(1, FINETUNE_EPOCHS + 1):
        epoch = HEAD_WARMUP_EPOCHS + epoch_offset
        efficientnet.train()
        train_loss = 0.0
        n_seen = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            for inputs, labels in tepoch:
                tepoch.set_description(f"Phase 2 - Epoch {epoch}/{HEAD_WARMUP_EPOCHS + FINETUNE_EPOCHS}")
                if USE_CHANNELS_LAST:
                    inputs = inputs.to(device, non_blocking=True,
                                       memory_format=torch.channels_last)
                else:
                    inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                inputs, labels = mixup_or_cutmix(
                    inputs, labels,
                    p=MIXUP_PROB, mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA,
                )

                finetune_optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', dtype=AUTOCAST_DTYPE):
                    outputs = efficientnet(inputs)
                    loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(efficientnet.parameters(), max_norm=1.0)
                finetune_optimizer.step()

                ema_model.update_parameters(efficientnet)

                train_loss += loss.item() * inputs.size(0)
                n_seen += inputs.size(0)

        train_loss = train_loss / max(n_seen, 1)

        # Validation -- live + EMA
        valid_loss, valid_mAP, _ = evaluate(
            efficientnet, valid_loader, device, num_labels, criterion,
            autocast_dtype=AUTOCAST_DTYPE, tta_hflip=False,
            channels_last=USE_CHANNELS_LAST,
        )
        ema_valid_loss, ema_valid_mAP, _ = evaluate(
            ema_model, valid_loader, device, num_labels, criterion,
            autocast_dtype=AUTOCAST_DTYPE, tta_hflip=False,
            channels_last=USE_CHANNELS_LAST,
        )
        scheduler.step()

        current_lrs = [pg['lr'] for pg in finetune_optimizer.param_groups]
        result["epoch"].append(epoch)
        result["train_loss"].append(train_loss)
        result["valid_loss"].append(valid_loss)
        result["valid_mAP"].append(valid_mAP)
        result["valid_loss_ema"].append(ema_valid_loss)
        result["valid_mAP_ema"].append(ema_valid_mAP)
        # Log mid-segment decay-group LR (index 2 in 6-group layout)
        result["lr"].append(current_lrs[2])
        result["phase"].append(2)

        print(f'Phase 2 Epoch {epoch}/{HEAD_WARMUP_EPOCHS + FINETUNE_EPOCHS}, '
              f'Train Loss: {train_loss:.6f}, '
              f'Valid Loss: {valid_loss:.6f}, Valid mAP: {valid_mAP:.4f}, '
              f'EMA Loss: {ema_valid_loss:.6f}, EMA mAP: {ema_valid_mAP:.4f}, '
              f'LRs: {[f"{lr:.2e}" for lr in current_lrs]}')

        if SAVE_CHECKPOINT:
            live_sd = strip_compile_prefix(efficientnet.state_dict())
            ema_sd = strip_compile_prefix(ema_model.module.state_dict())
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': live_sd,
                'ema_state_dict': ema_sd,
                'optimizer_state_dict': finetune_optimizer.state_dict(),
                'image_size': IMAGE_SIZE, 'batch_size': BATCH_SIZE,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 'valid_mAP': valid_mAP,
                'ema_valid_loss': ema_valid_loss, 'ema_valid_mAP': ema_valid_mAP,
                'phase': 2,
            }
            with fs.open(f'{GCS_MODELS}/model_v{SAVE_AS_VERSION}_epoch{epoch}.pt', 'wb') as f:
                torch.save(checkpoint, f)

        if ema_valid_mAP > best_ema_mAP:
            best_ema_mAP = ema_valid_mAP
            best_ema_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f'Early stopping triggered after {EARLY_STOPPING_PATIENCE} '
                      f'epochs without EMA mAP improvement.')
                break
        if valid_mAP > best_live_mAP:
            best_live_mAP = valid_mAP

        with fs.open(
            f'{GCS_RESULTS}/version{SAVE_AS_VERSION}_ImageSize{IMAGE_SIZE}_LR{LEARNING_RATE}.csv',
            'w',
        ) as f:
            pd.DataFrame(result).to_csv(f, index=False)

    # =====================================================
    # Evaluate on test set using best EMA model
    # =====================================================
    best_epoch_idx = int(np.argmax(result["valid_mAP_ema"]))
    best_model_epoch = result["epoch"][best_epoch_idx]
    best_model_path = f'{GCS_MODELS}/model_v{SAVE_AS_VERSION}_epoch{best_model_epoch}.pt'
    print(f'\nBest (EMA) model: epoch {best_model_epoch} '
          f'with EMA valid mAP: {result["valid_mAP_ema"][best_epoch_idx]:.4f}')

    with fs.open(best_model_path, 'rb') as f:
        best_checkpoint = torch.load(f, map_location=device)

    # Rebuild fresh, uncompiled, no-grad-ckpt model and load EMA weights.
    final_model = models.efficientnet_v2_l(weights=None)
    final_model.classifier = nn.Sequential(
        nn.LazyLinear(2048), nn.BatchNorm1d(2048), nn.ReLU(),
        nn.Dropout(0.5), nn.LazyLinear(num_labels)
    )
    final_model.to(device)
    with torch.no_grad():
        _ = final_model(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    final_model.load_state_dict(best_checkpoint['ema_state_dict'])
    if USE_CHANNELS_LAST:
        final_model = final_model.to(memory_format=torch.channels_last)
    final_model.eval()

    # Recompute BN running stats for EMA on a no-aug pass over the train subset.
    print('Recomputing BatchNorm running stats for EMA model...')
    bn_dataset = PythonGeneDataset(
        labels_df=train_df, img_dir=f'{DATA_ROOT}/img/',
        indices=train_sub_idx, transform=eval_transform,
    )
    bn_loader = DataLoader(
        bn_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    update_bn(bn_loader, final_model, device=device)

    test_dataset = PythonGeneDataset(labels_df=test_df, img_dir=f'{DATA_ROOT}/img/', transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    test_loss, test_mAP, test_class_APs = evaluate(
        final_model, test_loader, device, num_labels, criterion,
        autocast_dtype=AUTOCAST_DTYPE, tta_hflip=TTA_HFLIP_FINAL,
        channels_last=USE_CHANNELS_LAST,
    )
    print(f'\nTest set mAP (EMA + TTA): {test_mAP:.4f}')
    print(f'Test Loss: {test_loss:.6f}')
    print('\nPer-class Average Precision:')
    for gene, ap in zip(clean_possible_genes, test_class_APs):
        print(f'  {gene}: {ap:.4f}')

    # --- Per-class threshold optimization on validation set (TTA-averaged) ---
    print('\n=== Optimizing per-class thresholds on validation set ===')
    from sklearn.metrics import f1_score, classification_report

    final_model.eval()
    all_val_preds, all_val_targets = [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            if USE_CHANNELS_LAST:
                inputs = inputs.to(device, non_blocking=True,
                                   memory_format=torch.channels_last)
            else:
                inputs = inputs.to(device, non_blocking=True)
            with autocast('cuda', dtype=AUTOCAST_DTYPE):
                logits = final_model(inputs)
                if TTA_HFLIP_FINAL:
                    logits_flip = final_model(torch.flip(inputs, dims=[3]))
                    probs = 0.5 * (torch.sigmoid(logits) + torch.sigmoid(logits_flip))
                else:
                    probs = torch.sigmoid(logits)
            all_val_preds.append(probs.float().cpu())
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
    thresholds_path = f'{GCS_RESULTS}/optimal_thresholds_v{SAVE_AS_VERSION}.json'
    with fs.open(thresholds_path, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f'Optimal thresholds saved to {thresholds_path}')

    # --- Test set predictions (TTA-averaged) for classification reports ---
    all_test_preds, all_test_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            if USE_CHANNELS_LAST:
                inputs = inputs.to(device, non_blocking=True,
                                   memory_format=torch.channels_last)
            else:
                inputs = inputs.to(device, non_blocking=True)
            with autocast('cuda', dtype=AUTOCAST_DTYPE):
                logits = final_model(inputs)
                if TTA_HFLIP_FINAL:
                    logits_flip = final_model(torch.flip(inputs, dims=[3]))
                    probs = 0.5 * (torch.sigmoid(logits) + torch.sigmoid(logits_flip))
                else:
                    probs = torch.sigmoid(logits)
            all_test_preds.append(probs.float().cpu())
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
