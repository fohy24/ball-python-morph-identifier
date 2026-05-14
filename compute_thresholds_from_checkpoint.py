"""
Compute optimal per-class thresholds from a saved checkpoint.

Usage:
    python compute_thresholds_from_checkpoint.py --checkpoint model/model_v1_epoch3.pt

Reproduces the exact same train/val split as the training script
(GroupShuffleSplit with random_state=123) so the validation set
matches what the model was trained against.
"""

import argparse
import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Must match training script
SUBSET = True
IMAGE_SIZE = 384
BATCH_SIZE = 16
RANDOM_STATE = 123
SUBSET_FRAC = 0.005
TRAIN_TEST_SPLIT = 0.85
TRAIN_VAL_SPLIT = 0.8


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


def build_model(num_labels):
    efficientnet = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')
    efficientnet.classifier = nn.Sequential(
        nn.LazyLinear(2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.LazyLinear(num_labels),
    )
    return efficientnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--csv', default='data/top37_gene_df.csv')
    parser.add_argument('--img-dir', default='data/img/')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: results/optimal_thresholds_<checkpoint_stem>.json)')
    parser.add_argument('--no-subset', action='store_true',
                        help='Disable SUBSET sampling (use full dataset)')
    args = parser.parse_args()

    use_subset = not args.no_subset and SUBSET

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output = f'results/optimal_thresholds_{stem}.json'
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # --- Reproduce splits exactly ---
    labels_extended = pd.read_csv(args.csv)
    if use_subset:
        labels_extended = labels_extended.sample(frac=SUBSET_FRAC, random_state=RANDOM_STATE)

    clean_possible_genes = labels_extended.columns.to_list()[7:]
    num_labels = len(clean_possible_genes)
    print(f'Number of labels: {num_labels}')

    listing_ids = labels_extended['index'].astype(str).str.split('-').str[0]
    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE)
    train_idx, _ = next(gss.split(labels_extended, groups=listing_ids))
    train_df = labels_extended.iloc[train_idx].reset_index(drop=True)

    train_listing_ids = train_df['index'].astype(str).str.split('-').str[0]
    gss_val = GroupShuffleSplit(n_splits=1, train_size=TRAIN_VAL_SPLIT, random_state=RANDOM_STATE)
    _, valid_sub_idx = next(gss_val.split(train_df, groups=train_listing_ids))

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.6007, 0.5679, 0.5206], std=[0.2411, 0.2392, 0.2479]),
    ])

    valid_dataset = PythonGeneDataset(
        labels_df=train_df, img_dir=args.img_dir,
        indices=valid_sub_idx, transform=eval_transform,
    )
    print(f'Valid size: {len(valid_dataset)}')

    num_workers = multiprocessing.cpu_count()
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # --- Load model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')

    efficientnet = build_model(num_labels)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Materialize LazyLinear with a dummy forward pass before loading state_dict
    efficientnet.to(device).eval()
    with torch.no_grad():
        _ = efficientnet(torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE, device=device))
    efficientnet.load_state_dict(checkpoint['model_state_dict'])
    efficientnet.eval()
    print(f'Loaded checkpoint: {args.checkpoint} '
          f'(epoch {checkpoint.get("epoch", "?")}, '
          f'phase {checkpoint.get("phase", "?")}, '
          f'valid_mAP {checkpoint.get("valid_mAP", float("nan")):.4f})')

    # --- Predict on validation set ---
    all_val_preds, all_val_targets = [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            outputs = efficientnet(inputs)
            all_val_preds.append(torch.sigmoid(outputs).cpu())
            all_val_targets.append(targets)

    all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
    all_val_targets = torch.cat(all_val_targets, dim=0).numpy()

    # --- Optimize thresholds ---
    print('\n=== Optimizing per-class thresholds on validation set ===')
    optimal_thresholds = []
    for i in range(num_labels):
        best_thresh, best_f1 = 0.5, 0.0
        for thresh in np.arange(0.1, 0.91, 0.05):
            preds_binary = (all_val_preds[:, i] >= thresh).astype(float)
            f1 = f1_score(all_val_targets[:, i], preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds.append(float(best_thresh))
        print(f'  {clean_possible_genes[i]}: threshold={best_thresh:.2f}, val F1={best_f1:.4f}')

    thresholds_dict = {gene: thresh for gene, thresh in zip(clean_possible_genes, optimal_thresholds)}
    with open(args.output, 'w') as f:
        json.dump(thresholds_dict, f, indent=2)
    print(f'\nOptimal thresholds saved to {args.output}')


if __name__ == '__main__':
    main()
