import torch
import numpy as np
from torch.amp import autocast


def compute_mAP(model, data_loader, device, num_labels):
    """Compute Mean Average Precision (VOC-style all-point interpolation)."""
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
            precision[j - 1] = max(precision[j - 1], precision[j])

        AP = 0.0
        for j in range(len(precision) - 1):
            AP += (recall[j + 1] - recall[j]) * precision[j + 1]

        APs.append(AP.item())

    if len(APs) == 0:
        return 0.0, []

    mAP = sum(APs) / len(APs)
    return mAP, APs


def optimize_thresholds(all_preds, all_targets, num_labels):
    """Find optimal per-class thresholds by sweeping to maximize F1."""
    from sklearn.metrics import f1_score

    optimal_thresholds = []
    for i in range(num_labels):
        best_thresh = 0.5
        best_f1 = 0.0
        for thresh in np.arange(0.1, 0.91, 0.05):
            preds_binary = (all_preds[:, i] >= thresh).astype(float)
            f1 = f1_score(all_targets[:, i], preds_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)

    return optimal_thresholds
