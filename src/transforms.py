import torch
from torchvision.transforms import v2

# Dataset-specific normalization (computed from ball python images)
DATASET_MEAN = [0.6007, 0.5679, 0.5206]
DATASET_STD = [0.2411, 0.2392, 0.2479]


def get_train_transform(image_size):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        v2.RandomErasing(p=0.2),
    ])


def get_eval_transform(image_size):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])


def get_tta_hflip_transform(image_size):
    """Transform that always applies horizontal flip (for TTA)."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size)),
        v2.RandomHorizontalFlip(p=1.0),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
    ])
