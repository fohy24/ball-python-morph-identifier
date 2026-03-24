from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Data
    data_csv: str = 'data/selected_gene_df.csv'
    img_dir: str = 'data/img/'
    subset_frac: float | None = None  # Set to e.g. 0.01 for quick testing

    # Image
    image_size: int = 480

    # Training
    batch_size: int = 36
    head_warmup_epochs: int = 4
    head_warmup_lr: float = 1e-3
    finetune_epochs: int = 14
    finetune_lr: float = 4e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Early stopping
    early_stopping_patience: int = 5

    # Focal loss
    focal_gamma: float = 2.0

    # Checkpointing
    save_checkpoint: bool = True
    save_as_version: str = "14"
    load_checkpoint: bool = False
    checkpoint_version: int = 1
    checkpoint_epoch: int = 20

    # Splits
    train_size: float = 0.85
    train_val_split: float = 0.8
    seed: int = 123
    val_seed: int = 42
