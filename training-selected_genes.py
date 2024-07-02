import pandas as pd
import numpy as np
import ast

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn, optim
from torchvision import datasets, utils, models
# from torchinfo import summary
import torch.nn.functional as F
from torchvision.transforms import v2
import multiprocessing

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import altair as alt
alt.data_transformers.enable("vegafusion")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

#########################################################################

SUBSET = False
IMAGE_SIZE = 256
BATCH_SIZE = 256
LEARNING_RATE = 0.0002

LOAD_CHECKPOINT = False
checkpoint_version = 1
checkpoint_epoch = 20
num_additional_epochs = 30

SAVE_CHECKPOINT = False
SAVE_AS_VERSION = 6

#########################################################################

# Change csv name as needed
labels_extended = pd.read_csv('data/selected_gene_df.csv')

if SUBSET:
    labels_extended = labels_extended.sample(frac=0.01, random_state=123)

clean_possible_genes = labels_extended.columns.to_list()[7:]

train_df, test_df = train_test_split(labels_extended, train_size=0.85, random_state=123)
print(f'Shape of trainset: {train_df.shape}')
print(f'Shape of testset: {test_df.shape}')


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
        # Parse labels here based on your CSV structure and required format
        labels = torch.tensor(self.labels_df.iloc[idx, 7:].astype('float32').values)
        
        if self.transform:
            image = self.transform(image)

        return image, labels



transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True), # Normalize expects float input
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

full_dataset = PythonGeneDataset(labels_df=train_df, img_dir='data/img/', transform=transform)

# Split dataset
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
valid_size = total_size - train_size
train_indices, valid_indices = torch.utils.data.random_split(np.arange(total_size), [train_size, valid_size])

# Create train and validation datasets
train_dataset = Subset(full_dataset, train_indices)
valid_dataset = Subset(full_dataset, valid_indices)

# Initialize DataLoaders
print(f'Number of workers: {multiprocessing.cpu_count()}')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True)

densenet = models.densenet201(weights='DenseNet201_Weights.DEFAULT')

for param in densenet.parameters():  # Freeze parameters so we don't update them
    param.requires_grad = False

for name, child in densenet.named_children():
    if name == 'features':
        for child_name, parameters in child.named_children():
            if 'denseblock2' in child_name or 'denseblock3' in child_name or 'denseblock4' in child_name:
                for param in parameters.parameters():
                    param.requires_grad = True



num_labels = len(clean_possible_genes)
print(f'Number of labels: {num_labels}')

new_layers = nn.Sequential(
    nn.Linear(1920, 1000),  # Reduce dimension from 1024 to 500
    nn.BatchNorm1d(1000),   # Normalize the activations from the previous layer
    nn.ReLU(),             # Non-linear activation function
    nn.Dropout(0.5),       # Dropout for regularization (50% probability)
    nn.Linear(1000, num_labels)  # Final layer for class predictions
)
densenet.classifier = new_layers

# Implement focal loss for label imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability is 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')
densenet.to(device)

focal_loss = FocalLoss()
criterion = focal_loss

# criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=LEARNING_RATE)

def train_model(model, criterion, optimizer, start_epoch, total_epochs, version=1, save_checkpoint=True):
    """
    Train a model with specified criterion and optimizer. 

    Parameters:
    - model (torch.nn.Module): The neural network model to be trained.
    - criterion (torch.nn.Module): The loss function used for the training.
    - optimizer (torch.optim.Optimizer): The optimizer used for parameter updates.
    - start_epoch (int): The starting epoch number for training. Useful for resuming training. Set to 1 if starting from scratch.
    - total_epochs (int): The total number of epochs to train the model for.
    - version (int, optional): Version number of the model checkpoint files. Default is 1. Recommend to specify when save_checkpoint=True.
    - save_checkpoint (bool, optional): Flag to control whether to save checkpoints after each epoch. Default is True.

    Outputs:
    - The function does not return any values but prints training and validation loss after each epoch and saves model checkpoints if specified.

    Example:
    ```
    # Assuming criterion and optimizer are predefined, this will train the densenet model for 10 epochs, starting from epoch 21.
    # This is useful when you have checkpoint models up until epoch 20.
    # Checkpoints will be saved and named as version 2.
    train_model(densenet, criterion, optimizer,
            start_epoch=21,
            total_epochs=30,
            version=2,
            save_checkpoint=True)
    ```

    Note:
    - Ensure that the device (CPU or GPU) is appropriately set for the model, criterion, and data tensors.
    - The global variables BATCH_SIZE, IMAGE_SIZE, train_loader, and valid_loader are used within this function and should be defined in the scope where this function is called.
    """
    print(f'Start training - batch size: {BATCH_SIZE} & image size: {IMAGE_SIZE}')
    
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()  # Set model to training mode

        with tqdm(train_loader, unit="batch") as tepoch:
            train_loss = 0.0
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{total_epochs}")
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # Calculate average loss for the epoch
            train_loss = train_loss / len(train_loader.dataset)

            # Validation of the model
            model.eval()  # Set model to evaluate mode
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)

            # Calculate average loss over validation data
            valid_loss = valid_loss / len(valid_loader.dataset)

            # Checkpoint
            if save_checkpoint:
                CHECKPOINT_PATH = f'model/model_v{version}_epoch{epoch}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'image_size': IMAGE_SIZE,
                    'batch_size': BATCH_SIZE,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    }, CHECKPOINT_PATH)

            # Print training/validation statistics
            print(f'Epoch {epoch}/{total_epochs}, Train Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}')
            tepoch.set_postfix(train_loss=train_loss,
                               valid_loss=valid_loss)


def get_epoch(LOAD_CHECKPOINT, num_additional_epochs, checkpoint_version, checkpoint_epoch):
    if LOAD_CHECKPOINT:
        checkpoint_path = f'model/full_dataset_v{checkpoint_version}_epoch{checkpoint_epoch}.pt'
        checkpoint = torch.load(checkpoint_path)
        print(f'Loaded version {checkpoint_version}-epoch {checkpoint_epoch}')

        densenet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint_epoch + 1
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']

        print(f'Checkpoint Train Loss: {train_loss:.5f} & Valid Loss: {valid_loss:.5f}')
    else:
        start_epoch = 1  # Start from scratch if not loading from checkpoint

    total_epochs = start_epoch + num_additional_epochs - 1
    return start_epoch, total_epochs


start_epoch, total_epochs = get_epoch(LOAD_CHECKPOINT, num_additional_epochs, checkpoint_version, checkpoint_epoch)
print(f'Starting at Epoch: {start_epoch}, Ending at Epoch: {total_epochs}')


train_model(densenet, criterion, optimizer,
            start_epoch=start_epoch,
            total_epochs=total_epochs,
            version=SAVE_AS_VERSION,
            save_checkpoint=SAVE_CHECKPOINT)

# Loss on test set
test_dataset = PythonGeneDataset(labels_df=test_df, img_dir='data/img/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

densenet.eval()  # Set model to evaluate mode

test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = densenet(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

# Calculate average loss over validation data
test_loss = test_loss / len(test_loader.dataset)
print(f'Loss on Test set: {test_loss:.5f}')


