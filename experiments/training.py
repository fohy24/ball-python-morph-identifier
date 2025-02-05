# %%
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

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
import altair as alt
alt.data_transformers.enable("vegafusion")

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# %% [markdown]
# ## Load Data

# %%
labels_extended = pd.read_csv('data/labels_extended.csv')

# %%
SUBSET = True
if SUBSET:
    labels_extended = labels_extended.sample(frac=0.01, random_state=123)

# %%
clean_possible_genes = labels_extended.columns.to_list()[7:]

# %%
train_df, test_df = train_test_split(labels_extended, train_size=0.8, random_state=123)
print(f'Shape of trainset: {train_df.shape}')
print(f'Shape of testset: {test_df.shape}')

# %%
image = Image.open("data/img/10000-0.png")

# Convert the image to a tensor
transform_image = v2.ToImage()
tensor_image = transform_image(image)
tensor_image.shape

# %%
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


# %%
IMAGE_SIZE = 512

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
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %%
# class PythonGeneClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(PythonGeneClassifier, self).__init__()
#         # Increasing the complexity of the network
#         self.feature_extractor = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Dropout(0.25)  # Adding dropout for regularization
#         )

#         # Flatten layer is moved to the forward function
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 4 * 4, 512),  # Adjusted for 32x32 input images; calculate accordingly
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )

#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = x.view(x.size(0), -1)  # Flatten the tensor
#         x = self.classifier(x)
#         return x

# %% [markdown]
# ### Transfer Learning (DenseNet)

# %%
densenet = models.densenet201(weights='DenseNet201_Weights.DEFAULT')

for param in densenet.parameters():  # Freeze parameters so we don't update them
    param.requires_grad = False

# %%
densenet.classifier

# %%
num_labels = len(clean_possible_genes)
new_layers = nn.Sequential(
    nn.Linear(1920, 1000),  # Reduce dimension from 1024 to 500
    nn.BatchNorm1d(1000),   # Normalize the activations from the previous layer
    nn.ReLU(),             # Non-linear activation function
    nn.Dropout(0.5),       # Dropout for regularization (50% probability)
    nn.Linear(1000, num_labels)  # Final layer for class predictions
)
densenet.classifier = new_layers
densenet.classifier

# %%
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


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')
densenet.to(device)

focal_loss = FocalLoss()
criterion = focal_loss

# criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)

# %%
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


# %%
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

# %%
LOAD_CHECKPOINT = True
checkpoint_version = 1
checkpoint_epoch = 20
num_additional_epochs = 3

start_epoch, total_epochs = get_epoch(LOAD_CHECKPOINT, num_additional_epochs, checkpoint_version, checkpoint_epoch)
print(f'Starting at Epoch: {start_epoch}, Ending at Epoch: {total_epochs}')

# %%
train_model(densenet, criterion, optimizer,
            start_epoch=start_epoch,
            total_epochs=total_epochs,
            version=2,
            save_checkpoint=True)

# %% [markdown]
# ### CNN from scratch

# %%
# model = PythonGeneClassifier(num_classes=len(clean_possible_genes))
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters())

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Training on {device}')
# model.to(device)

# summary(model, input_size=(3, IMAGE_SIZE, IMAGE_SIZE))

# %%
# def trainer(model, criterion, optimizer, trainloader, validloader, epochs=20, patience=5, verbose=True):
#     """Simple training wrapper for PyTorch network."""
#     train_loss, valid_loss, valid_accuracy = [], [], []
#     consec_increases = 0
#     for epoch in range(epochs):  # for each epoch
#         train_batch_loss = 0
#         valid_batch_loss = 0
#         valid_batch_acc = 0
#         # Training
#         for X, y in trainloader:
#             X, y = X.to(device), y.to(device)
#             optimizer.zero_grad()
#             y_hat = model(X)
#             loss = criterion(y_hat, y)
#             loss.backward()
#             optimizer.step()
#             train_batch_loss += loss.item()
#         train_loss.append(train_batch_loss / len(trainloader))
#         # Validation
#         with torch.no_grad():
#             for X, y in validloader:
#                 X, y = X.to(device), y.to(device)
#                 y_hat = model(X)
#                 loss = criterion(y_hat, y)
#                 valid_batch_loss += loss.item()
#                 _, predicted = torch.max(y_hat.data, 1)
#                 valid_batch_acc += (predicted == y).sum().item() / y.size(0)
#         valid_loss.append(valid_batch_loss / len(validloader))
#         valid_accuracy.append(valid_batch_acc / len(validloader))  # accuracy
#         model.train()
#         # Print progress
#         if verbose:
#             print(f"Epoch {epoch + 1}:",
#                   f"Train Loss: {train_loss[-1]:.3f}.",
#                   f"Valid Loss: {valid_loss[-1]:.3f}.",
#                   f"Valid Accuracy: {valid_accuracy[-1]:.2f}.")
#         # Early stopping
#         if epoch > 0 and valid_loss[-1] > valid_loss[-2]:
#             consec_increases += 1
#         else:
#             consec_increases = 0
#         if consec_increases == patience:
#             print(f"Stopped early at epoch {epoch + 1:3}: val loss increased for {consec_increases} consecutive epochs!")
#             break
#     results = {"train_loss": train_loss, "valid_loss": valid_loss, "valid_accuracy": valid_accuracy}
#     return results

# %% [markdown]
# ## Prediction

# %%
# Prediction by image
img_code = '20112-0'
img = Image.open(f'data/img/{img_code}.png')
input_img = transform(img)
input_img = input_img.unsqueeze(0)
input_img.shape

# Load the checkpoint model
densenet = models.densenet201(weights='DenseNet201_Weights.DEFAULT')
densenet.classifier = new_layers
optimizer = torch.optim.Adam(densenet.parameters(), lr=0.001)

checkpoint = torch.load(f'model/full_dataset_v1_epoch20.pt')
densenet.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
train_loss = checkpoint['train_loss']
valid_loss = checkpoint['valid_loss']

densenet.eval()

# If using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
densenet.to(device)
input_img = input_img.to(device)

with torch.no_grad():
    output = densenet(input_img)

predicted_probs = torch.sigmoid(output).to('cpu')
prediction = pd.DataFrame(predicted_probs, index=['predictions'],
                          columns=clean_possible_genes).T.sort_values(by=['predictions'], ascending=False)

# True labels
print(labels_extended.query(f'index == "{img_code}"').genes.to_list())

prediction.head(10)

# %%
# Loss on test set
test_dataset = PythonGeneDataset(labels_df=test_df, img_dir='data/img/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

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


