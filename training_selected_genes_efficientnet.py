import pandas as pd
import numpy as np
import ast

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch import nn, optim
from torchvision import datasets, utils, models
# from torchinfo import summary
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR
import torch.optim as optim
from torchvision.transforms import v2
import multiprocessing

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
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
IMAGE_SIZE = 480
BATCH_SIZE = 36
LEARNING_RATE = 0.00004

LOAD_CHECKPOINT = False
checkpoint_version = 1
checkpoint_epoch = 20
num_additional_epochs = 18

SAVE_CHECKPOINT = True
SAVE_AS_VERSION = "13_1"

#########################################################################

labels_extended = pd.read_csv('data/selected_gene_df.csv')

if SUBSET:
    labels_extended = labels_extended.sample(frac=0.01, random_state=123)

clean_possible_genes = labels_extended.columns.to_list()[7:]
print(f'Number of labels: {len(clean_possible_genes)}')

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
        labels = torch.tensor(self.labels_df.iloc[idx, 7:].astype('float32').values)
        
        if self.transform:
            image = self.transform(image)   

        return image, labels


transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.ToDtype(torch.float32), # , scale=True
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

full_dataset = PythonGeneDataset(labels_df=train_df, img_dir='data/img/', transform=transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')

total_size = len(full_dataset)
train_size = int(0.8 * total_size)
valid_size = total_size - train_size
generator1 = torch.Generator(device='cpu').manual_seed(42)
train_indices, valid_indices = torch.utils.data.random_split(
    np.arange(total_size),
    [train_size, valid_size],
    generator=generator1
    )


train_dataset = Subset(full_dataset, train_indices)
valid_dataset = Subset(full_dataset, valid_indices)


print(f'Number of workers: {multiprocessing.cpu_count()}')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=multiprocessing.cpu_count(), pin_memory=True)

efficientnet = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')

for param in efficientnet.parameters():
    param.requires_grad = True

# for name, child in efficientnet.named_children():
#     if name == 'features':
#         for child_name, parameters in child.named_children():
#             if 'denseblock2' in child_name or 'denseblock3' in child_name or 'denseblock4' in child_name:
#                 for param in parameters.parameters():
#                     param.requires_grad = True



num_labels = len(clean_possible_genes)
print(f'Number of labels: {num_labels}')

new_layers = nn.Sequential(
    nn.LazyLinear(2048),  
    nn.BatchNorm1d(2048),  
    nn.ReLU(),            
    nn.Dropout(0.5),      
    nn.LazyLinear(num_labels) 
)

efficientnet.classifier = new_layers

# Implement focal loss for label imbalance
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) 
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
efficientnet.to(device)

focal_loss = FocalLoss()
criterion = focal_loss

# criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(efficientnet.parameters(), lr=LEARNING_RATE)
# scheduler = ExponentialLR(optimizer, gamma=0.7)
scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_additional_epochs)
scaler = GradScaler()

result = {
    "epoch": [],
    "train_loss": [],
    "valid_loss": [],
    "lr": []
}

def train_model(model, criterion, optimizer, start_epoch, total_epochs, result_dict, version=1, save_checkpoint=True):
    print(f'Start training - batch size: {BATCH_SIZE} & image size: {IMAGE_SIZE}')
    
    for epoch in range(start_epoch, total_epochs + 1):
        model.train()

        with tqdm(train_loader, unit="batch") as tepoch:
            train_loss = 0.0
            for inputs, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{total_epochs}")
                inputs, labels = inputs.to(device), labels.to(device)
                
                # forward pass
                with autocast(dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # backward 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                train_loss += loss.item() * inputs.size(0)

            train_loss = train_loss / len(train_loader.dataset)

            # Validation
            model.eval() 
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    with autocast(dtype=torch.float16):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)

            valid_loss = valid_loss / len(valid_loader.dataset)
            scheduler.step()

            # Save model
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
                
            result_dict["epoch"].append(epoch)
            result_dict["train_loss"].append(train_loss)
            result_dict["valid_loss"].append(valid_loss)
            result_dict["lr"].append(scheduler.get_last_lr()[0])

            print(f'Epoch {epoch}/{total_epochs}, Train Loss: {train_loss:.5f}, Valid Loss: {valid_loss:.5f}, Learning rate: {scheduler.get_last_lr()}')
            tepoch.set_postfix(train_loss=train_loss,
                               valid_loss=valid_loss)

        # Save results
        pd.DataFrame(result).to_csv(f'results/version{SAVE_AS_VERSION}_ImageSize{IMAGE_SIZE}_LR{LEARNING_RATE}.csv', index=False)


def get_epoch(LOAD_CHECKPOINT, num_additional_epochs, checkpoint_version, checkpoint_epoch):
    if LOAD_CHECKPOINT:
        checkpoint_path = f'model/full_dataset_v{checkpoint_version}_epoch{checkpoint_epoch}.pt'
        checkpoint = torch.load(checkpoint_path)
        print(f'Loaded version {checkpoint_version}-epoch {checkpoint_epoch}')

        efficientnet.load_state_dict(checkpoint['model_state_dict'])
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


train_model(efficientnet, criterion, optimizer,
            start_epoch=start_epoch,
            total_epochs=total_epochs,
            result_dict=result,
            version=SAVE_AS_VERSION,
            save_checkpoint=SAVE_CHECKPOINT)


# Loss on test set
test_dataset = PythonGeneDataset(labels_df=test_df, img_dir='data/img/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

best_model_epoch = result["epoch"][result["valid_loss"].index(min(result["valid_loss"]))]
best_model_path = f'model/model_v{SAVE_AS_VERSION}_epoch{best_model_epoch}.pt'
best_model = torch.load(best_model_path, map_location=device)

efficientnet = models.efficientnet_v2_l(weights='EfficientNet_V2_L_Weights.DEFAULT')
efficientnet.classifier = new_layers

efficientnet.load_state_dict(best_model['model_state_dict'])
epoch = best_model['epoch']
train_loss = best_model['train_loss']
valid_loss = best_model['valid_loss']
print(f'Calculating test loss on model_v{SAVE_AS_VERSION}_epoch{best_model_epoch}...')

efficientnet.to(device).eval()

test_loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        with autocast(dtype=torch.float16):
            outputs = efficientnet(inputs)
            loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

# Calculate average loss over validation data
test_loss = test_loss / len(test_loader.dataset)
print(f'Loss on Test set: {test_loss:.5f}')


