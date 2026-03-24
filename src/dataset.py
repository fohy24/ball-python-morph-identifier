import os
import torch
from torch.utils.data import Dataset
from PIL import Image


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
