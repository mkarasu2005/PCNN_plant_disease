# dataset.py
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Default image root (Windows path)
IMAGE_ROOT_DEFAULT = r"D:\Final year project PCNN\PlantVillage"

def default_transforms(image_size=224, train=True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(0.2,0.2,0.2,0.02),
            T.RandomRotation(20),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size*1.14)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

class PlantVillageImageDataset(Dataset):
    """
    Image-only dataset. Expects images organized in class subfolders under images_root.
    """
    def __init__(self, images_root=IMAGE_ROOT_DEFAULT, transform=None):
        self.images_root = images_root
        self.transform = transform or default_transforms(224, train=True)
        self.samples = []
        classes = []
        for root, dirs, files in os.walk(images_root):
            # skip root itself
            break
        # gather subfolders as classes
        for d in os.listdir(images_root):
            dp = os.path.join(images_root, d)
            if os.path.isdir(dp):
                classes.append(d)
                for f in os.listdir(dp):
                    if f.lower().endswith((".jpg",".jpeg",".png")):
                        self.samples.append((os.path.join(dp,f), d))
        self.classes = sorted(list(set(classes)))
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[cls]
        return {"image": img, "label": label, "path": path}

def make_dataloaders(images_root=IMAGE_ROOT_DEFAULT, batch_size=32, image_size=224, val_split=0.15, num_workers=4, seed=42):
    dataset = PlantVillageImageDataset(images_root=images_root, transform=default_transforms(image_size, train=True))
    n = len(dataset)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    split = int(n*(1-val_split))
    train_idx, val_idx = idx[:split], idx[split:]
    from torch.utils.data import Subset
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, dataset.classes
