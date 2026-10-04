# dataset_loader.py

import os
import sys
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from TrainData.transform import train_transform, val_transform
except ImportError:
    from transform import train_transform, val_transform

data_dir = os.path.join(BASE_DIR, "processed_data")

train_dataset = datasets.ImageFolder(
    root=f"{data_dir}/train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=f"{data_dir}/val",
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    root=f"{data_dir}/test",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

print("Classes:", train_dataset.classes)