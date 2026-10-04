# transform.py
# Backward-compatibility alias pointing to TrainData/transform.py

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from TrainData.transform import (
    train_transform,
    val_transform,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__all__ = [
    "train_transform",
    "val_transform",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
