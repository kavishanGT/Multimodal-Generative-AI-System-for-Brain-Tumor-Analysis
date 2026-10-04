# save_labels.py

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from preprocess_data.data_loader import train_dataset, val_dataset, test_dataset
except ImportError:
    from data_loader import train_dataset, val_dataset, test_dataset

labels = train_dataset.classes

label_path = os.path.join(os.path.dirname(__file__), "labels.json")
with open(label_path, "w") as f:
    json.dump(labels, f)

print(f"Labels saved to {label_path}: {labels}")


# dataset_stats.py

print("Train size:", len(train_dataset))
print("Validation size:", len(val_dataset))
print("Test size:", len(test_dataset))