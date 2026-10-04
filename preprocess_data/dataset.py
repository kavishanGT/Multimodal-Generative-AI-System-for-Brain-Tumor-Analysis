# check_dataset.py

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "data")

classes = os.listdir(dataset_path)

print("Classes found:", classes)

for cls in classes:
    path = os.path.join(dataset_path, cls)
    print(cls, ":", len(os.listdir(path)), "images")