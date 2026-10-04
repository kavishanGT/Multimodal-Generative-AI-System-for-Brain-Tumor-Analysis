

import os
import shutil
import random

random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(BASE_DIR, "data")
output_dir = os.path.join(BASE_DIR, "processed_data")

train_ratio = 0.7
val_ratio = 0.15

classes = os.listdir(dataset_dir)

for cls in classes:

    cls_path = os.path.join(dataset_dir, cls)

    images = os.listdir(cls_path)

    random.shuffle(images)

    total = len(images)

    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for split, img_list in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):

        split_dir = os.path.join(
            output_dir,
            split,
            cls
        )

        os.makedirs(split_dir, exist_ok=True)

        for img in img_list:

            src = os.path.join(cls_path, img)
            dst = os.path.join(split_dir, img)

            shutil.copy(src, dst)

print("Dataset split complete!")