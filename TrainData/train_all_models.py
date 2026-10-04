# train_all_models.py
# Unified training script for all 3 models with proper training practices

import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from TrainData.transform import train_transform, val_transform
except ImportError:
    from transform import train_transform, val_transform

from models import MODEL_REGISTRY


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # Early stopping patience
NUM_WORKERS = 0  # Windows compatibility

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
print("=" * 70)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
train_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/train",
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/val",
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

CLASS_NAMES = train_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)

print(f"Classes: {CLASS_NAMES}")
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
print("=" * 70)


# ──────────────────────────────────────────────
# Training Function
# ──────────────────────────────────────────────
def train_one_model(model_name, model_class):
    """Train a single model with early stopping and LR scheduling."""

    print(f"\n{'='*70}")
    print(f"  TRAINING: {model_name}")
    print(f"{'='*70}")

    # Initialize model
    model = model_class(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer — only optimize trainable parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Cosine annealing LR scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_times": []
    }

    # Early stopping
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ── Training Phase ──
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{model_name}] Epoch {epoch+1}/{EPOCHS} Train",
            leave=False
        )

        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*correct/total:.1f}%"
            })

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ── Validation Phase ──
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / total
        val_acc = 100.0 * correct / total

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - epoch_start

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epoch_times"].append(epoch_time)

        print(
            f"  Epoch {epoch+1:2d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Early stopping check (based on val accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  ✓ New best val accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ✗ Early stopping at epoch {epoch+1} (patience={PATIENCE})")
                break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save checkpoint
    checkpoint_path = os.path.join(
        CHECKPOINTS_DIR,
        f"{model_name.replace(' ', '_').replace('-', '_').lower()}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  Model saved: {checkpoint_path}")

    # Save training history
    history_path = os.path.join(
        RESULTS_DIR,
        f"{model_name.replace(' ', '_').replace('-', '_').lower()}_history.json"
    )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Summary
    total_time = sum(history["epoch_times"])
    summary = {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_train_time": total_time,
        "epochs_trained": len(history["train_loss"]),
        "checkpoint_path": checkpoint_path,
    }

    print(f"\n  SUMMARY — {model_name}")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  Training Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Epochs Trained: {summary['epochs_trained']}")

    return model, summary, history


# ──────────────────────────────────────────────
# Train All Models
# ──────────────────────────────────────────────
if __name__ == "__main__":
    all_summaries = {}

    for model_name, model_class in MODEL_REGISTRY.items():
        model, summary, history = train_one_model(model_name, model_class)
        all_summaries[model_name] = summary
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save all summaries
    with open(os.path.join(RESULTS_DIR, "training_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print comparison table
    print(f"\n{'='*70}")
    print("  MODEL COMPARISON — VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Val Acc':>10} {'Params':>12} {'Train Time':>12}")
    print("-" * 56)

    best_model = None
    best_acc = 0

    for name, s in all_summaries.items():
        print(
            f"{name:<20} {s['best_val_acc']:>9.2f}% "
            f"{s['total_params']:>12,} "
            f"{s['total_train_time']:>10.1f}s"
        )
        if s["best_val_acc"] > best_acc:
            best_acc = s["best_val_acc"]
            best_model = name

    print("-" * 56)
    print(f"\n  🏆 BEST MODEL: {best_model} ({best_acc:.2f}%)")

    # Save best model info
    with open(os.path.join(RESULTS_DIR, "best_model.json"), "w") as f:
        json.dump({
            "best_model": best_model,
            "best_val_acc": best_acc,
            "checkpoint": all_summaries[best_model]["checkpoint_path"]
        }, f, indent=2)

    print(f"\n  Results saved to: {RESULTS_DIR}/")
    print(f"  Checkpoints saved to: {CHECKPOINTS_DIR}/")
    print("  Training complete!")
