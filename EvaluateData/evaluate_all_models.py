# evaluate_all_models.py
# Comprehensive evaluation of all 3 trained models with full metrics + plots

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    f1_score
)

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from TrainData.transform import val_transform
except ImportError:
    from transform import val_transform

from models import MODEL_REGISTRY


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
BATCH_SIZE = 16
NUM_WORKERS = 0

os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot styling
plt.style.use("seaborn-v0_8-darkgrid")
COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
MODEL_COLORS = {"EfficientNet-B0": "#2196F3", "ResNet-50": "#FF5722", "DeiT-Small": "#4CAF50"}


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
test_dataset = datasets.ImageFolder(
    root=f"{DATA_DIR}/test",
    transform=val_transform
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

CLASS_NAMES = test_dataset.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"Test set: {len(test_dataset)} images, Classes: {CLASS_NAMES}")


# ──────────────────────────────────────────────
# Evaluation Function
# ──────────────────────────────────────────────
def evaluate_model(model_name, model_class):
    """Evaluate a single model with comprehensive metrics."""

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {model_name}")
    print(f"{'='*60}")

    # Load model
    model = model_class(num_classes=NUM_CLASSES)
    ckpt_name = f"{model_name.replace(' ', '_').replace('-', '_').lower()}.pth"
    ckpt_path = os.path.join(CHECKPOINTS_DIR, ckpt_name)

    if not os.path.exists(ckpt_path):
        print(f"  ✗ Checkpoint not found: {ckpt_path}")
        return None

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_inference_time = 0.0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {model_name}", leave=False):
            images = images.to(DEVICE)

            start_time = time.time()
            outputs = model(images)
            total_inference_time += time.time() - start_time

            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ── Metrics ──
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1_weighted = f1_score(all_labels, all_preds, average="weighted") * 100
    f1_macro = f1_score(all_labels, all_preds, average="macro") * 100

    # Per-class metrics
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    report_text = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES
    )

    # ROC-AUC (One-vs-Rest)
    try:
        roc_auc_macro = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average="macro"
        ) * 100
        roc_auc_per_class = {}
        for i, cls in enumerate(CLASS_NAMES):
            binary_labels = (all_labels == i).astype(int)
            roc_auc_per_class[cls] = roc_auc_score(
                binary_labels, all_probs[:, i]
            ) * 100
    except Exception as e:
        roc_auc_macro = 0.0
        roc_auc_per_class = {}
        print(f"  ROC-AUC computation error: {e}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Inference time
    avg_inference_ms = (total_inference_time / len(test_dataset)) * 1000

    # Print results
    print(f"\n  Classification Report:")
    print(report_text)
    print(f"  Overall Accuracy:  {accuracy:.2f}%")
    print(f"  F1 Weighted:       {f1_weighted:.2f}%")
    print(f"  F1 Macro:          {f1_macro:.2f}%")
    print(f"  ROC-AUC Macro:     {roc_auc_macro:.2f}%")
    print(f"  Avg Inference:     {avg_inference_ms:.2f} ms/image")

    results = {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_per_class": roc_auc_per_class,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "avg_inference_ms": avg_inference_ms,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
    }

    return results


# ──────────────────────────────────────────────
# Plotting Functions
# ──────────────────────────────────────────────
def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, linecolor="white",
        annot_kws={"size": 14, "fontweight": "bold"}
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("-", "_").lower()
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curves(all_labels, all_probs, model_name):
    """Plot ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cls in enumerate(CLASS_NAMES):
        binary_labels = (all_labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, i])
        auc = roc_auc_score(binary_labels, all_probs[:, i])
        ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                label=f"{cls} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("-", "_").lower()
    path = os.path.join(PLOTS_DIR, f"roc_curves_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_curves(model_name):
    """Plot training and validation loss/accuracy curves."""
    safe_name = model_name.replace(" ", "_").replace("-", "_").lower()
    history_path = os.path.join(RESULTS_DIR, f"{safe_name}_history.json")

    if not os.path.exists(history_path):
        print(f"  ✗ History not found: {history_path}")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"Loss Curves — {model_name}", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "r-o", markersize=4, label="Val Acc")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].set_title(f"Accuracy Curves — {model_name}", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"training_curves_{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_model_comparison(all_results):
    """Plot side-by-side comparison of all models."""
    model_names = [r["model_name"] for r in all_results]
    accuracies = [r["accuracy"] for r in all_results]
    f1_scores_w = [r["f1_weighted"] for r in all_results]
    roc_aucs = [r["roc_auc_macro"] for r in all_results]
    inf_times = [r["avg_inference_ms"] for r in all_results]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    colors = [MODEL_COLORS.get(n, "#999") for n in model_names]

    # Accuracy
    bars = axes[0].bar(model_names, accuracies, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Test Accuracy (%)", fontsize=13, fontweight="bold")
    axes[0].set_ylim([max(0, min(accuracies) - 10), 105])
    for bar, val in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)

    # F1 Weighted
    bars = axes[1].bar(model_names, f1_scores_w, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("F1 Score Weighted (%)", fontsize=13, fontweight="bold")
    axes[1].set_ylim([max(0, min(f1_scores_w) - 10), 105])
    for bar, val in zip(bars, f1_scores_w):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)

    # ROC-AUC
    bars = axes[2].bar(model_names, roc_aucs, color=colors, edgecolor="white", linewidth=1.5)
    axes[2].set_title("ROC-AUC Macro (%)", fontsize=13, fontweight="bold")
    axes[2].set_ylim([max(0, min(roc_aucs) - 10), 105])
    for bar, val in zip(bars, roc_aucs):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontweight="bold", fontsize=11)

    # Inference Time
    bars = axes[3].bar(model_names, inf_times, color=colors, edgecolor="white", linewidth=1.5)
    axes[3].set_title("Inference Time (ms/image)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, inf_times):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{val:.1f}", ha="center", fontweight="bold", fontsize=11)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15, labelsize=9)

    plt.suptitle("Model Comparison — Brain Tumor Classification",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved comparison chart: {path}")


def plot_per_class_comparison(all_results):
    """Bar chart comparing per-class F1 for each model."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(NUM_CLASSES)
    width = 0.25

    for i, r in enumerate(all_results):
        f1_vals = [r["classification_report"][cls]["f1-score"] * 100 for cls in CLASS_NAMES]
        bars = ax.bar(x + i * width, f1_vals, width,
                      label=r["model_name"],
                      color=list(MODEL_COLORS.values())[i],
                      edgecolor="white", linewidth=1)
        for bar, val in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Tumor Class", fontsize=12, fontweight="bold")
    ax.set_ylabel("F1-Score (%)", fontsize=12, fontweight="bold")
    ax.set_title("Per-Class F1-Score Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 110])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "per_class_f1_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────
if __name__ == "__main__":
    all_results = []

    for model_name, model_class in MODEL_REGISTRY.items():
        results = evaluate_model(model_name, model_class)
        if results is not None:
            all_results.append(results)

            # Generate per-model plots
            cm = np.array(results["confusion_matrix"])
            plot_confusion_matrix(cm, model_name)
            plot_roc_curves(results["all_labels"], results["all_probs"], model_name)
            plot_training_curves(model_name)

    if len(all_results) > 1:
        # Generate comparison plots
        plot_model_comparison(all_results)
        plot_per_class_comparison(all_results)

    # ── Find best model ──
    best_result = max(all_results, key=lambda r: r["accuracy"])
    print(f"\n{'='*60}")
    print(f"  🏆 BEST MODEL: {best_result['model_name']}")
    print(f"     Accuracy:     {best_result['accuracy']:.2f}%")
    print(f"     F1 Weighted:  {best_result['f1_weighted']:.2f}%")
    print(f"     ROC-AUC:      {best_result['roc_auc_macro']:.2f}%")
    print(f"     Inference:    {best_result['avg_inference_ms']:.2f} ms/image")
    print(f"{'='*60}")

    # Save evaluation results (without numpy arrays)
    eval_summary = []
    for r in all_results:
        eval_summary.append({
            "model_name": r["model_name"],
            "accuracy": r["accuracy"],
            "f1_weighted": r["f1_weighted"],
            "f1_macro": r["f1_macro"],
            "roc_auc_macro": r["roc_auc_macro"],
            "roc_auc_per_class": r["roc_auc_per_class"],
            "avg_inference_ms": r["avg_inference_ms"],
            "confusion_matrix": r["confusion_matrix"],
            "classification_report": r["classification_report"],
        })

    with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    # Update best model info
    with open(os.path.join(RESULTS_DIR, "best_model.json"), "w") as f:
        json.dump({
            "best_model": best_result["model_name"],
            "accuracy": best_result["accuracy"],
            "f1_weighted": best_result["f1_weighted"],
            "roc_auc_macro": best_result["roc_auc_macro"],
            "checkpoint": os.path.join(
                CHECKPOINTS_DIR,
                f"{best_result['model_name'].replace(' ', '_').replace('-', '_').lower()}.pth"
            )
        }, f, indent=2)

    print(f"\n  All evaluation results saved to: {RESULTS_DIR}/")
    print(f"  Plots saved to: {PLOTS_DIR}/")
    print("  Evaluation complete!")
