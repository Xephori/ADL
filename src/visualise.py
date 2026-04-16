from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

def plot_training_curves(
    log_csv_path: str,
    title_prefix: str = "",
    save_dir: Optional[str] = None,
) -> None:
    df = pd.read_csv(log_csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title_prefix} Loss Curves".strip())
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(df["epoch"], df["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(df["epoch"], df["val_acc"], label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{title_prefix} Accuracy Curves".strip())
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        tag = title_prefix.replace(" ", "_").lower() or "training"
        fig.savefig(Path(save_dir) / f"{tag}_curves.png", dpi=150, bbox_inches="tight")

    plt.show()

#Confusion matrix
def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (7, 6),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

# model comparison bar chart
def plot_model_comparison(
    model_names: List[str],
    accuracies: List[float],
    metric_name: str = "Top-1 Accuracy",
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#3498db", "#e67e22", "#2ecc71"]
    bars = ax.bar(model_names, [a * 100 for a in accuracies], color=colors[:len(model_names)])

    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc*100:.1f}%", ha="center", va="bottom", fontweight="bold",
        )
    ax.set_ylabel(f"{metric_name} (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

# per class accuracy
def plot_per_class_accuracy(
    cm: np.ndarray,
    class_names: List[str],
    top_n: int = 20,
    title: str = "Per-Class Accuracy",
    save_path: Optional[str] = None,
) -> None:
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    sorted_idx = np.argsort(per_class_acc)

    show_idx = np.concatenate([sorted_idx[:top_n], sorted_idx[-top_n:]])
    show_idx = np.unique(show_idx)
    show_idx = show_idx[np.argsort(per_class_acc[show_idx])]

    fig, ax = plt.subplots(figsize=(10, max(6, len(show_idx) * 0.3)))

    names = [class_names[i] for i in show_idx]
    accs = per_class_acc[show_idx] * 100
    colors = ["#e74c3c" if a < 50 else "#2ecc71" for a in accs]

    ax.barh(names, accs, color=colors)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

# attention heatmap over video frames
def plot_attention_frames(
    frames: List[np.ndarray],
    attention_weights: np.ndarray,
    predicted_label: str,
    true_label: str,
    save_path: Optional[str] = None,
) -> None:
    num_frames = len(frames)
    fig, axes = plt.subplots(2, 1, figsize=(num_frames * 1.8, 5),
                              gridspec_kw={"height_ratios": [4, 1]})
    ax_frames = axes[0]
    ax_frames.set_xlim(0, num_frames)
    ax_frames.set_ylim(0, 1)
    ax_frames.set_aspect("equal")
    ax_frames.axis("off")

    for i, (frame, weight) in enumerate(zip(frames, attention_weights)):
        extent = [i + 0.05, i + 0.95, 0.05, 0.95]
        ax_frames.imshow(frame, extent=extent, aspect="auto")

    pred_label_only = predicted_label.split(" (")[0]
    correct = pred_label_only == true_label
    colour = "green" if correct else "red"
    ax_frames.set_title(
        f"Pred: {predicted_label} | True: {true_label}",
        fontsize=13, color=colour, fontweight="bold",
    )
    ax_bar = axes[1]
    colours = plt.cm.Reds(attention_weights / (attention_weights.max() + 1e-8))
    ax_bar.bar(range(num_frames), attention_weights, color=colours, width=0.8)
    ax_bar.set_xlabel("Frame index")
    ax_bar.set_ylabel("Attn weight")
    ax_bar.set_xlim(-0.5, num_frames - 0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()

# class distribution
def plot_class_distribution(
    metadata_csv: str,
    save_path: Optional[str] = None,
) -> None:
    df = pd.read_csv(metadata_csv)
    counts = df["label"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.25)))
    ax.barh(counts.index, counts.values, color="#3498db")
    ax.set_xlabel("Number of samples")
    ax.set_title("Class Distribution (WLASL-100)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
