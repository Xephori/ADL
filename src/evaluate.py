from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from src.config import Config, load_config
from src.dataset import create_dataloaders
from src.models import get_model

# Prediction helpers
@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for videos, labels in tqdm(loader, desc="Evaluating", leave=False):
        videos = videos.to(device)
        logits = model(videos)
        if isinstance(logits, dict):
            logits = logits["logits"]

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        labels_np = labels.numpy()

        all_labels.append(labels_np)
        all_preds.append(preds)
        all_probs.append(probs)

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )

# Metrics
def classification_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: List[str] | None = None,
) -> Dict[str, object]:
    num_classes = probs.shape[1]

    top1_acc = accuracy_score(labels, preds)
    top5_acc = top_k_accuracy_score(labels, probs, k=min(5, num_classes), labels=list(range(num_classes)))
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    macro_recall = recall_score(labels, preds, average="macro", zero_division=0)

    report = classification_report(
        labels, preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    return {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str] | None = None,
) -> Dict[str, object]:
    labels, preds, probs = get_predictions(model, loader, device)
    metrics = classification_metrics(labels, preds, probs, class_names)
    return metrics

def print_metrics(metrics: Dict[str, object]) -> None:
    print(f"  Top-1 Accuracy : {metrics['top1_accuracy']:.4f} ({metrics['top1_accuracy']*100:.2f}%)")
    print(f"  Top-5 Accuracy : {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall   : {metrics['macro_recall']:.4f}")

# CLI entry point
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, required=True,
                        help="model_a | model_b | model_c")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to .pth state dict")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Config file")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"], help="Which split to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.model.name = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_csv = os.path.join("data", "metadata.csv")
    train_loader, val_loader, test_loader, num_classes, label_to_idx = create_dataloaders(
        metadata_csv=metadata_csv,
        num_frames=cfg.data.num_frames,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        top_n_classes=cfg.model.num_classes if cfg.model.num_classes < 100 else 0,
    )

    loader = test_loader if args.split == "test" else val_loader
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(num_classes)]

    model = get_model(
        cfg.model.name,
        num_classes=num_classes,
        block_channels=cfg.model.cnn_channels,
        lstm_hidden_size=cfg.model.lstm_hidden_size,
        lstm_num_layers=cfg.model.lstm_num_layers,
        lstm_bidirectional=cfg.model.lstm_bidirectional,
        lstm_dropout=cfg.model.lstm_dropout,
        attention_hidden_dim=cfg.model.attention_hidden_dim,
        classifier_dropout=cfg.model.classifier_dropout,
    ).to(device)

    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"[evaluate] Loaded weights from {args.weights}")

    metrics = evaluate_model(model, loader, device, class_names)
    print_metrics(metrics)

    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    report_path = os.path.join(cfg.paths.results_dir, f"{args.model}_{args.split}_report.csv")
    report_df.to_csv(report_path)
    print(f"[evaluate] Per-class report saved {report_path}")


if __name__ == "__main__":
    main()
