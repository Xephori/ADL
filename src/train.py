from __future__ import annotations
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import Config, apply_cli_overrides, get_cli_parser, load_config
from src.dataset import create_dataloaders
from src.models import get_model

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# class weights — computed from CSV metadata (no video loading needed)
def compute_class_weights(metadata_csv: str, num_classes: int, device: torch.device) -> torch.Tensor:
    df = pd.read_csv(metadata_csv)
    train_df = df[df["split"] == "train"]
    counts = torch.zeros(num_classes)
    for label_idx in train_df["label_idx"]:
        counts[int(label_idx)] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)

# Train / validate one epoch
def mixup_data(x, y, alpha=0.2):
    """Blend pairs of samples: x' = lam*x_i + (1-lam)*x_j, returns mixed x, y_a, y_b, lam."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, logits, y_a, y_b, lam):
    """Compute blended loss for mixup."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 0.0,
    mixup_alpha: float = 0.0,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in tqdm(loader, desc="  train", leave=False):
        videos = videos.to(device)
        labels = labels.to(device).long()

        if mixup_alpha > 0:
            videos, labels_a, labels_b, lam = mixup_data(videos, labels, mixup_alpha)

        optimizer.zero_grad()
        logits = model(videos)
        if isinstance(logits, dict):
            logits = logits["logits"]

        if mixup_alpha > 0:
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        preds = logits.argmax(dim=1)
        if mixup_alpha > 0:
            correct += (lam * (preds == labels_a).sum().item() + (1 - lam) * (preds == labels_b).sum().item())
        else:
            correct += (preds == labels).sum().item()
        total += labels.size(0) if mixup_alpha == 0 else labels_a.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for videos, labels in tqdm(loader, desc="  val  ", leave=False):
        videos = videos.to(device)
        labels = labels.to(device).long()

        logits = model(videos)
        if isinstance(logits, dict):
            logits = logits["logits"]

        loss = criterion(logits, labels)

        running_loss += loss.item() * videos.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

# main training
def train(cfg: Config, run_name: str | None = None, loaders=None) -> Dict[str, object]:
    set_seed(cfg.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # data — reuse loaders if provided
    metadata_csv = os.path.join("data", "metadata.csv")
    if loaders is not None:
        train_loader, val_loader, test_loader, num_classes = loaders
    else:
        train_loader, val_loader, test_loader, num_classes, label_to_idx = create_dataloaders(
            metadata_csv=metadata_csv,
            num_frames=cfg.data.num_frames,
            image_size=cfg.data.image_size,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            top_n_classes=cfg.model.num_classes if cfg.model.num_classes < 100 else 0,
        )

    # model
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model: {cfg.model.name} | Params: {total_params:,} (trainable: {trainable_params:,})")

    # loss
    if cfg.training.use_class_weights:
        weights = compute_class_weights(metadata_csv, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.training.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)

    # optimiser
    if cfg.training.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

    # scheduler
    if cfg.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.step_size,
            gamma=cfg.training.step_gamma,
        )
    
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    os.makedirs(cfg.paths.save_dir, exist_ok=True)

    tag = run_name or cfg.model.name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_csv_path = os.path.join(cfg.paths.results_dir, f"{tag}_{timestamp}_log.csv")
    best_model_path = os.path.join(cfg.paths.save_dir, f"{tag}_{timestamp}_best.pth")
    log_rows = []
    best_val_loss = float("inf")
    patience_counter = 0

    # training loop
    print(f"[train] Starting training for {cfg.training.epochs} epochs …")
    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, grad_clip=cfg.training.grad_clip, mixup_alpha=cfg.training.mixup_alpha)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "time_s": elapsed,
        })
        print(
            f"  Epoch {epoch:3d}/{cfg.training.epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f} | {elapsed:.1f}s"
        )
        # early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.training.early_stopping_patience:
                print(f"[train] Early stopping at epoch {epoch} (patience={cfg.training.early_stopping_patience})")
                break

        # save log after every epoch (in case session dies)
        pd.DataFrame(log_rows).to_csv(log_csv_path, index=False)

    # training log
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_csv_path, index=False)
    print(f"[train] Training log saved {log_csv_path}")

    # final test eval
    print("[train] Evaluating best model on test set …")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"[train] Test loss={test_loss:.4f}  Test acc={test_acc:.4f}")

    return {
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_model_path": best_model_path,
        "log_csv_path": log_csv_path,
    }

def main() -> None:
    parser = get_cli_parser()
    parser.add_argument("--train-all", action="store_true", help="Train all 3 models with one data load")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.train_all:
        # Load data ONCE, train all 3 models
        metadata_csv = os.path.join("data", "metadata.csv")
        train_loader, val_loader, test_loader, num_classes, _ = create_dataloaders(
            metadata_csv=metadata_csv,
            num_frames=cfg.data.num_frames,
            image_size=cfg.data.image_size,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            top_n_classes=cfg.model.num_classes if cfg.model.num_classes < 100 else 0,
        )
        loaders = (train_loader, val_loader, test_loader, num_classes)
        for model_name, run_name in [("model_a", "model_a_baseline"), ("model_b", "model_b_lstm"), ("model_c", "model_c_attention")]:
            print(f"\n{'='*50}")
            print(f"  Training {model_name}")
            print(f"{'='*50}")
            cfg.model.name = model_name
            results = train(cfg, run_name=run_name, loaders=loaders)
            print(f"[train] {model_name} done — Test acc: {results['test_acc']:.4f}\n")
        return

    print(f"[train] Config: model={cfg.model.name}, lr={cfg.training.learning_rate}, "
          f"batch={cfg.training.batch_size}, frames={cfg.data.num_frames}, "
          f"lstm_hidden={cfg.model.lstm_hidden_size}, lstm_layers={cfg.model.lstm_num_layers}")

    results = train(cfg, run_name=args.run_name)
    print(f"\n[train] Done, Best val loss: {results['best_val_loss']:.4f}, "
          f"Test acc: {results['test_acc']:.4f}")
    
if __name__ == "__main__":
    main()
