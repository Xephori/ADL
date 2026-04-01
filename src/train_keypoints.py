"""
Training script for the MediaPipe keypoint model (KeypointLSTM).

Usage:
    python -m src.train_keypoints
    python -m src.train_keypoints --epochs 200 --lr 1e-3 --run-name kp_v1

Reuses train_one_epoch / validate from src.train — they are model-agnostic.
"""
from __future__ import annotations
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from src.dataset_keypoints import create_keypoint_dataloaders
from src.models.model_keypoint import KeypointLSTM
from src.train import set_seed, validate, train_one_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Train KeypointLSTM on MediaPipe hand landmarks")
    parser.add_argument("--metadata-csv", default="data/metadata.csv")
    parser.add_argument("--cache-dir", default="data/keypoints_cache")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--noise-std", type=float, default=0.02,
                        help="Std of Gaussian noise added to keypoints during training (0=off)")
    parser.add_argument("--run-name", type=str, default="kp_lstm")
    parser.add_argument("--save-dir", default="saved_models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[kp-train] Device: {device}")

    train_loader, val_loader, test_loader, num_classes, _ = create_keypoint_dataloaders(
        metadata_csv=args.metadata_csv,
        batch_size=args.batch_size,
        top_n_classes=args.num_classes,
        cache_dir=args.cache_dir,
    )

    model = KeypointLSTM(
        input_dim=63,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[kp-train] Model params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(args.save_dir, f"{args.run_name}_{timestamp}_best.pth")
    log_csv_path = os.path.join(args.results_dir, f"{args.run_name}_{timestamp}_log.csv")

    best_val_acc = 0.0
    patience_counter = 0
    log_rows = []

    # Keep a clean copy of training keypoints to restore after each noisy epoch
    original_train_keypoints = train_loader.keypoints.clone() if args.noise_std > 0 else None

    print(f"[kp-train] Starting training for {args.epochs} epochs ...")
    print(f"[kp-train] Keypoint noise std: {args.noise_std}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Keypoint noise augmentation — add Gaussian noise to training keypoints each epoch
        # Simulates different hand sizes, finger lengths, slight pose variations
        # Restored after each epoch so noise never accumulates
        if args.noise_std > 0:
            noise = torch.randn_like(original_train_keypoints) * args.noise_std
            train_loader.keypoints = original_train_keypoints + noise

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip=1.0, mixup_alpha=0.0,
        )
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
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Saved best model → {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[kp-train] Early stopping at epoch {epoch} (patience={args.patience})")
                break

        pd.DataFrame(log_rows).to_csv(log_csv_path, index=False)

    print(f"[kp-train] Training log saved → {log_csv_path}")

    print("[kp-train] Evaluating best model on test set ...")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"[kp-train] Test loss={test_loss:.4f}  Test acc={test_acc:.4f}")
    print(f"\n[kp-train] Done. Best val acc: {best_val_acc:.4f}  Test acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
