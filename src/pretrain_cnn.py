# Pre-train the custom CNN backbone on individual video frames
from __future__ import annotations
import argparse
import os
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.cnn_backbone import CNNBackbone

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_frames_from_cache(
    metadata_csv: str,
    cache_dir: str,
    split: str,
    top_n_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    df = pd.read_csv(metadata_csv)

    # Filter to top N classes
    if top_n_classes > 0:
        train_counts = df[df["split"] == "train"]["label"].value_counts()
        top_labels = train_counts.head(top_n_classes).index.tolist()
        df = df[df["label"].isin(top_labels)]

    labels_sorted = sorted(df["label"].unique())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels_sorted)}
    num_classes = len(labels_sorted)

    old_to_new = {}
    for _, row in df.iterrows():
        old_idx = int(row["label_idx"])
        if old_idx not in old_to_new:
            old_to_new[old_idx] = label_to_idx[row["label"]]

    split_df = df[df["split"] == split]
    indices = split_df.index.tolist()

    frames_list = []
    labels_list = []

    for i in indices:
        pt_path = os.path.join(cache_dir, f"{i}.pt")
        if not os.path.exists(pt_path):
            continue
        d = torch.load(pt_path, weights_only=False)
        video_frames = d["frames"] 
        old_label = int(d["label_idx"])
        new_label = old_to_new[old_label]

        # Split video into individual frames
        for t in range(video_frames.shape[0]):
            frames_list.append(video_frames[t])  
            labels_list.append(new_label)

    all_frames = torch.stack(frames_list) 
    all_labels = torch.tensor(labels_list, dtype=torch.long)
    return all_frames, all_labels, num_classes


def main():
    parser = argparse.ArgumentParser(description="Pre-train CNN backbone on individual video frames")
    parser.add_argument("--metadata", default="data/metadata.csv", help="Path to metadata CSV")
    parser.add_argument("--cache-dir", default="data/frames_cache", help="Path to pre-extracted .pt frames")
    parser.add_argument("--top-n-classes", type=int, default=10, help="Number of classes to use")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="saved_models/cnn_backbone_pretrained.pth",
                        help="Where to save pretrained backbone weights")
    parser.add_argument("--load-from", default=None,
                        help="Load backbone weights before training (for fine-tuning after ASL pre-training)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain] Device: {device}")

    # Load and split frames
    print("[pretrain] Loading training frames...")
    train_frames, train_labels, num_classes = load_frames_from_cache(
        args.metadata, args.cache_dir, "train", args.top_n_classes)
    print(f"[pretrain] Train: {len(train_labels)} individual frames, {num_classes} classes")

    print("[pretrain] Loading val frames...")
    val_frames, val_labels, _ = load_frames_from_cache(
        args.metadata, args.cache_dir, "val", args.top_n_classes)
    print(f"[pretrain] Val: {len(val_labels)} individual frames")

    # DataLoaders
    train_dataset = TensorDataset(train_frames, train_labels)
    val_dataset = TensorDataset(val_frames, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model CNN backbone + simple classification head
    backbone = CNNBackbone(in_channels=3, block_channels=[32, 64, 128, 256])

    if args.load_from and os.path.exists(args.load_from):
        backbone.load_state_dict(torch.load(args.load_from, map_location="cpu", weights_only=True))
        print(f"[pretrain] Loaded backbone weights from {args.load_from}")

    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(backbone.feature_dim, num_classes),
    )

    model = nn.Sequential(backbone, classifier).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] Params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best_val_acc = 0.0

    print(f"[pretrain] Starting pre-training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_correct, train_total = 0, 0
        train_loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item() * labels.size(0)

        train_acc = train_correct / train_total
        train_loss = train_loss_sum / train_total

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
                val_loss_sum += loss.item() * labels.size(0)

        val_acc = val_correct / val_total
        val_loss = val_loss_sum / val_total
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(backbone.state_dict(), args.output)
            print(f"  Saved backbone → {args.output} (val_acc={val_acc:.4f})")

    print(f"\n[pretrain] Done. Best val acc: {best_val_acc:.4f}")
    print(f"[pretrain] Backbone weights saved to: {args.output}")


if __name__ == "__main__":
    main()
