# Pre-train the custom CNN backbone on ASL Alphabet dataset
from __future__ import annotations
import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from src.models.cnn_backbone import CNNBackbone


class PreloadedDataset(Dataset):

    def __init__(self, base_dataset, indices, transform_on_tensor=None):
        self.images = []
        self.labels = []
        self.transform_on_tensor = transform_on_tensor
        print(f"[pretrain-asl] Preloading {len(indices)} images into RAM...", flush=True)
        for i, idx in enumerate(indices):
            img, label = base_dataset[idx] 
            self.images.append(img)
            self.labels.append(label)
            if (i + 1) % 2000 == 0:
                print(f"    loaded {i+1}/{len(indices)}", flush=True)
        print(f"[pretrain-asl] Preloading done.", flush=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    parser = argparse.ArgumentParser(description="Pre-train CNN backbone on ASL Alphabet dataset")
    parser.add_argument("--data-dir", default="data/asl_alphabet", help="Path to ASL Alphabet folder")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=224, help="Image resize dimension")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of data for validation")
    parser.add_argument("--output", default="saved_models/cnn_backbone_asl_pretrained.pth",
                        help="Where to save pretrained backbone weights")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Max images per class (e.g. 500 to speed up training)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain-asl] Device: {device}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Load dataset
    print(f"[pretrain-asl] Loading dataset from {args.data_dir}...")
    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
    num_classes = len(full_dataset.classes)
    print(f"[pretrain-asl] Found {len(full_dataset)} images, {num_classes} classes: {full_dataset.classes}")

    # Subsample per class if requested
    if args.max_per_class is not None:
        from collections import defaultdict
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(full_dataset.samples):
            class_indices[label].append(idx)
        subset_indices = []
        for label, idxs in class_indices.items():
            if len(idxs) > args.max_per_class:
                generator_sub = torch.Generator().manual_seed(42)
                perm = torch.randperm(len(idxs), generator=generator_sub).tolist()
                idxs = [idxs[i] for i in perm[:args.max_per_class]]
            subset_indices.extend(idxs)
        full_dataset = Subset(full_dataset, subset_indices)
        print(f"[pretrain-asl] Subsampled to {len(full_dataset)} images ({args.max_per_class}/class)")

    # Split into train/val
    total = len(full_dataset)
    val_size = int(total * args.val_split)
    train_size = total - val_size

    # Use fixed seed
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Preload all images into RAM for fast training
    print(f"[pretrain-asl] Train: {len(train_indices)} | Val: {len(val_indices)}")
    train_dataset = PreloadedDataset(full_dataset, train_indices)

    val_dataset_base = datasets.ImageFolder(args.data_dir, transform=val_transform)
    if args.max_per_class is not None:
        val_dataset_base = Subset(val_dataset_base, subset_indices)
    val_dataset = PreloadedDataset(val_dataset_base, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Model CNN backbone + classification head for ASL letters
    backbone = CNNBackbone(in_channels=3, block_channels=[32, 64, 128, 256])
    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(backbone.feature_dim, num_classes),
    )
    model = nn.Sequential(backbone, classifier).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain-asl] Params: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    best_val_acc = 0.0

    print(f"[pretrain-asl] Starting pre-training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_correct, train_total = 0, 0
        train_loss_sum = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            train_loss_sum += loss.item() * labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"    batch {batch_idx+1}/{len(train_loader)}", end="\r")

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

    print(f"\n[pretrain-asl] Done. Best val acc: {best_val_acc:.4f}")
    print(f"[pretrain-asl] Backbone weights saved to: {args.output}")


if __name__ == "__main__":
    main()
