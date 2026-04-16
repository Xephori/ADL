from __future__ import annotations
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Frame extraction helpers

def extract_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def sample_frames(
    frames: List[np.ndarray],
    num_frames: int = 16,
    jitter: bool = False,
) -> List[np.ndarray]:
    total = len(frames)
    if total == 0:
        raise ValueError("Video has no frames")

    if total < num_frames:
        frames = frames + [frames[-1]] * (num_frames - total)
        total = len(frames)

    max_offset = max(total - num_frames, 0)
    if jitter and max_offset > 0:
        start = random.randint(0, max_offset)
        frames = frames[start:]
        total = len(frames)

    indices = np.linspace(0, total - 1, num_frames).astype(int)
    return [frames[i] for i in indices]

# Transforms

def get_train_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# Dataset

class WLASLDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[T.Compose] = None,
        num_frames: int = 16,
        jitter: bool = False,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.num_frames = num_frames
        self.jitter = jitter

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        video_path: str = row["video_path"]
        label: int = int(row["label_idx"])

        try:
            frames = extract_frames(video_path)
        except RuntimeError:
            # Skip corrupted video, return a random other sample
            return self.__getitem__((idx + 1) % len(self))
        frames = sample_frames(frames, self.num_frames, jitter=self.jitter)

        processed: List[torch.Tensor] = []
        for frame in frames:
            if self.transform is not None:
                frame = self.transform(frame)
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            processed.append(frame)

        video_tensor = torch.stack(processed)  
        return video_tensor, label

# Cached dataset

class CachedWLASLDataset(Dataset):
    def __init__(self, indices: List[int], cache_dir: str, train_aug: bool = False):
        self.indices = indices
        self.cache_dir = cache_dir
        self.train_aug = train_aug
        self.mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float16).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, dtype=torch.float16).view(1, 3, 1, 1)
        self.train_flip = train_aug
        from concurrent.futures import ThreadPoolExecutor
        print(f"[dataset] Loading {len(indices)} cached files into RAM...")
        def _load(i):
            d = torch.load(os.path.join(cache_dir, f"{i}.pt"), weights_only=False)
            return (d["frames"], int(d["label_idx"]))
        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded = list(pool.map(_load, indices))
        self.all_frames = torch.stack([x[0] for x in loaded])
        self.all_labels = torch.tensor([x[1] for x in loaded], dtype=torch.long)
        del loaded
        print(f"[dataset] All {len(indices)} files loaded into RAM.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        frames = self.all_frames[idx] 
        label = self.all_labels[idx].item()
        if self.train_flip and torch.rand(1).item() > 0.5:
            frames = frames.flip(-1)
        return frames, label

# Fast batching

class FastTensorLoader:
    def __init__(self, frames: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool = False):
        self.frames = frames
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.labels)
        if self.shuffle:
            idx = torch.randperm(n)
        else:
            idx = torch.arange(n)
        for i in range(0, n, self.batch_size):
            batch_idx = idx[i:i+self.batch_size]
            yield self.frames[batch_idx], self.labels[batch_idx]

# DataLoader factory

def create_dataloaders(
    metadata_csv: str,
    num_frames: int = 16,
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 0,
    cache_dir: str = "data/frames_cache",
    top_n_classes: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, dict]:
    df = pd.read_csv(metadata_csv)

    if top_n_classes > 0:
        train_counts = df[df["split"] == "train"]["label"].value_counts()
        top_labels = train_counts.head(top_n_classes).index.tolist()
        df = df[df["label"].isin(top_labels)]
        print(f"[dataset] Filtered to top {top_n_classes} classes: {len(df)} samples")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    labels_sorted = sorted(df["label"].unique())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels_sorted)}
    num_classes = len(labels_sorted)

    old_to_new = {}
    for _, row in df.iterrows():
        lbl = row["label"]
        old_idx = int(row["label_idx"])
        if lbl in label_to_idx and old_idx not in old_to_new:
            old_to_new[old_idx] = label_to_idx[lbl]

    use_cache = os.path.isdir(cache_dir) and len(os.listdir(cache_dir)) > 0
    if use_cache:
        print(f"[dataset] Using pre-extracted frames from {cache_dir}")
        train_dataset = CachedWLASLDataset(train_df.index.tolist(), cache_dir, train_aug=True)
        val_dataset = CachedWLASLDataset(val_df.index.tolist(), cache_dir, train_aug=False)
        test_dataset = CachedWLASLDataset(test_df.index.tolist(), cache_dir, train_aug=False)
        for ds in [train_dataset, val_dataset, test_dataset]:
            ds.all_labels = torch.tensor([old_to_new[l.item()] for l in ds.all_labels], dtype=torch.long)
        train_loader = FastTensorLoader(train_dataset.all_frames, train_dataset.all_labels, batch_size, shuffle=True)
        val_loader = FastTensorLoader(val_dataset.all_frames, val_dataset.all_labels, batch_size, shuffle=False)
        test_loader = FastTensorLoader(test_dataset.all_frames, test_dataset.all_labels, batch_size, shuffle=False)
    else:
        print(f"[dataset] No frame cache found — extracting from video on the fly (slow)")
        # Remap label_idx to contiguous 0..num_classes-1 using the same mapping as cache path
        df = df.copy()
        df["label_idx"] = df["label"].map(label_to_idx)
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        train_transform = get_train_transform(image_size)
        eval_transform = get_eval_transform(image_size)
        train_dataset = WLASLDataset(train_df, transform=train_transform, num_frames=num_frames, jitter=True)
        val_dataset = WLASLDataset(val_df, transform=eval_transform, num_frames=num_frames, jitter=False)
        test_dataset = WLASLDataset(test_df, transform=eval_transform, num_frames=num_frames, jitter=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    n_train = train_dataset.all_frames.shape[0] if use_cache else len(train_dataset)
    n_val = val_dataset.all_frames.shape[0] if use_cache else len(val_dataset)
    n_test = test_dataset.all_frames.shape[0] if use_cache else len(test_dataset)
    print(f"[dataset] Train: {n_train} | Val: {n_val} | Test: {n_test} | Classes: {num_classes}")
    return train_loader, val_loader, test_loader, num_classes, label_to_idx
