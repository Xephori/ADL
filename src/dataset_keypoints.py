#Dataset and dataloader for MediaPipe keypoint sequences.

from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class KeypointDataset(Dataset):
    def __init__(
        self,
        indices: List[int],
        cache_dir: str,
        label_remap: Dict[int, int],
    ) -> None:
        self.indices = indices
        self.cache_dir = cache_dir

        print(f"[kp-dataset] Loading {len(indices)} keypoint files into RAM...", flush=True)

        def _load(i):
            d = torch.load(os.path.join(cache_dir, f"{i}.pt"), weights_only=False)
            return (d["keypoints"], int(d["label_idx"]))

        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded = list(pool.map(_load, indices))

        self.all_keypoints = torch.stack([x[0] for x in loaded]) 
        raw_labels = [x[1] for x in loaded]
        self.all_labels = torch.tensor(
            [label_remap[l] for l in raw_labels], dtype=torch.long
        )
        del loaded
        print(f"[kp-dataset] Loaded {len(indices)} files.", flush=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.all_keypoints[idx], self.all_labels[idx].item()


class FastKeypointLoader:

    def __init__(
        self,
        keypoints: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        self.keypoints = keypoints
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.labels) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.labels)
        idx = torch.randperm(n) if self.shuffle else torch.arange(n)
        for i in range(0, n, self.batch_size):
            batch_idx = idx[i : i + self.batch_size]
            yield self.keypoints[batch_idx], self.labels[batch_idx]


def create_keypoint_dataloaders(
    metadata_csv: str,
    batch_size: int = 8,
    top_n_classes: int = 0,
    cache_dir: str = "data/keypoints_cache",
) -> Tuple[FastKeypointLoader, FastKeypointLoader, FastKeypointLoader, int, dict]:
    if not os.path.isdir(cache_dir) or len(os.listdir(cache_dir)) == 0:
        raise RuntimeError(
            f"Keypoint cache not found at '{cache_dir}'. "
            "Run: python -m src.extract_keypoints"
        )

    df = pd.read_csv(metadata_csv)

    if top_n_classes > 0:
        train_counts = df[df["split"] == "train"]["label"].value_counts()
        top_labels = train_counts.head(top_n_classes).index.tolist()
        df = df[df["label"].isin(top_labels)]
        print(f"[kp-dataset] Filtered to top {top_n_classes} classes: {len(df)} samples")

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    labels_sorted = sorted(df["label"].unique())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels_sorted)}
    num_classes = len(labels_sorted)

    old_to_new: Dict[int, int] = {}
    for _, row in df.iterrows():
        old_idx = int(row["label_idx"])
        new_idx = label_to_idx[row["label"]]
        old_to_new[old_idx] = new_idx

    train_ds = KeypointDataset(train_df.index.tolist(), cache_dir, old_to_new)
    val_ds = KeypointDataset(val_df.index.tolist(), cache_dir, old_to_new)
    test_ds = KeypointDataset(test_df.index.tolist(), cache_dir, old_to_new)

    train_loader = FastKeypointLoader(train_ds.all_keypoints, train_ds.all_labels, batch_size, shuffle=True)
    val_loader = FastKeypointLoader(val_ds.all_keypoints, val_ds.all_labels, batch_size, shuffle=False)
    test_loader = FastKeypointLoader(test_ds.all_keypoints, test_ds.all_labels, batch_size, shuffle=False)

    print(
        f"[kp-dataset] Train: {len(train_ds)} | Val: {len(val_ds)} | "
        f"Test: {len(test_ds)} | Classes: {num_classes}"
    )
    return train_loader, val_loader, test_loader, num_classes, label_to_idx
