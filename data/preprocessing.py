import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import kagglehub

# Download dataset
path = kagglehub.dataset_download("waseemnagahhenes/sign-language-dataset-wlasl-videos")

# Find the video directory
VIDEO_DIR = os.path.join(path, "dataset", "SL")
if not os.path.exists(VIDEO_DIR):
    VIDEO_DIR = os.path.join(path, "SL")
print("Path to dataset files:", path)
print("Video directory:", VIDEO_DIR)

data = []
for label in sorted(os.listdir(VIDEO_DIR)):
    label_dir = os.path.join(VIDEO_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for video_file in sorted(os.listdir(label_dir)):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(label_dir, video_file)
            data.append({
                "video_path": video_path,
                "label": label
            })

df = pd.DataFrame(data)
print(f"\nTotal videos found: {len(df)}")
print(f"Total classes found: {df['label'].nunique()}")


# Filter to top 100 classes (WLASL-100 subset)
# Pick the 100 classes with the most video samples
NUM_CLASSES = 100
class_counts = df["label"].value_counts()
top_classes = class_counts.head(NUM_CLASSES).index.tolist()
df = df[df["label"].isin(top_classes)].reset_index(drop=True)
print(f"\nAfter filtering to top {NUM_CLASSES} classes:")
print(f"  Videos: {len(df)}")
print(f"  Classes: {df['label'].nunique()}")

# Create label index mapping
labels = sorted(df["label"].unique())
label_to_idx = {label: i for i, label in enumerate(labels)}
df["label_idx"] = df["label"].map(label_to_idx)
num_classes = len(labels)
print(f"  Number of classes: {num_classes}")

# Class distribution
print("\nClass distribution:")
class_counts = df["label_idx"].value_counts().sort_index()
print(f"  Min samples per class: {class_counts.min()}")
print(f"  Max samples per class: {class_counts.max()}")
print(f"  Mean samples per class: {class_counts.mean():.1f}")
print(f"  Classes with only 1 sample: {sum(class_counts == 1)}")
print(f"  Classes with 2-3 samples: {sum((class_counts >= 2) & (class_counts <= 3))}")

# Split: train (75%) val (15%) test (10%)
rare_classes = set(class_counts[class_counts < 4].index.tolist())
rare_mask = df["label_idx"].isin(rare_classes)
df_rare = df[rare_mask].copy()
df_common = df[~rare_mask].copy()

print(f"\n  Common classes (>= 4 samples): {df_common['label_idx'].nunique()} classes, {len(df_common)} videos")
print(f"  Rare classes (< 4 samples): {df_rare['label_idx'].nunique()} classes, {len(df_rare)} videos")

# Stratified split on common classes
train_df, temp_df = train_test_split(
    df_common,
    test_size=0.25,
    stratify=df_common["label_idx"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.4,
    stratify=temp_df["label_idx"],
    random_state=42
)

# Add split column
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()
train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"


df_rare = df_rare.copy()
df_rare["split"] = "train"


full_df = pd.concat([train_df, val_df, test_df, df_rare], ignore_index=True)


print(f"\nFinal split sizes:")
for split_name in ["train", "val", "test"]:
    n = (full_df["split"] == split_name).sum()
    print(f"  {split_name}: {n} samples ({n/len(full_df)*100:.1f}%)")
print(f"  Total: {len(full_df)} samples")

# Save metadata CSV and label map
script_dir = os.path.dirname(os.path.abspath(__file__))
metadata_csv = os.path.join(script_dir, "metadata.csv")
label_map_csv = os.path.join(script_dir, "label_map.csv")

full_df.to_csv(metadata_csv, index=False)
print(f"\nMetadata saved → {metadata_csv}")

label_map_df = pd.DataFrame(list(label_to_idx.items()), columns=["label", "label_idx"])
label_map_df.to_csv(label_map_csv, index=False)
print(f"Label map saved → {label_map_csv}")
