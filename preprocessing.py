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

# Download latest version
path = kagglehub.dataset_download("waseemnagahhenes/sign-language-dataset-wlasl-videos", output_dir=".\\data", force_download=True)
VIDEO_DIR = os.path.join(path, "dataset/SL")
print("Path to dataset files:", path)

data = []
for label in os.listdir(VIDEO_DIR):
    label_dir = os.path.join(VIDEO_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for video_file in os.listdir(label_dir):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(label_dir, video_file)
            data.append({
                "video_path": video_path,
                "label": label
            })

df = pd.DataFrame(data)
print(df.head())

labels = sorted(df["label"].unique())
label_to_idx = {label: i for i, label in enumerate(labels)}
df["label_idx"] = df["label"].map(label_to_idx)
num_classes = len(labels)
print("Number of classes:", num_classes)

# Check class distribution
print("\nClass distribution:")
class_counts = df["label_idx"].value_counts().sort_index()
print(class_counts)
print(f"\nClasses with only 1 sample: {sum(class_counts == 1)}")
print(f"Classes with 2-3 samples: {sum((class_counts >= 2) & (class_counts <= 3))}")

# Split the data - first split with stratification
train_df, temp_df = train_test_split(
    df,
    test_size=0.25,
    stratify=df["label_idx"],
    random_state=42
)

# Check temp_df class distribution
print(f"\nAfter first split, temp_df class distribution:")
temp_class_counts = temp_df["label_idx"].value_counts().sort_index()
print(f"Classes with only 1 sample in temp_df: {sum(temp_class_counts == 1)}")

# Second split - remove stratification to avoid the error
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.4,  # 40% of temp_df = 10% of total for test, 60% of temp_df = 15% of total for val
    random_state=42
)

# Print split statistics
print(f"\nFinal split sizes:")
print(f"Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
print(f"Total: {len(train_df) + len(val_df) + len(test_df)} samples")

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames

def sample_frames(frames, num_frames=16):
    total = len(frames)
    if total == 0:
        raise ValueError("Video has no frames")
    if total < num_frames:
        frames += [frames[-1]] * (num_frames - total)
    indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    sampled = [frames[i] for i in indices]
    return sampled

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class WLASLDataset(Dataset):

    def __init__(self, dataframe, transform=None, num_frames=16):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        video_path = self.df.loc[idx, "video_path"]
        label = self.df.loc[idx, "label_idx"]
        frames = extract_frames(video_path)
        frames = sample_frames(frames, self.num_frames)
        processed = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            processed.append(frame)
        video_tensor = torch.stack(processed)
        return video_tensor, label
    
train_dataset = WLASLDataset(
    train_df,
    transform=train_transform
)

val_dataset = WLASLDataset(
    val_df,
    transform=test_transform
)

test_dataset = WLASLDataset(
    test_df,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
