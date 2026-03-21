"""Pre-extract and save video frames as .pt tensors for fast training."""
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from src.dataset import extract_frames, sample_frames, get_eval_transform

def preextract(metadata_csv: str, output_dir: str, num_frames: int = 16, image_size: int = 224):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metadata_csv)
    resize = get_eval_transform(image_size)

    skipped = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting frames"):
        out_path = os.path.join(output_dir, f"{i}.pt")
        if os.path.exists(out_path):
            continue
        try:
            frames = extract_frames(row["video_path"])
            frames = sample_frames(frames, num_frames, jitter=False)
            tensors = [resize(f) for f in frames]
            video_tensor = torch.stack(tensors)  # [T, 3, H, W]
            torch.save({"frames": video_tensor, "label_idx": int(row["label_idx"])}, out_path)
        except Exception as e:
            print(f"  Skipping {row['video_path']}: {e}")
            skipped += 1

    print(f"Done. Extracted {len(df) - skipped}/{len(df)} videos. Saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--output-dir", default="data/frames_cache")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    preextract(args.metadata, args.output_dir, args.num_frames, args.image_size)
