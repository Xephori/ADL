"""Pre-extract and save video frames as .npz (uint8) for fast training."""
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


def preextract(metadata_csv: str, output_dir: str, num_frames: int = 16, image_size: int = 224):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metadata_csv)

    skipped = 0
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting frames"):
        out_path = os.path.join(output_dir, f"{i}.npz")
        if os.path.exists(out_path):
            continue
        try:
            cap = cv2.VideoCapture(row["video_path"])
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {row['video_path']}")
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (image_size, image_size))
                frames.append(frame)
            cap.release()

            if len(frames) == 0:
                raise RuntimeError("Video has no frames")

            # Sample num_frames uniformly
            if len(frames) < num_frames:
                frames = frames + [frames[-1]] * (num_frames - len(frames))
            indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
            sampled = np.stack([frames[j] for j in indices])  # [T, H, W, 3] uint8

            np.savez_compressed(out_path, frames=sampled, label_idx=int(row["label_idx"]))
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
