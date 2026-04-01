"""
Extract MediaPipe hand keypoints from pre-extracted frame cache.
Reads .pt files from data/frames_cache/ (produced by preextract_frames.py),
de-normalises the tensors back to uint8, runs MediaPipe Hands on each frame.

Saves one .pt file per video into data/keypoints_cache/:
  {
    "keypoints": tensor[16, 63],   # 16 frames × 21 landmarks × (x,y,z)
    "label_idx": int
  }

Normalisation: wrist (landmark 0) is subtracted from all landmarks so that
hand position in the frame doesn't matter — only hand shape/pose does.

Run once before training:
    python -m src.extract_keypoints
"""
from __future__ import annotations
import argparse
import os
import sys

import mediapipe as mp
import numpy as np
import pandas as pd
import torch


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_frame(frame_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a stored frame tensor back to a uint8 HxWx3 RGB numpy array.
    Stored frames are float16, ImageNet-normalised, shape [3, H, W].
    """
    f = frame_tensor.float()                        # float16 → float32
    f = f * IMAGENET_STD + IMAGENET_MEAN            # undo normalisation → [0, 1] approx
    f = (f * 255.0).clamp(0, 255).byte()            # → uint8
    return f.permute(1, 2, 0).numpy()               # [3,H,W] → [H,W,3]


def extract_keypoints_from_cache(
    frames_tensor: torch.Tensor,   # [16, 3, H, W]  float16
    hands_detector,
) -> torch.Tensor:
    """
    Run MediaPipe Hands on each de-normalised frame.
    Returns tensor [16, 63].
    If no hand detected in a frame, that frame is zeros.
    All frames normalised: subtract wrist landmark (landmark 0).
    """
    num_frames = frames_tensor.shape[0]
    keypoints = []

    for i in range(num_frames):
        frame_rgb = denormalize_frame(frames_tensor[i])   # [H, W, 3] uint8
        result = hands_detector.process(frame_rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)  # [21, 3]
            coords = coords - coords[0]   # subtract wrist → translation-invariant
        else:
            coords = np.zeros((21, 3), dtype=np.float32)

        keypoints.append(coords.flatten())   # [63]

    return torch.tensor(np.stack(keypoints), dtype=torch.float32)   # [16, 63]


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints from frames cache")
    parser.add_argument("--metadata-csv", default="data/metadata.csv")
    parser.add_argument("--frames-cache", default="data/frames_cache", help="Input: pre-extracted frames")
    parser.add_argument("--output-dir", default="data/keypoints_cache", help="Output: keypoint .pt files")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if .pt already exists")
    args = parser.parse_args()

    if not os.path.exists(args.metadata_csv):
        print(f"[keypoints] ERROR: metadata.csv not found at {args.metadata_csv}")
        sys.exit(1)

    if not os.path.isdir(args.frames_cache):
        print(f"[keypoints] ERROR: frames cache not found at {args.frames_cache}")
        print("[keypoints] Run preextract_frames.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.metadata_csv)
    print(f"[keypoints] Found {len(df)} rows in metadata.csv")
    print(f"[keypoints] Reading frames from: {args.frames_cache}")
    print(f"[keypoints] Saving keypoints to: {args.output_dir}")

    try:
        # MediaPipe >= 0.10 new API
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        hands = None
        use_new_api = True
    except Exception:
        use_new_api = False

    if not use_new_api:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
        )
    else:
        # Fall back: try legacy path via mp.solutions
        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                min_detection_confidence=0.3,
            )
            use_new_api = False
        except Exception as e:
            print(f"[keypoints] ERROR: Could not initialise MediaPipe Hands: {e}")
            print("[keypoints] Try: pip install mediapipe==0.10.9")
            sys.exit(1)

    errors = 0
    no_hand_count = 0

    for i in range(len(df)):
        out_path = os.path.join(args.output_dir, f"{i}.pt")
        if not args.overwrite and os.path.exists(out_path):
            continue

        frames_path = os.path.join(args.frames_cache, f"{i}.pt")
        label_idx = int(df.iloc[i]["label_idx"])

        if not os.path.exists(frames_path):
            print(f"[keypoints] WARNING: frame cache missing for row {i}: {frames_path}")
            errors += 1
            torch.save({
                "keypoints": torch.zeros(16, 63),
                "label_idx": label_idx,
            }, out_path)
            continue

        d = torch.load(frames_path, weights_only=False)
        frames_tensor = d["frames"]   # [16, 3, H, W] float16

        kp = extract_keypoints_from_cache(frames_tensor, hands)

        # Warn if model never saw a hand in this video
        zero_frames = (kp.abs().sum(dim=1) == 0).sum().item()
        if zero_frames == 16:
            no_hand_count += 1
            if no_hand_count <= 10:
                print(f"[keypoints] WARNING: no hand detected in any frame — row {i} ({df.iloc[i]['label']})")

        torch.save({
            "keypoints": kp,
            "label_idx": label_idx,
        }, out_path)

        if (i + 1) % 100 == 0:
            total = len(df)
            print(f"[keypoints] {i+1}/{total} processed ...", flush=True)

    hands.close()

    total_saved = len([f for f in os.listdir(args.output_dir) if f.endswith(".pt")])
    print(f"\n[keypoints] Done.")
    print(f"[keypoints] .pt files saved: {total_saved}")
    print(f"[keypoints] Missing frame cache files: {errors}")
    print(f"[keypoints] Videos with no hand detected: {no_hand_count}")


if __name__ == "__main__":
    main()
