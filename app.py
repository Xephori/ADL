from __future__ import annotations
import os
import tempfile

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.config import load_config
from src.models import get_model

WEIGHTS    = "saved_models/iter5_asl_pretrain_model_a_best.pth"
CONFIG     = "configs/default.yaml"
METADATA   = "data/metadata.csv"
FRAMES_DIR = "data/frames_cache"
MODEL_NAME = "model_a"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

LABEL_NAMES = {
    0: "accident", 1: "before",  2: "computer", 3: "cool",   4: "cousin",
    5: "drink",    6: "give",    7: "go",        8: "inform", 9: "thin",
}
CLASSES = [LABEL_NAMES[i] for i in range(10)]

EVAL_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cfg = load_config(CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(
    MODEL_NAME, num_classes=10,
    block_channels=cfg.model.cnn_channels,
    lstm_hidden_size=cfg.model.lstm_hidden_size,
    lstm_num_layers=cfg.model.lstm_num_layers,
    lstm_bidirectional=cfg.model.lstm_bidirectional,
    lstm_dropout=cfg.model.lstm_dropout,
    attention_hidden_dim=cfg.model.attention_hidden_dim,
    classifier_dropout=cfg.model.classifier_dropout,
).to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=False))
model.eval()

df_full = pd.read_csv(METADATA)

_ALL = [1102, 1130, 1138, 1125, 1070, 1047, 1098]
DEMO_VIDEOS = [i for i in _ALL if os.path.exists(os.path.join(FRAMES_DIR, f"{i}.pt"))]
NUM_VIDEOS  = len(DEMO_VIDEOS)

def denormalize(t):
    t = t.float() * IMAGENET_STD + IMAGENET_MEAN
    return (t * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

def frames_tensor_to_gif(frames: torch.Tensor) -> str:
    imgs = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(denormalize(frames[i]))
        W, H = img.size
        imgs.append(img.resize((W * 2, H * 2), Image.LANCZOS))
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    imgs[0].save(tmp.name, save_all=True, append_images=imgs[1:], loop=0, duration=120)
    return tmp.name

def make_result_html(probs, pred_idx, true_label=None):
    pred_label = LABEL_NAMES[pred_idx]
    confidence = round(float(probs[pred_idx]) * 100, 1)

    if true_label is not None:
        correct = pred_label == true_label
        border  = "#4caf50" if correct else "#f44336"
        bg      = "#1a3a1a" if correct else "#3a1a1a"
        verdict = f'<div style="margin-top:8px;font-size:1.1em;font-weight:700;color:{border};">{"CORRECT" if correct else "WRONG"}</div>'
        true_line = f'<div style="color:#aaa;font-size:0.85em;margin-top:4px;">True sign: <strong style="color:#fff;">{true_label.upper()}</strong></div>'
    else:
        border  = "#4f8ef7"
        bg      = "#1a2a3a"
        verdict = ""
        true_line = ""

    confidence = round(float(probs[pred_idx]) * 100, 1)
    result = (
        f'<div style="border:2px solid {border};background:{bg};border-radius:10px;padding:20px 24px;margin-bottom:16px;">'
        f'  <div style="color:#aaa;font-size:0.8em;margin-bottom:4px;">PREDICTED SIGN</div>'
        f'  <div style="color:#fff;font-size:2.2em;font-weight:700;letter-spacing:1px;">{pred_label.upper()}</div>'
        f'  <div style="color:#aaa;font-size:0.9em;margin-top:6px;">Confidence: <strong style="color:#fff;">{confidence}%</strong></div>'
        f'  {verdict}{true_line}'
        f'</div>'
    )

    bars = ""
    for i, cls in enumerate(CLASSES):
        pct = round(float(probs[i]) * 100, 1)
        if true_label is not None:
            correct_pred = pred_label == true_label
            color = "#4caf50" if i == pred_idx and correct_pred else "#f44336" if i == pred_idx else "#3a3a3a"
        else:
            color = "#4f8ef7" if i == pred_idx else "#3a3a3a"
        label_color = "#fff" if i == pred_idx else "#aaa"
        bars += (
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
            f'  <div style="width:90px;text-align:right;font-size:1em;color:{label_color};font-weight:{"600" if i==pred_idx else "400"};">{cls}</div>'
            f'  <div style="flex:1;background:#222;border-radius:4px;height:22px;">'
            f'    <div style="width:{pct}%;background:{color};height:100%;border-radius:4px;"></div>'
            f'  </div>'
            f'  <div style="width:46px;font-size:0.95em;color:#aaa;">{pct}%</div>'
            f'</div>'
        )
    confidence_html = (
        f'<div style="padding:8px 0;">'
        f'  <div style="color:#777;font-size:0.8em;letter-spacing:.05em;margin-bottom:14px;">CONFIDENCE</div>'
        f'  {bars}'
        f'</div>'
    )
    return result, confidence_html

def make_gif_preset(video_num: int) -> str:
    frames = torch.load(
        os.path.join(FRAMES_DIR, f"{DEMO_VIDEOS[video_num]}.pt"), weights_only=False
    )["frames"]
    return frames_tensor_to_gif(frames)

@torch.no_grad()
def run_predict_preset(video_num: int):
    row_idx    = DEMO_VIDEOS[video_num]
    true_label = df_full.iloc[row_idx]["label"]
    frames     = torch.load(os.path.join(FRAMES_DIR, f"{row_idx}.pt"), weights_only=False)["frames"]
    probs      = F.softmax(model(frames.float().unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    pred_idx   = int(probs.argmax())
    return make_result_html(probs, pred_idx, true_label)

def extract_frames_from_video(video_path: str, num_frames: int = 16):
    cap = cv2.VideoCapture(video_path)
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(raw) == 0:
        raise ValueError("Could not read any frames from video")
    # pad if too short
    while len(raw) < num_frames:
        raw.append(raw[-1])
    indices = np.linspace(0, len(raw) - 1, num_frames).astype(int)
    sampled = [raw[i] for i in indices]
    tensors = [EVAL_TRANSFORM(f) for f in sampled]
    return torch.stack(tensors), sampled

def preview_upload(video_path: str):
    if video_path is None:
        return None
    _, raw_frames = extract_frames_from_video(video_path)
    # Build GIF from raw (un-normalized) frames
    imgs = []
    for f in raw_frames:
        img = Image.fromarray(f)
        W, H = img.size
        imgs.append(img.resize((W * 2, H * 2), Image.LANCZOS))
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    imgs[0].save(tmp.name, save_all=True, append_images=imgs[1:], loop=0, duration=120)
    return tmp.name

@torch.no_grad()
def run_predict_upload(video_path: str):
    if video_path is None:
        return "", ""
    frames_tensor, _ = extract_frames_from_video(video_path)
    probs    = F.softmax(model(frames_tensor.unsqueeze(0).to(device)), dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    # No true label for uploaded videos
    return make_result_html(probs, pred_idx, true_label=None)

css = """
* { font-family: ui-sans-serif, system-ui, sans-serif !important; }
footer { display: none !important; }
#gif-box { height: 540px !important; }
#gif-box img { height: 100% !important; width: 100% !important; object-fit: contain !important; }
"""

with gr.Blocks(title="Sign Language Word Recognition", theme=gr.themes.Base(), css=css) as demo:

    gr.Markdown("# Sign Language Word Recognition")

    with gr.Row():
        with gr.Column(scale=1):
            upload_in = gr.Video(label="Upload a signing video", sources=["upload"], loop=True, autoplay=True)
            run_btn   = gr.Button("Run Model", variant="primary", visible=False)

        with gr.Column(scale=1):
            result_out     = gr.HTML()
            confidence_out = gr.HTML()

    def on_upload(video_path):
        if video_path is None:
            return gr.update(visible=False), "", ""
        return gr.update(visible=True), "", ""

    upload_in.change(fn=on_upload, inputs=upload_in, outputs=[run_btn, result_out, confidence_out])
    run_btn.click(fn=run_predict_upload, inputs=upload_in, outputs=[result_out, confidence_out])

if __name__ == "__main__":
    demo.launch(share=False)
