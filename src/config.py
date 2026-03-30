#config system loading hyperparameters
from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml

#data hyperparameters
@dataclass
class DataConfig:
    video_dir: str = "data/dataset/SL"
    num_frames: int = 16
    image_size: int = 224
    num_workers: int = 2
    seed: int = 42

#model hyperparameters
@dataclass
class ModelConfig:
    name: str = "model_c"
    num_classes: int = 100
    cnn_blocks: int = 4
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    feature_dim: int = 256
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True
    lstm_dropout: float = 0.3
    attention_hidden_dim: int = 128
    classifier_dropout: float = 0.4
    pretrained_cnn_path: str = ""
    freeze_cnn: bool = False

#training hyperparameters
@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 80
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    step_size: int = 20
    step_gamma: float = 0.5
    early_stopping_patience: int = 12
    use_class_weights: bool = True
    grad_clip: float = 0.0
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0

#file path for saving configs
@dataclass
class PathsConfig:
    save_dir: str = "saved_models"
    results_dir: str = "results"
    log_csv: str = "results/training_log.csv"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _update_dataclass(dc: object, d: dict) -> None:
    for key, value in d.items():
        if hasattr(dc, key):
            setattr(dc, key, value)


def load_config(yaml_path: Optional[str] = None) -> Config:
    cfg = Config()

    if yaml_path is not None:
        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        if raw:
            if "data" in raw:
                _update_dataclass(cfg.data, raw["data"])
            if "model" in raw:
                _update_dataclass(cfg.model, raw["model"])
            if "training" in raw:
                _update_dataclass(cfg.training, raw["training"])
            if "paths" in raw:
                _update_dataclass(cfg.paths, raw["paths"])

    return cfg


def get_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train sign language classification models"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name override: model_a | model_b | model_c")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate override")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max epochs override")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size override")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Frames per video override")
    parser.add_argument("--lstm-hidden", type=int, default=None,
                        help="LSTM hidden size override")
    parser.add_argument("--lstm-layers", type=int, default=None,
                        help="LSTM num layers override")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Classifier dropout override")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom name for this run (used in log filenames)")
    parser.add_argument("--pretrained-cnn", type=str, default=None,
                        help="Path to pretrained CNN backbone weights (.pth)")
    parser.add_argument("--freeze-cnn", action="store_true", default=False,
                        help="Freeze CNN backbone weights during training")
    return parser


def apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.model is not None:
        cfg.model.name = args.model
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.num_frames is not None:
        cfg.data.num_frames = args.num_frames
    if args.lstm_hidden is not None:
        cfg.model.lstm_hidden_size = args.lstm_hidden
    if args.lstm_layers is not None:
        cfg.model.lstm_num_layers = args.lstm_layers
    if args.dropout is not None:
        cfg.model.classifier_dropout = args.dropout
    if getattr(args, 'pretrained_cnn', None) is not None:
        cfg.model.pretrained_cnn_path = args.pretrained_cnn
    if getattr(args, 'freeze_cnn', False):
        cfg.model.freeze_cnn = True
    return cfg
