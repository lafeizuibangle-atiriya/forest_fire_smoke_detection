"""Typed config loading.

Design: configs live in YAML, validated by pydantic at startup so a missing
or malformed field fails fast instead of crashing halfway through a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelCfg(BaseModel):
    weights: str = "yolo11n.pt"
    conf: float = 0.4
    iou: float = 0.3
    # 推理设备："" = 交给 ultralytics 自动选择（有 CUDA 即用 GPU），
    # "0" = cuda:0，"cpu" = 强制 CPU
    device: str = ""


class DataCfg(BaseModel):
    yaml: str


class TrainCfg(BaseModel):
    epochs: int = 200
    imgsz: int = 640
    batch: int = 16
    patience: int = 30
    name: str = "run"
    # Ultralytics 的 run 输出根目录，训练产物落在 {project}/{name}/
    project: str = "runs/detect"
    # "0" 表示 cuda:0，"0,1" 多卡，写 "cpu" 则强制用 CPU
    device: str = "0"
    workers: int = 8
    # AMP 混合精度：显存约减半、速度提升，YOLO 训练默认开启，几乎无精度损失
    amp: bool = True
    # 把小数据集预加载进内存可显著提速，代价是吃 RAM
    cache: bool = False
    # ---- 数据增强（针对烟雾目标的调优空间）----
    # Mosaic：4 图拼接，对小目标召回帮助最大
    mosaic: float = 1.0
    # Copy-Paste：把实例贴到其他图上，制造更多小目标样本
    copy_paste: float = 0.0
    # 尺度抖动幅度（相对 ±0.5 即 0.5x~1.5x）
    scale: float = 0.5
    # MixUp：两图按比例混合，提升鲁棒性，烟雾任务常用 0~0.1
    mixup: float = 0.0
    # 多尺度训练：每个 batch 的 imgsz 在 0.5x~1.5x 之间随机波动。
    # EDA 显示目标宽度跨度 28 倍（24px~668px），固定 imgsz 无法同时覆盖
    multi_scale: bool = False


class TrainConfig(BaseModel):
    model: ModelCfg
    data: DataCfg
    train: TrainCfg = Field(default_factory=TrainCfg)


class SourceCfg(BaseModel):
    kind: str = "video"
    path: str = ""


class DecisionCfg(BaseModel):
    window: int = 10
    min_hits: int = 6


class OutputCfg(BaseModel):
    save_dir: str = "results/infer"


class InferConfig(BaseModel):
    model: ModelCfg
    source: SourceCfg
    decision: DecisionCfg = Field(default_factory=DecisionCfg)
    output: OutputCfg = Field(default_factory=OutputCfg)


class ServerCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class NotifierCfg(BaseModel):
    channels: list[str] = Field(default_factory=lambda: ["console"])


class AppConfig(BaseModel):
    model: ModelCfg
    server: ServerCfg = Field(default_factory=ServerCfg)
    decision: DecisionCfg = Field(default_factory=DecisionCfg)
    notifier: NotifierCfg = Field(default_factory=NotifierCfg)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, raising a clear error if it is missing."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"config not found: {file}")
    with file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_train_config(path: str | Path = "configs/train.yaml") -> TrainConfig:
    return TrainConfig(**load_yaml(path))


def load_infer_config(path: str | Path = "configs/infer.yaml") -> InferConfig:
    return InferConfig(**load_yaml(path))


def load_app_config(path: str | Path = "configs/app.yaml") -> AppConfig:
    return AppConfig(**load_yaml(path))
