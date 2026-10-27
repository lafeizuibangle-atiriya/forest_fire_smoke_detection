"""Train the YOLO smoke/fire detector.

Usage:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train_figlib.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsd.core.config import load_train_config  # noqa: E402
from ffsd.core.logging_utils import get_logger  # noqa: E402

log = get_logger("train")


def _check_device(device: str) -> None:
    """Fail fast with an actionable message when CUDA is requested but unusable."""
    import torch

    log.info("torch=%s built_for_cuda=%s", torch.__version__, torch.version.cuda)
    if device == "cpu":
        log.warning("device=cpu, training will be slow")
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but CUDA is unavailable. "
            f"torch={torch.__version__} (a '+cpu' build cannot use the GPU). "
            "Install a CUDA build, e.g. "
            "pip install --index-url https://download.pytorch.org/whl/cu121 "
            "torch==2.4.1+cu121 torchvision==0.19.1+cu121"
        )
    props = torch.cuda.get_device_properties(0)
    log.info(
        "using GPU: %s (%.1f GB, sm_%d%d)",
        props.name,
        props.total_memory / 1024**3,
        props.major,
        props.minor,
    )


def _dump_metrics(metrics: object, run_dir: Path) -> None:
    """Persist validation metrics as JSON so reports never depend on stdout."""
    box = metrics.box
    payload = {
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": {},
    }
    names = getattr(metrics, "names", {}) or {}
    for i, ap in enumerate(box.ap50):
        payload["per_class"][names.get(i, str(i))] = {
            "mAP50": round(float(ap), 4),
            "mAP50-95": round(float(box.ap[i]), 4),
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
        }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("metrics -> %s", run_dir / "metrics.json")
    log.info(
        "mAP50=%.4f mAP50-95=%.4f precision=%.4f recall=%.4f",
        payload["mAP50"],
        payload["mAP50-95"],
        payload["precision"],
        payload["recall"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--resume", action="store_true", help="continue from {project}/{name}/weights/last.pt"
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    _check_device(cfg.train.device)

    from ultralytics import YOLO

    if args.resume:
        last = Path(cfg.train.project) / cfg.train.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last} not found")
        log.info("resuming from %s", last)
        model = YOLO(str(last))
        model.train(resume=True)
        return

    log.info("loading pretrained weights: %s", cfg.model.weights)
    model = YOLO(cfg.model.weights)
    model.train(
        data=cfg.data.yaml,
        epochs=cfg.train.epochs,
        imgsz=cfg.train.imgsz,
        batch=cfg.train.batch,
        patience=cfg.train.patience,
        name=cfg.train.name,
        project=cfg.train.project,
        device=cfg.train.device,
        workers=cfg.train.workers,
        amp=cfg.train.amp,
        cache=cfg.train.cache,
        mosaic=cfg.train.mosaic,
        copy_paste=cfg.train.copy_paste,
        scale=cfg.train.scale,
        mixup=cfg.train.mixup,
        multi_scale=cfg.train.multi_scale,
        exist_ok=True,
    )

    metrics = model.val()
    run_dir = Path(model.trainer.save_dir)
    _dump_metrics(metrics, run_dir)


if __name__ == "__main__":
    main()
