"""Evaluate a trained checkpoint and render side-by-side GT / prediction grids.

Usage:
    python scripts/evaluate.py --weights runs/detect/figlib_smoke_v11n/weights/best.pt \
        --data data/yolo --split val --out results/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_names(data_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names", [])
    return names if isinstance(names, list) else [names[k] for k in sorted(names)]


def read_gt(label_path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append(
            (
                int((cx - bw / 2) * w),
                int((cy - bh / 2) * h),
                int((cx + bw / 2) * w),
                int((cy + bh / 2) * h),
            )
        )
    return boxes


def panel(image: np.ndarray, boxes, title: str, color) -> np.ndarray:
    canvas = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
    bar = np.full((38, canvas.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(bar, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    height = max(t.shape[0] for t in tiles)
    width = max(t.shape[1] for t in tiles)
    padded = [
        cv2.copyMakeBorder(
            t,
            0,
            height - t.shape[0],
            0,
            width - t.shape[1],
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        for t in tiles
    ]
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i : i + cols]
        while len(row) < cols:
            row.append(np.full_like(padded[0], 255))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/yolo")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--val-conf",
        type=float,
        default=0.001,
        help="confidence used when computing the confusion matrix",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--tile", type=int, default=520)
    parser.add_argument("--max", type=int, default=12)
    parser.add_argument("--out", default="results/eval")
    args = parser.parse_args()

    from ultralytics import YOLO

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = load_names(root / "data.yaml")

    model = YOLO(args.weights)

    # ---- quantitative -------------------------------------------------
    metrics = model.val(
        data=str(root / "data.yaml"),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.val_conf,
        iou=0.6,
        device=args.device,
        plots=True,
        project=str(out),
        name="val",
        exist_ok=True,
    )
    summary = {
        "split": args.split,
        "images": int(len(list((root / "images" / args.split).glob("*.jpg")))),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "classes": names,
        "speed_ms": {k: round(float(v), 2) for k, v in (metrics.speed or {}).items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # ---- qualitative: GT | prediction ---------------------------------
    images = sorted((root / "images" / args.split).glob("*.jpg"))
    wanted = [
        p for p in images if (root / "labels" / args.split / f"{p.stem}.txt").stat().st_size > 0
    ]
    picks = wanted[: args.max]

    tiles = []
    for idx, path in enumerate(picks):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gt = read_gt(root / "labels" / args.split / f"{path.stem}.txt", w, h)
        result = model.predict(
            str(path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        pred = [tuple(int(v) for v in b) for b in result.boxes.xyxy.cpu().numpy()]
        left = panel(img, gt, f"GT  {len(gt)} box", (0, 160, 0))
        right = panel(img, pred, f"PRED  {len(pred)} box  conf>={args.conf}", (0, 100, 255))
        pair_raw = np.hstack([left, right])
        cv2.imwrite(
            str(out / f"pair_{idx:02d}_{path.stem}.jpg"),
            pair_raw,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90],
        )
        scale = args.tile / pair_raw.shape[1]
        tiles.append(cv2.resize(pair_raw, (args.tile, int(pair_raw.shape[0] * scale))))

    if tiles:
        sheet = grid(tiles, 1)
        cv2.imwrite(str(out / "gt_vs_pred.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"wrote {out / 'gt_vs_pred.jpg'} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
