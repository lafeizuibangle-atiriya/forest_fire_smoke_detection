"""Render a grid of YOLO labels on top of their images for a sanity check.

A wrong coordinate convention (absolute vs normalized, x1y1x2y2 vs cxcywh)
is obvious the moment you look at the boxes, so this runs before training.

Usage:
    python scripts/preview_dataset.py --data data/yolo --out results/labels_preview.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_names(data_yaml: Path) -> dict[int, str]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = cfg.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def draw(img_path: Path, label_path: Path, color: tuple[int, int, int]) -> np.ndarray:
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(img_path)
    h, w = image.shape[:2]
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/yolo")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--tile", type=int, default=420)
    parser.add_argument("--out", default="results/labels_preview.jpg")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = Path(args.data)
    images = sorted((root / "images" / args.split).glob("*.jpg"))
    rng = random.Random(args.seed)

    positive, background = [], []
    for p in images:
        lbl = root / "labels" / args.split / f"{p.stem}.txt"
        has = lbl.exists() and lbl.stat().st_size > 0
        (positive if has else background).append(p)

    print(
        f"{args.split}: {len(images)} images -> {len(positive)} smoke, {len(background)} background"
    )
    picks = rng.sample(positive, min(len(positive), args.count - 2))
    picks += rng.sample(background, min(len(background), 2))

    tiles = []
    for p in picks:
        lbl = root / "labels" / args.split / f"{p.stem}.txt"
        color = (0, 140, 255) if (lbl.exists() and lbl.stat().st_size) else (120, 120, 120)
        img = draw(p, lbl, color)
        img = cv2.resize(img, (args.tile, int(args.tile * img.shape[0] / img.shape[1])))
        img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        tiles.append(img)

    rows = []
    for i in range(0, len(tiles), args.cols):
        row = tiles[i : i + args.cols]
        height = max(t.shape[0] for t in row)
        row = [
            cv2.copyMakeBorder(
                t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            for t in row
        ]
        rows.append(np.hstack(row))
    width = max(r.shape[1] for r in rows)
    grid = np.vstack(
        [
            cv2.copyMakeBorder(
                r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            for r in rows
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print(f"wrote {out} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
