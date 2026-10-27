"""Sliced (tiled) inference for small smoke plumes -- a SAHI-style workaround.

The temporal EDA shows the earliest annotated plume is only ~900 px^2 (~30 px
wide, 2% of frame width). At imgsz 960 that is ~14 px -- below detection
limits, and raising imgsz further hits VRAM limits for a marginal gain.

Slicing attacks the problem from the other side: cut the 2048x1536 frame into
overlapping 960px tiles, run the detector per tile, then merge + NMS. A 30px
plume inside a 960px tile is 3.1% of the tile -- the same fraction a 64px
plume has in the full frame, i.e. every tile sees a "zoomed-in" view.

No extra dependency: tiling + NMS are implemented with torch/torchvision.

Usage:
    python scripts/sliced_infer.py --weights models/best.pt --eval          # metrics on val
    python scripts/sliced_infer.py --weights models/best.pt --image <path>  # side-by-side viz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms as tv_nms


def tile_ranges(size: int, slice_size: int, stride: int) -> list[int]:
    """Start positions of tiles along one axis, last tile right-aligned."""
    if size <= slice_size:
        return [0]
    ranges = list(range(0, size - slice_size + 1, stride))
    if ranges[-1] + slice_size < size:
        ranges.append(size - slice_size)
    return ranges


def sliced_predict(
    model,
    image: np.ndarray,
    slice_size: int = 960,
    overlap: float = 0.25,
    conf: float = 0.25,
    device: str = "0",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run tiled inference and return (xyxy, conf, elapsed_ms)."""
    h, w = image.shape[:2]
    stride = int(slice_size * (1 - overlap))
    ys = tile_ranges(h, slice_size, stride)
    xs = tile_ranges(w, slice_size, stride)

    boxes_all, conf_all = [], []
    t0 = time.perf_counter()
    for y0 in ys:
        for x0 in xs:
            tile = image[y0 : y0 + slice_size, x0 : x0 + slice_size]
            result = model.predict(tile, imgsz=slice_size, conf=conf, device=device, verbose=False)[
                0
            ]
            if len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy().astype(float)
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            boxes_all.append(xyxy)
            conf_all.append(result.boxes.conf.cpu().numpy())
    elapsed = (time.perf_counter() - t0) * 1000

    if not boxes_all:
        return np.zeros((0, 4)), np.zeros((0,)), elapsed
    xyxy = np.concatenate(boxes_all)
    conf = np.concatenate(conf_all)
    keep = tv_nms(
        torch.tensor(xyxy, dtype=torch.float32),
        torch.tensor(conf, dtype=torch.float32),
        iou_threshold=0.5,
    ).numpy()
    return xyxy[keep], conf[keep], elapsed


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (n,4) and (m,4) xyxy arrays."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def match_metrics(
    gt_boxes: list[np.ndarray], pred_boxes: list[np.ndarray], iou_thr: float = 0.5
) -> dict:
    tp = fp = fn = 0
    for gt, pred in zip(gt_boxes, pred_boxes, strict=False):
        if len(gt) == 0 and len(pred) == 0:
            continue
        if len(gt) == 0:
            fp += len(pred)
            continue
        if len(pred) == 0:
            fn += len(gt)
            continue
        m = iou_matrix(gt, pred)
        matched = set()
        for gi in range(len(gt)):
            best = max(
                range(len(pred)), key=lambda pi: m[gi, pi] if pi not in matched else -1, default=-1
            )
            if best >= 0 and m[gi, best] >= iou_thr:
                matched.add(best)
                tp += 1
            else:
                fn += 1
        fp += len(pred) - len(matched)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/yolo")
    parser.add_argument("--slice", type=int, default=960)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image", help="single image for side-by-side viz")
    parser.add_argument("--eval", action="store_true", help="full val-set comparison")
    parser.add_argument("--out", default="results/sliced")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.image:
        img = cv2.imread(args.image)
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        full = model.predict(
            args.image, imgsz=args.slice, conf=args.conf, device=args.device, verbose=False
        )[0]
        t_full = (time.perf_counter() - t0) * 1000
        sl_boxes, sl_conf, t_sliced = sliced_predict(
            model, img, args.slice, args.overlap, args.conf, args.device
        )

        def panel(boxes, title, color, ms):
            canvas = img.copy()
            for b in boxes:
                cv2.rectangle(canvas, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 4)
            cv2.putText(
                canvas,
                f"{title}  {len(boxes)} boxes  {ms:.0f} ms",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            return canvas

        left = panel(full.boxes.xyxy.cpu().numpy(), "FULL-FRAME", (0, 160, 0), t_full)
        right = panel(sl_boxes, f"SLICED {args.slice}px o={args.overlap}", (0, 100, 255), t_sliced)
        left = cv2.resize(left, (w // 2, h // 2))
        right = cv2.resize(right, (w // 2, h // 2))
        sheet = np.hstack([left, right])
        target = out / f"sliced_{Path(args.image).stem}.jpg"
        cv2.imwrite(str(target), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(
            f"wrote {target}  full={len(full.boxes)} boxes / {t_full:.0f}ms   sliced={len(sl_boxes)} / {t_sliced:.0f}ms"
        )
        return

    if args.eval:
        root = Path(args.data)
        images = sorted((root / "images" / "val").glob("*.jpg"))
        gt_all, full_all, sliced_all = [], [], []
        t_full_sum = t_sliced_sum = 0.0

        def read_gt(p: Path) -> np.ndarray:
            h, w = cv2.imread(str(p)).shape[:2]
            boxes = []
            lbl = root / "labels" / "val" / f"{p.stem}.txt"
            if lbl.exists():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    boxes.append(
                        [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                    )
            return np.array(boxes) if boxes else np.zeros((0, 4))

        for _i, p in enumerate(images):
            img = cv2.imread(str(p))
            gt_all.append(read_gt(p))

            t0 = time.perf_counter()
            r = model.predict(
                str(p), imgsz=args.slice, conf=args.conf, device=args.device, verbose=False
            )[0]
            t_full_sum += time.perf_counter() - t0
            full_all.append(r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4)))

            sb, _, ts = sliced_predict(model, img, args.slice, args.overlap, args.conf, args.device)
            t_sliced_sum += ts
            sliced_all.append(sb)

        m_full = match_metrics(gt_all, full_all)
        m_slice = match_metrics(gt_all, sliced_all)
        n = len(images)
        report = {
            "images": n,
            "slice": args.slice,
            "overlap": args.overlap,
            "conf": args.conf,
            "full_frame": {**m_full, "ms_per_image": round(t_full_sum / n * 1000, 1)},
            "sliced": {**m_slice, "ms_per_image": round(t_sliced_sum / n * 1000, 1)},
        }
        (out / "sliced_eval.json").write_text(
            __import__("json").dumps(report, indent=2), encoding="utf-8"
        )
        print(__import__("json").dumps(report, indent=2))


if __name__ == "__main__":
    main()
