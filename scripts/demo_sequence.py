"""Render a detection timeline over one FIgLib fire sequence.

Smoke detection cannot be judged frame by frame: the interesting question is
whether the system stays quiet during the pre-ignition frames and then raises
the alarm shortly after the plume appears. This script walks a sequence in
chronological order (the filenames carry the offset from plume appearance),
runs the detector, feeds the hits through the temporal decision rule, and
writes a single timeline image.

Usage:
    python scripts/demo_sequence.py \
        --weights runs/detect/figlib_smoke_v11n/weights/best.pt \
        --seq 20160718_FIRE_mw-e-mobo-c --frames 10
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsd.alert.decision import TemporalDecision  # noqa: E402

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)


def frame_offset(name: str) -> int:
    match = OFFSET_RE.search(name)
    return int(match.group(1)) if match else 0


def tile(
    image: np.ndarray,
    offset: int,
    boxes: list[tuple[int, int, int, int]],
    alarming: bool,
    width: int = 460,
) -> np.ndarray:
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 100, 255), 3)
    scale = width / image.shape[1]
    image = cv2.resize(image, (width, int(image.shape[0] * scale)))

    bar_h = 72
    bar = np.full((bar_h, width, 3), 255, dtype=np.uint8)
    sign = f"T{offset:+.0f}s"
    label = (
        "plume appears" if offset == 0 else ("before ignition" if offset < 0 else "after ignition")
    )
    cv2.putText(bar, sign, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(bar, label, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 90), 1, cv2.LINE_AA)

    if alarming:
        cv2.rectangle(bar, (width - 96, 8), (width - 8, 62), (0, 0, 220), -1)
        cv2.putText(
            bar,
            "ALARM",
            (width - 90, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.rectangle(bar, (width - 96, 8), (width - 8, 62), (225, 225, 225), -1)
        cv2.putText(
            bar,
            "quiet",
            (width - 82, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )

    return np.vstack([bar, image])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seq", default="20160718_FIRE_mw-e-mobo-c")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--out", default="results/demo")
    args = parser.parse_args()

    from ultralytics import YOLO

    seq_dir = Path(args.raw) / args.seq
    frames = sorted(seq_dir.glob("*.jpg"), key=lambda p: frame_offset(p.name))
    if not frames:
        raise SystemExit(f"no frames in {seq_dir}")

    picks = [
        frames[min(len(frames) - 1, round(i * (len(frames) - 1) / max(1, args.frames - 1)))]
        for i in range(args.frames)
    ]
    picks = list(dict.fromkeys(picks))

    model = YOLO(args.weights)
    decision = TemporalDecision(window=args.window, min_hits=args.min_hits)

    tiles, rows_log = [], []
    for path in picks:
        offset = frame_offset(path.name)
        result = model.predict(
            str(path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        boxes = [tuple(int(v) for v in b) for b in result.boxes.xyxy.cpu().numpy()]
        alarming = decision.update(len(boxes) > 0)
        rows_log.append((offset, len(boxes), alarming))
        tiles.append(tile(cv2.imread(str(path)), offset, boxes, alarming))

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
    for i in range(0, len(padded), args.cols):
        row = padded[i : i + args.cols]
        while len(row) < args.cols:
            row.append(np.full_like(padded[0], 255))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"timeline_{args.seq}.jpg"
    cv2.imwrite(str(target), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(f"sequence: {args.seq}  ({len(frames)} frames total)")
    print(f"rule: alarm when >={args.min_hits}/{args.window} recent frames contain smoke")
    for offset, n, alarming in rows_log:
        print(f"  T{offset:+6d}s  boxes={n}  alarm={'YES' if alarming else 'no'}")
    first = next((o for o, n, a in rows_log if a), None)
    print(f"first alarm at T{first:+d}s" if first is not None else "never alarmed")
    print(f"wrote {target} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
