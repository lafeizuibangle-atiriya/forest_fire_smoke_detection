"""Render a detection demo GIF over one FIgLib sequence.

Frames are sampled evenly across the whole sequence (pre-ignition -> plume),
each is run through the detector, and the result is written as an animated
GIF for the README. The caption bar shows detection count and alarm state.

Usage:
    python scripts/make_gif.py --weights models/best.pt --seq 20160718_FIRE_mw-e-mobo-c
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsd.alert.decision import TemporalDecision  # noqa: E402

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)


def frame_offset(name: str) -> int:
    match = OFFSET_RE.search(name)
    return int(match.group(1)) if match else 0


def caption(
    width: int,
    offset: int,
    n_boxes: int,
    alarming: bool,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> Image.Image:
    bar = Image.new("RGB", (width, 56), (255, 255, 255))
    draw = ImageDraw.Draw(bar)
    sign = f"T{offset:+d}s"
    state = "SMOKE DETECTED" if n_boxes else "no detection"
    draw.text((12, 16), sign, fill=(25, 25, 25), font=font)
    draw.text((150, 18), state, fill=(200, 60, 0) if n_boxes else (120, 120, 120), font=font)
    if alarming:
        draw.rounded_rectangle((width - 150, 8, width - 12, 48), radius=8, fill=(220, 30, 30))
        draw.text((width - 128, 16), "ALARM", fill=(255, 255, 255), font=font)
    return bar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seq", default="20160718_FIRE_mw-e-mobo-c")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--frames", type=int, default=14)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--out", default="docs/images/demo.gif")
    args = parser.parse_args()

    from ultralytics import YOLO

    seq_dir = Path(args.raw) / args.seq
    frames = sorted(seq_dir.glob("*.jpg"), key=lambda p: frame_offset(p.name))
    picks = [
        frames[min(len(frames) - 1, round(i * (len(frames) - 1) / max(1, args.frames - 1)))]
        for i in range(args.frames)
    ]
    picks = list(dict.fromkeys(picks))

    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except OSError:
        font = ImageFont.load_default()

    model = YOLO(args.weights)
    decision = TemporalDecision(window=3, min_hits=2)

    pil_frames = []
    for path in picks:
        offset = frame_offset(path.name)
        result = model.predict(
            str(path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        n_boxes = len(result.boxes)
        alarming = decision.update(n_boxes > 0)

        plotted = result.plot(labels=False, conf=False, line_width=3)  # BGR ndarray
        rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        scale = args.width / img.width
        img = img.resize((args.width, int(img.height * scale)), Image.LANCZOS)
        frame = Image.new("RGB", (args.width, img.height + 56), (255, 255, 255))
        frame.paste(caption(args.width, offset, n_boxes, alarming, font), (0, 0))
        frame.paste(img, (0, 56))
        pil_frames.append(frame)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil_frames[0].save(
        out,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
    )
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({len(pil_frames)} frames, {size_kb:.0f} KB, {args.fps} fps)")


if __name__ == "__main__":
    main()
