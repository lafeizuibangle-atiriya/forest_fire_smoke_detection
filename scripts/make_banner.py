"""Render the README banner (title card with key metrics)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 420
BG_TOP, BG_BOT = (16, 24, 43), (30, 41, 72)
ACCENT = (255, 122, 41)


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.truetype("arial.ttf", size)


def main() -> None:
    img = Image.new("RGB", (W, H))
    draw = ImageDraw.Draw(img)
    # vertical gradient
    for y in range(H):
        t = y / H
        color = tuple(int(BG_TOP[i] + (BG_BOT[i] - BG_TOP[i]) * t) for i in range(3))
        draw.line([(0, y), (W, y)], fill=color)
    # accent bars + smoke dots decoration
    draw.rectangle([0, 0, 14, H], fill=ACCENT)
    import random

    rng = random.Random(7)
    for _ in range(140):
        x, y = rng.randint(80, W), rng.randint(0, H)
        r = rng.randint(2, 7)
        shade = rng.randint(55, 90)
        draw.ellipse((x, y, x + r * 2, y + r * 2), fill=(shade, shade + 10, shade + 25))

    draw.text((64, 96), "Wildfire Smoke Detection", fill=(255, 255, 255), font=font(64))
    draw.text(
        (66, 186), "YOLO11  ·  FIgLib  ·  real-time alarm", fill=(160, 174, 200), font=font(30)
    )

    metrics = [
        ("mAP@0.5", "0.665"),
        ("Precision", "0.756"),
        ("Recall", "0.532"),
        ("Inference", "5.2 ms"),
    ]
    x = 64
    for label, value in metrics:
        draw.text((x, 268), value, fill=ACCENT, font=font(58))
        draw.text((x + 6, 348), label, fill=(150, 160, 185), font=font(24, bold=False))
        x += 300

    out = Path("docs/images/banner.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
