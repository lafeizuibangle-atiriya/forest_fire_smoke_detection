"""Probe: per-block robust z-scores against a pre-fire background model.

Global illumination drift raises every block together, so a global energy
threshold cannot separate early smoke from drift. This probe normalizes
each pixel by its own pre-fire median absolute deviation, then looks at
the max block-mean z-score per frame. Smoke is a localized anomaly and
should stand out; uniform drift should not.

Oracle setup: the background model uses ALL pre-fire frames (upper bound).
Usage:
    python scripts/zscore_probe.py --raw data/raw
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)


def frame_offset(name: str) -> int:
    match = OFFSET_RE.search(name)
    return int(match.group(1)) if match else 0


class ZScorer:
    def __init__(self, resize_w: int = 960, block: int = 32) -> None:
        self.resize_w = resize_w
        self.block = block

    def gray(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        scale = self.resize_w / img.shape[1]
        resized = cv2.resize(img, (self.resize_w, int(img.shape[0] * scale)))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def fit(self, pre_grays: list[np.ndarray]) -> None:
        stack = np.stack(pre_grays)
        self.bg = np.median(stack, axis=0)
        # Noise floor: static sky pixels have MAD~0, which would blow up the
        # z-scores. Sensor + JPEG noise is ~2 gray levels; use it as floor.
        self.mad = np.maximum(np.median(np.abs(stack - self.bg), axis=0), 2.0)

    def zmax(self, gray: np.ndarray) -> float:
        z = np.abs(gray - self.bg) / self.mad
        b = self.block
        h, w = z.shape
        blocks = z[: h // b * b, : w // b * b].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
        return float(blocks.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    args = parser.parse_args()

    print(f"{'sequence':34} | {'pre_z_max':>9} | {'T+0':>6} | {'T+120':>6} | {'T+240':>6} | sep")
    print("-" * 82)
    counts = {"YES": 0, "@120": 0, "@240": 0, "NO": 0}
    for d in sorted(Path(args.raw).iterdir()):
        if not d.is_dir():
            continue
        frames = sorted(d.glob("*.jpg"), key=lambda p: frame_offset(p.name))
        grays = [(frame_offset(p.name), p) for p in frames]
        pre = [o for o, _ in grays if o < 0]
        if len(pre) < 10:
            continue
        scorer = ZScorer()
        scorer.fit([scorer.gray(p) for o, p in grays if o < 0])
        pre_z = [scorer.zmax(scorer.gray(p)) for o, p in grays if o < 0]
        at = {o: scorer.zmax(scorer.gray(p)) for o, p in grays if o >= 0}
        pre_max = max(pre_z)
        t0, t120, t240 = at.get(0, 0.0), at.get(120, 0.0), at.get(240, 0.0)
        if t0 > pre_max:
            sep = "YES"
        elif t120 > pre_max:
            sep = "@120"
        elif t240 > pre_max:
            sep = "@240"
        else:
            sep = "NO"
        counts[sep] += 1
        print(f"{d.name:34} | {pre_max:9.2f} | {t0:6.2f} | {t120:6.2f} | {t240:6.2f} | {sep}")
    print("-" * 82)
    print(f"summary: {counts}")


if __name__ == "__main__":
    main()
