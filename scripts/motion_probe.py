"""Probe: is frame-differencing motion energy separable between pre-fire
background frames and early-fire frames on FIgLib 60s-interval sequences?

The single-frame detector outputs zero confidence for the first ~10 minutes
of both reference sequences, so a fused motion branch would have nothing to
fuse with. This probe checks whether plain abs-diff energy (and a strong
local-block variant) separates early-fire frames from pre-fire background.
If it does, the temporal branch is worth building; if not, that is a result
too - the branch needs real-rate video, not 60s stills.

Usage:
    python scripts/motion_probe.py --raw data/raw --seqs 20160718_FIRE_mw-e-mobo-c
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


def motion_energy(prev: np.ndarray, curr: np.ndarray, block: int = 32) -> tuple[float, float]:
    """Mean abs-diff over the frame, and the 99th-percentile block energy."""
    a = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY).astype(np.int16)
    diff = np.abs(a - b)
    h, w = diff.shape
    hb, wb = h // block, w // block
    blocks = diff[: hb * block, : wb * block].reshape(hb, block, wb, block).mean(axis=(1, 3))
    return float(diff.mean()), float(np.percentile(blocks, 99))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument(
        "--seqs", nargs="+", default=["20160718_FIRE_mw-e-mobo-c", "20160718_FIRE_mg-s-iqeye"]
    )
    parser.add_argument("--resize", type=int, default=960)
    args = parser.parse_args()

    for seq in args.seqs:
        frames = sorted((Path(args.raw) / seq).glob("*.jpg"), key=lambda p: frame_offset(p.name))
        if len(frames) < 2:
            print(f"skip {seq}")
            continue

        print(f"\n=== {seq} ===")
        print(f"{'offset':>8} | {'mean diff':>9} | {'block p99':>9} | phase")
        print("-" * 52)
        pre: list[tuple[float, float]] = []
        early: list[tuple[float, float]] = []
        prev_img = None
        for path in frames:
            img = cv2.imread(str(path))
            if img is None:
                continue
            scale = args.resize / img.shape[1]
            img = cv2.resize(img, (args.resize, int(img.shape[0] * scale)))
            offset = frame_offset(path.name)
            if prev_img is not None:
                mean_d, block_d = motion_energy(prev_img, img)
                # Attribute the diff to the newer frame of the pair.
                phase = (
                    "pre-fire" if offset < 0 else ("early (0-600s)" if offset <= 600 else "later")
                )
                row = (mean_d, block_d)
                if offset < 0:
                    pre.append(row)
                elif offset <= 600:
                    early.append(row)
                print(f"T{offset:+7d}s | {mean_d:9.3f} | {block_d:9.3f} | {phase}")
            prev_img = img

        if pre and early:
            pre_m = np.array([x[0] for x in pre])
            early_m = np.array([x[0] for x in early])
            pre_b = np.array([x[1] for x in pre])
            early_b = np.array([x[1] for x in early])
            print(f"\npre-fire  mean diff: median={np.median(pre_m):.3f}  max={pre_m.max():.3f}")
            print(f"early     mean diff: median={np.median(early_m):.3f}  min={early_m.min():.3f}")
            print(f"block p99 pre-fire: median={np.median(pre_b):.3f}  max={pre_b.max():.3f}")
            print(f"block p99 early   : median={np.median(early_b):.3f}  min={early_b.min():.3f}")
            separable_mean = early_m.min() > pre_m.max()
            separable_block = early_b.min() > pre_b.max()
            print(
                f"mean-diff fully separable: {separable_mean}; block p99 fully separable: {separable_block}"
            )


if __name__ == "__main__":
    main()
