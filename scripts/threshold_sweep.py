"""Sweep the detector confidence threshold over full FIgLib sequences.

Measures, for each conf threshold, how the temporal metrics change:
first detection offset, first alarm offset (sliding-window vote), and
pre-ignition false hits/alarms. Predictions run once at a low conf and are
reused for every threshold, so the sweep costs a single inference pass.

Usage:
    python scripts/threshold_sweep.py --weights models/best_pseudo.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsd.alert.decision import TemporalDecision  # noqa: E402

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)
THRESHOLDS = (0.25, 0.20, 0.15, 0.12, 0.10)


def frame_offset(name: str) -> int:
    match = OFFSET_RE.search(name)
    return int(match.group(1)) if match else 0


def first_alarm_offset(
    offsets: list[int], hits: list[bool], window: int, min_hits: int
) -> int | None:
    decision = TemporalDecision(window=window, min_hits=min_hits)
    for offset, hit in zip(offsets, hits, strict=False):
        if decision.update(hit):
            return offset
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/best_pseudo.pt")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument(
        "--seqs", nargs="+", default=["20160718_FIRE_mw-e-mobo-c", "20160718_FIRE_mg-s-iqeye"]
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min-hits", type=int, default=2)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    for seq in args.seqs:
        seq_dir = Path(args.raw) / seq
        frames = sorted(seq_dir.glob("*.jpg"), key=lambda p: frame_offset(p.name))
        if not frames:
            print(f"skip {seq}: no frames")
            continue

        # One inference pass at the lowest conf; reuse for all thresholds.
        offsets: list[int] = []
        scores: list[list[float]] = []
        for path in frames:
            result = model.predict(
                str(path), imgsz=args.imgsz, conf=args.min_conf, device=args.device, verbose=False
            )[0]
            confs = result.boxes.conf.cpu().numpy().tolist()
            offsets.append(frame_offset(path.name))
            scores.append(confs)

        print(
            f"\n=== {seq}  ({len(frames)} frames, weights={args.weights}, imgsz={args.imgsz}) ==="
        )
        print(
            f"{'conf':>6} | {'first detect':>12} | {'first alarm':>11} | {'pre-fire hits':>13} | {'pre-fire alarms':>15}"
        )
        print("-" * 72)
        for thr in THRESHOLDS:
            hits = [any(c >= thr for c in confs) for confs in scores]
            first_hit = next((o for o, h in zip(offsets, hits, strict=False) if h), None)
            first_alarm = first_alarm_offset(offsets, hits, args.window, args.min_hits)
            pre_hits = sum(1 for o, h in zip(offsets, hits, strict=False) if o < 0 and h)
            pre_alarms = 0
            decision = TemporalDecision(window=args.window, min_hits=args.min_hits)
            for o, h in zip(offsets, hits, strict=False):
                if decision.update(h) and o < 0:
                    pre_alarms += 1

            def fmt(v: int | None) -> str:
                return f"T{v:+d}s" if v is not None else "never"

            print(
                f"{thr:>6.2f} | {fmt(first_hit):>12} | {fmt(first_alarm):>11} | {pre_hits:>13d} | {pre_alarms:>15d}"
            )


if __name__ == "__main__":
    main()
