"""Evaluate the background-evidence branch against the detector baseline.

For each FIgLib sequence: run (a) the detector + sliding-window vote at
conf 0.25 and (b) the BackgroundMonitor, and report first-alarm offsets and
pre-ignition false alarms for both. This is the acceptance test for the
temporal branch: earlier first alarm, no new pre-fire false alarms.

Usage:
    python scripts/temporal_eval.py --weights models/best_pseudo.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsd.alert.background import BackgroundConfig, BackgroundMonitor  # noqa: E402
from ffsd.alert.decision import TemporalDecision  # noqa: E402

OFFSET_RE = re.compile(r"_([+-]\d+)\.jpg$", re.IGNORECASE)
CONF = 0.25


def frame_offset(name: str) -> int:
    match = OFFSET_RE.search(name)
    return int(match.group(1)) if match else 0


def fmt(offset: int | None) -> str:
    return f"T{offset:+d}s" if offset is not None else "never"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/best_pseudo.pt")
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument(
        "--seqs", nargs="*", default=None, help="default: all sequences under --raw"
    )
    args = parser.parse_args()

    raw = Path(args.raw)
    seqs = args.seqs or sorted(d.name for d in raw.iterdir() if d.is_dir())

    from ultralytics import YOLO

    model = YOLO(args.weights)

    print(f"detector baseline: conf={CONF}, vote {args.min_hits}/{args.window}")
    print(
        f"{'sequence':34} | {'det alarm':>10} | {'det preFA':>9} | {'bg alarm':>10} | {'bg preFA':>8}"
    )
    print("-" * 84)

    totals = {"det_pre": 0, "bg_pre": 0, "det_earlier": 0, "bg_earlier": 0}
    for seq in seqs:
        frames = sorted((raw / seq).glob("*.jpg"), key=lambda p: frame_offset(p.name))
        if not frames:
            continue

        detector = TemporalDecision(window=args.window, min_hits=args.min_hits)
        monitor = BackgroundMonitor(BackgroundConfig())
        det_alarm = det_pre = bg_alarm = bg_pre = None
        det_pre_n = bg_pre_n = 0

        for path in frames:
            offset = frame_offset(path.name)
            img = cv2.imread(str(path))

            result = model.predict(
                str(path), imgsz=args.imgsz, conf=CONF, device=args.device, verbose=False
            )[0]
            if detector.update(len(result.boxes) > 0):
                if det_alarm is None:
                    det_alarm = offset
                if offset < 0:
                    det_pre_n += 1

            evidence = monitor.update(img)
            if evidence.alarming:
                if bg_alarm is None:
                    bg_alarm = offset
                if offset < 0:
                    bg_pre_n += 1

        det_pre, bg_pre = det_pre_n, bg_pre_n
        totals["det_pre"] += det_pre
        totals["bg_pre"] += bg_pre
        if bg_alarm is not None and (det_alarm is None or bg_alarm < det_alarm):
            totals["bg_earlier"] += 1
        elif det_alarm is not None and (bg_alarm is None or det_alarm <= bg_alarm):
            totals["det_earlier"] += 1

        print(
            f"{seq:34} | {fmt(det_alarm):>10} | {det_pre:>9d} | {fmt(bg_alarm):>10} | {bg_pre:>8d}"
        )

    print("-" * 84)
    print(f"pre-fire false alarms: detector={totals['det_pre']}  background={totals['bg_pre']}")
    print(
        f"sequences: background earlier on {totals['bg_earlier']}, detector earlier/tied on {totals['det_earlier']}"
    )


if __name__ == "__main__":
    main()
