"""Run inference on a video / image source and save annotated frames.

Usage:
    python scripts/infer.py --config configs/infer.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from ffsd.alert.decision import TemporalDecision  # noqa: E402
from ffsd.core.config import load_infer_config  # noqa: E402
from ffsd.core.logging_utils import get_logger  # noqa: E402
from ffsd.inference.draw import draw_detections  # noqa: E402
from ffsd.inference.engine import InferenceEngine  # noqa: E402
from ffsd.inference.video_stream import VideoStream  # noqa: E402
from ffsd.models.yolo_detector import YOLODetector  # noqa: E402

log = get_logger("infer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference on a video/image source")
    parser.add_argument("--config", default="configs/infer.yaml")
    args = parser.parse_args()

    cfg = load_infer_config(args.config)

    detector = YOLODetector(
        cfg.model.weights, cfg.model.conf, cfg.model.iou, device=cfg.model.device
    )
    engine = InferenceEngine(detector)
    decision = TemporalDecision(cfg.decision.window, cfg.decision.min_hits)
    stream = VideoStream(cfg.source.path)

    save_dir = Path(cfg.output.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        for idx, frame in enumerate(stream.frames()):
            detections = engine.run(frame)
            alarming = decision.update(len(detections) > 0)
            vis = draw_detections(frame, detections, alarming=alarming)
            cv2.imwrite(str(save_dir / f"frame_{idx:05d}.jpg"), vis)
            if alarming:
                log.warning("ALARM frame=%d detections=%d", idx, len(detections))
    finally:
        stream.release()

    log.info("done. outputs saved to %s", save_dir)


if __name__ == "__main__":
    main()
