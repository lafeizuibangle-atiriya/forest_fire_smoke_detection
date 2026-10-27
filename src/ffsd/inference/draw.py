"""Drawing helpers for visualization."""

from __future__ import annotations

import cv2
import numpy as np

from ..models.base import Detection

_COLORS = {
    "smoke": (0, 165, 255),  # orange (BGR)
    "fire": (0, 0, 255),  # red
}


def draw_detections(
    frame: np.ndarray, detections: list[Detection], alarming: bool = False
) -> np.ndarray:
    """Draw boxes and an optional alarm banner on a copy of the frame."""
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        color = _COLORS.get(det.cls_name, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls_name} {det.confidence:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    if alarming:
        cv2.putText(
            vis,
            "ALARM",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return vis
