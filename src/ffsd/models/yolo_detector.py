"""Ultralytics YOLO backend."""

from __future__ import annotations

import numpy as np

from ..core.logging_utils import get_logger
from .base import Detection, Detector

log = get_logger("detector")


class YOLODetector(Detector):
    """YOLO detector backed by ultralytics.

    The heavy import happens inside __init__ on purpose: importing this module
    must not load torch/ultralytics (slow, allocates memory).
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.4,
        iou: float = 0.3,
        device: str | None = None,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._conf = conf
        self._iou = iou
        self._device = device or None
        log.info("loaded %s on device=%s", weights, self._device or "auto")

    def predict(self, frame: np.ndarray) -> list[Detection]:
        if self._device:
            result = self._model(
                frame, conf=self._conf, iou=self._iou, verbose=False, device=self._device
            )[0]
        else:
            result = self._model(frame, conf=self._conf, iou=self._iou, verbose=False)[0]

        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            cls_id = int(box.cls)
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    cls_name=str(names[cls_id]),
                    confidence=float(box.conf),
                    xyxy=xyxy,  # type: ignore[arg-type]
                )
            )
        return detections
