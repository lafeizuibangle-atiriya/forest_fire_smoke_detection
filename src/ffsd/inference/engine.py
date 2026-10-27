"""Inference engine: thin wrapper between detectors and callers."""

from __future__ import annotations

import numpy as np

from ..models.base import Detection, Detector


class InferenceEngine:
    """Runs a detector on frames. Keeps callers independent of the backend."""

    def __init__(self, detector: Detector) -> None:
        self._detector = detector

    def run(self, frame: np.ndarray) -> list[Detection]:
        return self._detector.predict(frame)
