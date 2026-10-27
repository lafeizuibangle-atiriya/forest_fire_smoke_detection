"""Detector abstraction.

Callers depend on this interface, not on ultralytics directly, so swapping the
backend (RT-DETR, a self-trained model, ...) touches only one file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    """A single detection result."""

    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class Detector(ABC):
    """Abstract detector interface."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR image and return detections."""
        raise NotImplementedError
