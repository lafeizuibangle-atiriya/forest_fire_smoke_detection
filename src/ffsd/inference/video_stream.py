"""Unified video source reader: file / RTSP / camera index."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


class VideoStream:
    """Reads frames from a video file, RTSP URL, or camera index."""

    def __init__(self, source: str | int) -> None:
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video source: {source!r}")

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        self._cap.release()
