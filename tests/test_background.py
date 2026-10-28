"""Tests for the background-evidence branch (src/ffsd/alert/background.py)."""

from __future__ import annotations

import numpy as np
import pytest

from ffsd.alert.background import BackgroundConfig, BackgroundMonitor


def frame_with_blob(
    size: tuple[int, int], center: tuple[int, int], radius: int, level: int = 40
) -> np.ndarray:
    """BGR frame: flat 128 background, optional bright disk (the 'plume')."""
    img = np.full((*size, 3), 128, dtype=np.uint8)
    if radius > 0:
        h, w = size
        y, x = np.ogrid[:h, :w]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        img[mask] = 128 + level
    return img


@pytest.fixture()
def monitor() -> BackgroundMonitor:
    return BackgroundMonitor(
        BackgroundConfig(resize_w=320, block=16, bg_frames=20, ratio=1.3, consecutive=3)
    )


def test_quiet_scene_never_alarms(monitor: BackgroundMonitor) -> None:
    rng = np.random.default_rng(0)
    for _ in range(40):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        frame += rng.integers(-1, 2, size=frame.shape, dtype=np.int16).astype(np.uint8)
        assert not monitor.update(frame).alarming


def test_growing_blob_alarms(monitor: BackgroundMonitor) -> None:
    rng = np.random.default_rng(1)
    # Warm up with 25 quiet frames so the baseline is established.
    for _ in range(25):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        frame += (
            rng.integers(-1, 2, size=frame.shape, dtype=np.int16).astype(np.int16).astype(np.uint8)
        )
        evidence = monitor.update(frame)
    assert not evidence.alarming

    # A plume appears and grows: alarm must fire within a few frames.
    alarmed_at = None
    for i, radius in enumerate(range(4, 40, 4)):
        frame = frame_with_blob((240, 320), (160, 60), radius)
        frame = frame.astype(np.int16) + rng.integers(-1, 2, size=frame.shape, dtype=np.int16)
        evidence = monitor.update(frame.astype(np.uint8))
        if evidence.alarming and alarmed_at is None:
            alarmed_at = i
    assert alarmed_at is not None and alarmed_at <= 3


def test_transient_blob_does_not_alarm(monitor: BackgroundMonitor) -> None:
    rng = np.random.default_rng(2)
    for _ in range(25):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        frame += rng.integers(-1, 2, size=frame.shape, dtype=np.int16).astype(np.uint8)
        monitor.update(frame)
    # One bright frame (a flash), then quiet again: no alarm may fire.
    monitor.update(frame_with_blob((240, 320), (160, 60), 20, level=60))
    for _ in range(10):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        frame += rng.integers(-1, 2, size=frame.shape, dtype=np.int16).astype(np.uint8)
        assert not monitor.update(frame).alarming
