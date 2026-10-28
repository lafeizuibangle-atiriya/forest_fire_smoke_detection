"""Background-subtraction evidence for early smoke detection.

The single-frame detector outputs zero confidence during the first ~10
minutes of a developing plume, so no threshold on detector conf can pull
first detection earlier than ~T+12min. The plume, however, grows in place
against a stable background, so its accumulated deviation from a running
background model is measurable from the first frame it appears (validated
on 6/9 local FIgLib sequences at T+0/T+120; the other 3 have contaminated
pre-fire segments - see docs/训练评估报告.md).

This module keeps the same contract as TemporalDecision: pure logic plus
numpy/opencv, no camera or GPU dependency, unit-testable with synthetic
frames.

Rule: the block-energy e_t (99th percentile of 32x32 block mean abs-diff
against the running median background) fires a candidate when BOTH
  1. e_t > ratio x median(past energies)          (above the noise floor)
  2. e_t >= max(past energies)                    (making a new high)
and this holds for `consecutive` frames in a row. A developing plume keeps
making new highs because it grows in place; a passing cloud spikes once and
plateaus, so it breaks the streak.

Background absorption is selective: only frames that do NOT look anomalous
enter the median model, so smoke never becomes "the new normal".
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BackgroundConfig:
    """Tuning for the background-evidence branch."""

    resize_w: int = 960
    block: int = 32
    bg_frames: int = 20  # running median window (frames, not seconds)
    ratio: float = 1.3  # candidate when e_t > ratio * median(past e)
    consecutive: int = 3  # candidate frames in a row required
    history: int = 15  # how many past energies the baseline uses


@dataclass
class BackgroundEvidence:
    """Per-frame result of the background branch."""

    energy: float  # block p99 of abs-diff vs background
    baseline: float  # running median of past energies (nan until warm-up)
    candidate: bool  # this frame exceeds the baseline
    alarming: bool  # `consecutive` candidates in a row reached


class BackgroundMonitor:
    """Running-median background model with an adaptive energy rule."""

    def __init__(self, config: BackgroundConfig | None = None) -> None:
        self.cfg = config or BackgroundConfig()
        self._bg: deque[np.ndarray] = deque(maxlen=self.cfg.bg_frames)
        self._energies: deque[float] = deque(maxlen=self.cfg.history)
        self._streak = 0

    def update(self, frame_bgr: np.ndarray) -> BackgroundEvidence:
        gray = self._to_gray(frame_bgr)
        if self._bg:
            bg = np.median(np.stack(self._bg), axis=0)
            energy = self._block_p99(np.abs(gray - bg))
        else:
            energy = 0.0

        if len(self._energies) >= 3:
            baseline = float(np.median(self._energies))
            new_high = energy >= max(self._energies)
            candidate = bool(energy > self.cfg.ratio * baseline and new_high)
        else:
            baseline = float("nan")
            candidate = False
        self._streak = self._streak + 1 if candidate else 0
        alarming = self._streak >= self.cfg.consecutive

        # Only absorb frames that look like background; smoke must never
        # become part of the model, or its evidence fades away.
        if not candidate:
            self._bg.append(gray)
        self._energies.append(energy)
        return BackgroundEvidence(
            energy=energy, baseline=baseline, candidate=candidate, alarming=alarming
        )

    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        w = self.cfg.resize_w
        scale = w / frame_bgr.shape[1]
        resized = cv2.resize(frame_bgr, (w, int(frame_bgr.shape[0] * scale)))
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def _block_p99(self, diff: np.ndarray) -> float:
        b = self.cfg.block
        h, w = diff.shape
        blocks = diff[: h // b * b, : w // b * b].reshape(h // b, b, w // b, b).mean(axis=(1, 3))
        return float(np.percentile(blocks, 99))
