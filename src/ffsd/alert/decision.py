"""Temporal false-positive suppression.

This is pure logic with no camera/GPU dependency, so it is unit-testable.
Rule: alarm only if at least `min_hits` of the last `window` frames are hits.
A drifting smoke plume keeps hitting; a transient cloud/reflection does not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class TemporalDecision:
    """Sliding-window vote over per-frame hit/miss."""

    window: int = 10
    min_hits: int = 6
    _buf: deque[bool] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError("window must be positive")
        if not 0 < self.min_hits <= self.window:
            raise ValueError("min_hits must be in (0, window]")
        self._buf = deque(maxlen=self.window)

    def update(self, hit: bool) -> bool:
        """Push the latest frame result and return whether to alarm."""
        self._buf.append(bool(hit))
        return self.is_alarming()

    def is_alarming(self) -> bool:
        return sum(self._buf) >= self.min_hits

    def reset(self) -> None:
        self._buf.clear()
