"""Unit tests for temporal false-positive suppression."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsd.alert.decision import TemporalDecision  # noqa: E402


def test_not_alarming_before_min_hits() -> None:
    decision = TemporalDecision(window=10, min_hits=6)
    for _ in range(5):
        assert decision.update(True) is False


def test_alarming_at_min_hits() -> None:
    decision = TemporalDecision(window=10, min_hits=6)
    for _ in range(5):
        decision.update(True)
    assert decision.update(True) is True


def test_window_slides_out_old_hits() -> None:
    decision = TemporalDecision(window=10, min_hits=6)
    for _ in range(6):
        decision.update(True)
    assert decision.is_alarming() is True

    # Feed 5 misses: only 5 of the last 10 frames are hits -> no alarm.
    for _ in range(5):
        decision.update(False)
    assert decision.is_alarming() is False


def test_invalid_params() -> None:
    with pytest.raises(ValueError):
        TemporalDecision(window=0, min_hits=1)
    with pytest.raises(ValueError):
        TemporalDecision(window=10, min_hits=11)
