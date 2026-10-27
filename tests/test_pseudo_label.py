"""Tests for the FIgLib filename parser and the pseudo-label helpers.

These guard the two places where a silent bug costs a whole training run:
parsing the time offset out of a filename, and bucketing boxes by size.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_figlib import frame_offset  # noqa: E402
from pseudo_label import _box_stats, evenly_pick  # noqa: E402


def test_frame_offset_keeps_the_sign() -> None:
    """Pre-ignition frames are negative, post-ignition frames positive."""
    assert frame_offset("1465063200_-02400.jpg") == -2400
    assert frame_offset("1465067760_+02160.jpg") == 2160
    assert frame_offset("1465067000_+00000.jpg") == 0


def test_frame_offset_returns_none_when_absent() -> None:
    assert frame_offset("fire_smoke_frame.jpg") is None
    assert frame_offset("1465063200.jpg") is None


def test_evenly_pick_covers_the_whole_range() -> None:
    """Even spacing is what keeps the pseudo set from clustering in one minute."""
    items = [f"{i}.jpg" for i in range(100)]
    picked = evenly_pick(items, 10)
    assert len(picked) == 10
    assert picked[0] == "0.jpg"
    assert int(picked[-1].split(".")[0]) >= 90


def test_evenly_pick_degrades_gracefully() -> None:
    assert evenly_pick(["a.jpg"], 10) == ["a.jpg"]
    assert evenly_pick([], 10) == []
    assert evenly_pick(["a.jpg", "b.jpg"], 0) == []


def test_box_stats_splits_coco_buckets() -> None:
    # 10x10 = small, 50x50 = medium, 200x200 = large (area thresholds 32², 96²)
    pairs = [(0.01, 0.01, 10, 10), (0.05, 0.05, 50, 50), (0.2, 0.2, 200, 200)]
    stats = _box_stats(pairs)
    assert stats["n"] == 3
    assert stats["small_pct"] == pytest.approx(33.3, abs=0.1)
    assert stats["medium_pct"] == pytest.approx(33.3, abs=0.1)
    assert stats["large_pct"] == pytest.approx(33.3, abs=0.1)
    assert stats["rel_w_median"] == pytest.approx(0.05, abs=1e-6)


def test_box_stats_handles_empty_input() -> None:
    assert _box_stats([])["n"] == 0
