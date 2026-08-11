"""Tests for the decay A/B's synthetic profile generator.

``_profile_for`` (tests/eval/decay_ab.py) maps an eval seed index to a
deterministic aging profile.  It must be deterministic (the A/B needs
reproducible arms) and must spread hot / medium / cold bands so decay
factors actually discriminate — otherwise the decay A/B is vacuous (every
factor ≈ 1.0 on a fresh corpus).
"""

from __future__ import annotations

import pytest

from tests.eval.decay_ab import _profile_for


class TestDecayProfile:
    def test_deterministic(self) -> None:
        assert _profile_for(7, 30) == _profile_for(7, 30)

    def test_hot_band_recent_and_well_recalled(self) -> None:
        p = _profile_for(0, 30)  # index 0 → hot band (1..10h, recall 10..15)
        assert 1 <= p["created_hours_ago"] <= 10
        assert p["recall_count"] >= 10
        assert p["recalled_hours_ago"] is not None

    def test_cold_band_old_and_sparse_recall(self) -> None:
        p = _profile_for(29, 30)  # index 29 → cold band (48..120h, recall 0..2)
        assert 48 <= p["created_hours_ago"] <= 120
        assert p["recall_count"] <= 2

    def test_never_recalled_cold_memory_has_no_recalled_at(self) -> None:
        """A cold memory with recall_count 0 is never-recalled (recalled_at
        NULL) — the time base falls back to created_at in the live decay."""
        # Find a cold-band index with recall_count 0 (index % 3 == 0).
        p = _profile_for(21, 30)
        assert p["recall_count"] == 0
        assert p["recalled_hours_ago"] is None

    def test_bands_cover_all_indices(self) -> None:
        for i in range(30):
            p = _profile_for(i, 30)
            assert p["created_hours_ago"] > 0
            assert 0 <= p["recall_count"] <= 15
