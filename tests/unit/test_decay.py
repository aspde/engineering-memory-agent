"""Tests for Ebbinghaus decay functions — pure math, no mocking needed."""

import math
from datetime import datetime, timedelta, timezone

from backend.service.decay import compute_decay_factor


class TestComputeDecay:
    def test_fresh_memory_returns_one(self) -> None:
        assert compute_decay_factor(None, 0) == 1.0

    def test_recent_recall_near_one(self) -> None:
        just_now = datetime.now(timezone.utc) - timedelta(minutes=5)
        factor = compute_decay_factor(just_now, 3)
        # Recently recalled with moderate repetitions — should be > 0.95
        assert factor > 0.95

    def test_long_gap_decays(self) -> None:
        two_days_ago = datetime.now(timezone.utc) - timedelta(hours=240)
        factor = compute_decay_factor(two_days_ago, 0)
        # Long gap, first recall — should be heavily decayed
        assert factor < 0.1

    def test_more_recalls_slows_decay(self) -> None:
        # Same elapsed time, different recall counts
        t = datetime.now(timezone.utc) - timedelta(hours=100)
        factor_few = compute_decay_factor(t, 1)
        factor_many = compute_decay_factor(t, 10)
        assert factor_many > factor_few

    def test_decay_factor_range(self) -> None:
        """Ensure all values are in [0, 1]."""
        for hours in [0, 1, 24, 168, 720]:
            for count in [0, 1, 5, 20]:
                t = datetime.now(timezone.utc) - timedelta(hours=hours)
                factor = compute_decay_factor(t, count)
                assert 0.0 <= factor <= 1.0

    def test_strength_never_zero(self) -> None:
        """Even never-recalled memories aren't instantly dead at t=0."""
        t = datetime.now(timezone.utc)
        factor = compute_decay_factor(t, 0)
        # e^(-0/S) = e^0 = 1.0 — but we return 1.0 early for recall_count == 0
        assert factor == 1.0
