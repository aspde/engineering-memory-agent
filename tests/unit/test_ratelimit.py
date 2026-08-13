"""Unit tests for the token-bucket rate limiter (``backend/api/ratelimit.py``).

Tests the pure limiter algorithm with a fake monotonic clock — no HTTP app,
no config coupling.  The ASGI middleware wiring (429 + Retry-After on real
requests, tier selection, test-env bypass) is covered in
``tests/api/test_ratelimit.py``.
"""

from __future__ import annotations

from backend.api.ratelimit import RateLimiter


class _FakeClock:
    """Stand-in for ``time.monotonic`` so tests can advance time deterministically."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_limiter() -> tuple[RateLimiter, _FakeClock]:
    clock = _FakeClock()
    limiter = RateLimiter()
    return limiter, clock


class TestTokenBucket:
    def test_first_request_always_allowed(self, monkeypatch) -> None:
        limiter, clock = _make_limiter()
        monkeypatch.setattr("backend.api.ratelimit.time.monotonic", clock)
        allowed, retry_after = limiter.allow(
            tier="chat", key="k", requests=5, window_seconds=60
        )
        assert allowed is True
        assert retry_after == 0

    def test_allows_up_to_capacity(self, monkeypatch) -> None:
        limiter, clock = _make_limiter()
        monkeypatch.setattr("backend.api.ratelimit.time.monotonic", clock)
        # 5 requests / 60s → capacity 5, so five back-to-back requests pass.
        for _ in range(5):
            allowed, _ = limiter.allow(
                tier="chat", key="k", requests=5, window_seconds=60
            )
            assert allowed is True
        # The sixth is refused.
        allowed, retry_after = limiter.allow(
            tier="chat", key="k", requests=5, window_seconds=60
        )
        assert allowed is False
        assert retry_after > 0

    def test_retry_after_is_bounded_and_positive(self, monkeypatch) -> None:
        limiter, clock = _make_limiter()
        monkeypatch.setattr("backend.api.ratelimit.time.monotonic", clock)
        for _ in range(5):
            limiter.allow(tier="chat", key="k", requests=5, window_seconds=60)
        _, retry_after = limiter.allow(
            tier="chat", key="k", requests=5, window_seconds=60
        )
        # Refill rate is 5/60 → one token every 12s; report at least 1s and
        # no more than the window (float drift can push ceil over 12).
        assert retry_after >= 1
        assert retry_after <= 60

    def test_bucket_refills_over_time(self, monkeypatch) -> None:
        limiter, clock = _make_limiter()
        monkeypatch.setattr("backend.api.ratelimit.time.monotonic", clock)
        for _ in range(5):
            limiter.allow(tier="chat", key="k", requests=5, window_seconds=60)
        assert limiter.allow(
            tier="chat", key="k", requests=5, window_seconds=60
        )[0] is False
        # After 12s one token has refilled → the next request passes.
        clock.advance(12.0)
        assert limiter.allow(
            tier="chat", key="k", requests=5, window_seconds=60
        )[0] is True

    def test_distinct_keys_have_independent_buckets(self, monkeypatch) -> None:
        limiter, clock = _make_limiter()
        monkeypatch.setattr("backend.api.ratelimit.time.monotonic", clock)
        for _ in range(3):
            limiter.allow(tier="chat", key="a", requests=3, window_seconds=60)
        # Key "a" is exhausted; key "b" starts full.
        assert limiter.allow(tier="chat", key="a", requests=3, window_seconds=60)[0] is False
        assert limiter.allow(tier="chat", key="b", requests=3, window_seconds=60)[0] is True

    def test_tiers_are_independent(self, monkeypatch) -> None:
        limiter, clock = _make_limiter()
        monkeypatch.setattr("backend.api.ratelimit.time.monotonic", clock)
        for _ in range(2):
            limiter.allow(tier="chat", key="k", requests=2, window_seconds=60)
        # chat exhausted, general untouched.
        assert limiter.allow(tier="chat", key="k", requests=2, window_seconds=60)[0] is False
        assert limiter.allow(tier="general", key="k", requests=2, window_seconds=60)[0] is True

    def test_reset_clears_all_buckets(self, monkeypatch) -> None:
        limiter, clock = _make_limiter()
        monkeypatch.setattr("backend.api.ratelimit.time.monotonic", clock)
        for _ in range(5):
            limiter.allow(tier="chat", key="k", requests=5, window_seconds=60)
        assert limiter.allow(tier="chat", key="k", requests=5, window_seconds=60)[0] is False
        limiter.reset()
        assert limiter.allow(tier="chat", key="k", requests=5, window_seconds=60)[0] is True

    def test_idle_buckets_are_pruned_under_pressure(self, monkeypatch) -> None:
        limiter, clock = _make_limiter()
        monkeypatch.setattr("backend.api.ratelimit.time.monotonic", clock)
        # Small cap so pruning actually engages.
        limiter._MAX_BUCKETS = 5
        limiter._IDLE_TIMEOUT = 100.0
        for i in range(5):
            limiter.allow(tier="general", key=f"k{i}", requests=10, window_seconds=60)
        # The first bucket is old enough to prune; the new key evicts it.
        clock.advance(150.0)
        limiter.allow(tier="general", key="k-new", requests=10, window_seconds=60)
        assert len(limiter._buckets) <= 5
