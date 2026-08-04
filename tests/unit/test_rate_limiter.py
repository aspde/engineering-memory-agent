"""Tests for RateLimiter — in-memory leaky-bucket for event-driven patrol."""

from __future__ import annotations

import pytest

from backend.service.rate_limiter import RateLimiter


class TestRateLimiter:
    """Verify rate-limiting behaviour."""

    @pytest.mark.asyncio
    async def test_allows_first_call(self) -> None:
        limiter = RateLimiter()
        assert await limiter.is_allowed("key-1", window_seconds=60) is True

    @pytest.mark.asyncio
    async def test_blocks_within_window(self) -> None:
        limiter = RateLimiter()
        assert await limiter.is_allowed("key-2", window_seconds=3600) is True
        # Second call within the same window should be blocked
        assert await limiter.is_allowed("key-2", window_seconds=3600) is False

    @pytest.mark.asyncio
    async def test_different_keys_independent(self) -> None:
        limiter = RateLimiter()
        assert await limiter.is_allowed("ci:job-a", window_seconds=3600) is True
        assert await limiter.is_allowed("ci:job-b", window_seconds=3600) is True

    @pytest.mark.asyncio
    async def test_reset_allows_again(self) -> None:
        limiter = RateLimiter()
        assert await limiter.is_allowed("key-3", window_seconds=3600) is True
        assert await limiter.is_allowed("key-3", window_seconds=3600) is False
        await limiter.reset("key-3")
        assert await limiter.is_allowed("key-3", window_seconds=3600) is True
