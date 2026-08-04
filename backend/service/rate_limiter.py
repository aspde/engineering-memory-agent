"""In-memory rate limiter for event-driven patrol triggers.

Prevents notification fatigue — if the same CI job fails 50 times in an hour,
EMA should trigger one patrol, not 50.

Uses an in-memory dict + asyncio.Lock.  No Redis dependency.  Restarting the
server resets all rate-limit windows (acceptable for dev/single-instance).
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple sliding-window rate limiter keyed by arbitrary strings.

    Usage::

        limiter = RateLimiter()
        if limiter.is_allowed("ci:build-frontend", window_seconds=3600):
            await trigger_patrol(...)
    """

    def __init__(self) -> None:
        self._timestamps: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str, window_seconds: int = 3600) -> bool:
        """Return True if *key* hasn't been seen within *window_seconds*.

        On first call (or after the window expires), records the timestamp
        and returns True.  Subsequent calls within the window return False.
        """
        now = time.monotonic()
        async with self._lock:
            last = self._timestamps.get(key)
            if last is not None and (now - last) < window_seconds:
                logger.debug("Rate-limited key=%r (last=%.0fs ago)", key, now - last)
                return False
            self._timestamps[key] = now
            return True

    async def reset(self, key: str) -> None:
        """Explicitly clear the rate limit for *key*."""
        async with self._lock:
            self._timestamps.pop(key, None)


# Module-level singleton for use by webhook routes.
_patrol_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Return the module-level rate-limiter singleton."""
    return _patrol_rate_limiter
