"""Thread-safe in-flight slot counters for concurrency caps.

Two callers bound in-flight work with the same pattern — a thread-locked
counter whose cap is read live from config:

- interactive agent runs (``backend/service/agent_service.py``)
- scenario runs (``backend/service/scenarios/__init__.py``)

Both stay plain counters rather than ``asyncio.Semaphore`` so they are safe
across pytest's function-scoped event loops (a semaphore binds to the loop
that created it).  This module is the single implementation; each caller
keeps its own module-level instance with its own cap and metrics hooks.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class SlotLimiter:
    """Bounded in-flight counter, safe to touch from any event loop.

    ``limit`` is a zero-arg callable evaluated on every ``try_acquire`` so a
    runtime config change takes effect immediately (same convention as the
    per-key limiter in ``backend/api/ratelimit.py``).  ``on_reject`` fires
    when an acquire is refused (metrics); ``on_change`` fires with the new
    count after every successful acquire/release (metrics).  Both hooks run
    while the internal lock is held — they must be cheap and must not
    re-enter this limiter.
    """

    def __init__(
        self,
        limit: Callable[[], int],
        *,
        on_reject: Callable[[], None] | None = None,
        on_change: Callable[[int], None] | None = None,
    ) -> None:
        self._limit = limit
        self._on_reject = on_reject
        self._on_change = on_change
        self._active = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> int:
        """Currently held slots (test fixtures read this to drain)."""
        with self._lock:
            return self._active

    def try_acquire(self) -> bool:
        """Reserve one slot; ``False`` when the cap is already reached."""
        with self._lock:
            if self._active >= self._limit():
                if self._on_reject is not None:
                    self._on_reject()
                return False
            self._active += 1
            if self._on_change is not None:
                self._on_change(self._active)
            return True

    def release(self) -> None:
        """Release a slot acquired by :meth:`try_acquire`."""
        with self._lock:
            self._active = max(self._active - 1, 0)
            if self._on_change is not None:
                self._on_change(self._active)
