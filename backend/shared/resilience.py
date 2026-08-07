"""Provider-call resilience — transport retry (tenacity) + circuit breaker.

Business code reaches LLM/embedding providers only through the abstract
``LLMProvider`` / ``EmbeddingProvider`` interfaces; this module is the
single place where those external calls get retry/backoff and a circuit
breaker:

- :func:`call_with_resilience` / :func:`call_with_resilience_sync` run an
  SDK call under tenacity exponential backoff (with jitter), retrying only
  *transient* failures (HTTP 429 / >=500, SDK timeouts, connection errors)
  classified by :func:`is_retryable`.
- :class:`CircuitBreaker` trips after *threshold* consecutive retryable
  failures and then fails fast for the cooldown window instead of burning
  retries and tokens against a down provider.  In-memory and keyed by
  provider name; a process restart resets it (acceptable for dev /
  single-instance, same tradeoff as ``rate_limiter.py``).

Layer split: this module owns *transport* retry.  ``structured.py`` owns
*semantic* retry (JSON parse / schema validation) for ``chat_json``, so
``chat_json`` is wired to the breaker only — adding tenacity there too
would nest the two retry loops (3 transport tries × 3 semantic tries).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import anthropic
import httpx
import openai
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.asyncio import AsyncRetrying

from backend.shared.config import config

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when a circuit breaker is open — callers fail fast."""


def is_retryable(exc: BaseException) -> bool:
    """Return True for transient provider errors worth retrying.

    Retryable: HTTP 429 / >=500, SDK timeouts / connection errors, and
    generic timeouts / connection errors.  Non-retryable 4xx (400/401/403/
    404) indicate a client-side problem retrying will not fix.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return isinstance(
        exc,
        (
            openai.APITimeoutError,
            openai.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            httpx.TimeoutException,
            httpx.TransportError,
            TimeoutError,
            ConnectionError,
        ),
    )


# ── Circuit breaker ──────────────────────────────────────────────────


class CircuitBreaker:
    """Lightweight in-memory circuit breaker (CLOSED ↔ OPEN).

    - **CLOSED**: failures accumulate; at ``failure_threshold`` consecutive
      retryable failures the breaker trips OPEN for ``cooldown_seconds``.
    - **OPEN**: :meth:`before_call` raises :class:`CircuitOpenError` so
      callers fail fast.  When the cooldown elapses the next call
      auto-resets to CLOSED (a recovery probe; a surviving failure starts
      the count over).  No strict half-open state machine — simple first.

    Thread-safe via a ``threading.Lock`` since the async and sync provider
    paths share one instance per name.
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
    ) -> None:
        self._name = name
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._open_until = 0.0  # time.monotonic() deadline; 0.0 = closed
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._open_until > time.monotonic()

    def before_call(self) -> None:
        """Raise :class:`CircuitOpenError` if open; otherwise pass.

        Called before every provider call.  When the cooldown has elapsed
        the breaker auto-resets, admitting a recovery probe.
        """
        now = time.monotonic()
        with self._lock:
            if self._open_until == 0.0:
                return
            if now < self._open_until:
                remaining = self._open_until - now
                raise CircuitOpenError(
                    f"Circuit breaker {self._name!r} is open — provider failed "
                    f"{self._failure_threshold}x consecutively; retry in "
                    f"~{remaining:.0f}s"
                )
            logger.warning("Circuit breaker %r recovered; allowing a probe", self._name)
            self._open_until = 0.0
            self._failures = 0

    def record_success(self) -> None:
        """Reset failure count (consecutive-failure semantics)."""
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def record_failure(self) -> None:
        """Count one retryable failure; trip the breaker at the threshold."""
        now = time.monotonic()
        with self._lock:
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._open_until = now + self._cooldown_seconds
                self._failures = 0
                logger.warning(
                    "Circuit breaker %r OPEN after %d consecutive failures; "
                    "failing fast for %.0fs",
                    self._name,
                    self._failure_threshold,
                    self._cooldown_seconds,
                )


_circuit_breakers: dict[str, CircuitBreaker] = {}
_circuit_breakers_lock = threading.Lock()


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Return the named singleton breaker, created with config defaults."""
    with _circuit_breakers_lock:
        breaker = _circuit_breakers.get(name)
        if breaker is None:
            breaker = CircuitBreaker(
                name,
                failure_threshold=config.resilience.circuit_breaker_threshold,
                cooldown_seconds=config.resilience.circuit_breaker_cooldown,
            )
            _circuit_breakers[name] = breaker
        return breaker


def reset_circuit_breakers() -> None:
    """Drop all breakers — tests use this to isolate state."""
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


# ── Retry + breaker wrappers ─────────────────────────────────────────


def _retry_kwargs() -> dict[str, Any]:
    """Build tenacity retry settings from config (read lazily per call)."""
    cfg = config.resilience
    return {
        "retry": retry_if_exception(is_retryable),
        "stop": stop_after_attempt(cfg.max_attempts),
        "wait": wait_exponential_jitter(
            initial=cfg.backoff_base, max=cfg.backoff_max, jitter=0.5
        ),
        "reraise": True,
        "before_sleep": before_sleep_log(logger, logging.WARNING),
    }


async def call_with_resilience(
    name: str,
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Run *operation* (an async SDK call) under circuit breaker + retry.

    The breaker is consulted first (fast-fail when open), then the call is
    retried by tenacity.  Retryable failures count toward the breaker;
    non-retryable ones propagate untouched.
    """
    breaker = get_circuit_breaker(name)
    breaker.before_call()
    retrier = AsyncRetrying(**_retry_kwargs())
    try:
        result = await retrier(operation)
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        raise
    breaker.record_success()
    return result


def call_with_resilience_sync(
    name: str,
    operation: Callable[[], T],
) -> T:
    """Sync variant of :func:`call_with_resilience`."""
    breaker = get_circuit_breaker(name)
    breaker.before_call()
    retrier = Retrying(**_retry_kwargs())
    try:
        result = retrier(operation)
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        raise
    breaker.record_success()
    return result


@contextlib.asynccontextmanager
async def circuit_breaker_guard(name: str):
    """Async context manager applying the breaker only (no tenacity retry).

    Used by ``chat_json``: its transport retries live in ``structured.py``
    (semantic retry), so this path only needs the breaker to fail fast when
    the provider is down.
    """
    breaker = get_circuit_breaker(name)
    breaker.before_call()
    try:
        yield
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        raise
    breaker.record_success()
