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
  retries and tokens against a down provider.  After the cooldown the
  breaker admits calls again — a persistent failure simply re-opens it
  after the threshold.  Deliberately no half-open recovery probe: callers
  sit behind tenacity transport retry and, on the LLM path, a cross-provider
  failover that engages the moment this breaker opens, so the cooldown
  window only needs to fail fast.  In-memory and keyed by provider name; a
  process restart resets it (acceptable for dev / single-instance).

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
from backend.shared.runtime_metrics import (
    inc_circuit_breaker_opens,
    inc_circuit_breaker_rejections,
    set_circuit_breaker_state,
)

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
    """Lightweight in-memory circuit breaker (CLOSED ↔ OPEN ↔ HALF-OPEN).

    - **CLOSED**: failures accumulate; at ``failure_threshold`` consecutive
      retryable failures the breaker trips OPEN for ``cooldown_seconds``.
    - **OPEN**: :meth:`before_call` raises :class:`CircuitOpenError` so
      callers fail fast.  When the cooldown elapses the breaker admits calls
      again (back to CLOSED); a persistent provider failure simply trips it
      again after the threshold.

    Deliberately NO half-open recovery probe.  Callers already sit behind
    tenacity transport retry, and on the LLM path a cross-provider failover
    engages the moment this breaker opens — so the cooldown window only
    needs to fail fast.  Re-admitting a few calls after it elapses is a
    standard way to re-probe the provider (resilience4j's OPEN→CLOSED is the
    same), and it drops the probe-token concurrency machinery (stale-probe
    deadlines, token-matched closes) that a half-open state would need to be
    correct at this process's low concurrency.

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
        """Admit a call or raise :class:`CircuitOpenError`.

        Closed → pass.  OPEN (cooldown not elapsed) → fail fast.  When the
        cooldown has elapsed the next caller re-opens the breaker (back to
        CLOSED) and is admitted — re-probing the provider with a live call
        instead of a separate half-open probe state.
        """
        now = time.monotonic()
        with self._lock:
            if now < self._open_until:
                remaining = self._open_until - now
                inc_circuit_breaker_rejections(self._name)
                raise CircuitOpenError(
                    f"Circuit breaker {self._name!r} is open — provider failed "
                    f"{self._failure_threshold}x consecutively; retry in "
                    f"~{remaining:.0f}s"
                )
            if self._open_until != 0.0:
                logger.warning("Circuit breaker %r recovered", self._name)
                self._open_until = 0.0
                self._failures = 0
                set_circuit_breaker_state(self._name, False)

    def record_success(self) -> None:
        """Reset the consecutive-failure count (CLOSED only).

        While the breaker is OPEN a success does not close it — only the
        cooldown-elapsed re-admission in :meth:`before_call` may.  (There is
        no half-open probe to token-match, so a stale success from an
        in-flight call that crossed a trip is simply ignored.)  The gauge is
        mirrored to closed so a healthy provider stays visible as 0.
        """
        with self._lock:
            if self._open_until == 0.0:
                self._failures = 0
                set_circuit_breaker_state(self._name, False)

    def record_failure(self) -> None:
        """Count one retryable failure; trip the breaker at the threshold."""
        now = time.monotonic()
        with self._lock:
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._open_until = now + self._cooldown_seconds
                self._failures = 0
                inc_circuit_breaker_opens(self._name)
                logger.warning(
                    "Circuit breaker %r OPEN after %d consecutive failures; "
                    "failing fast for %.0fs",
                    self._name,
                    self._failure_threshold,
                    self._cooldown_seconds,
                )
                set_circuit_breaker_state(self._name, True)


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
        result: Any = await retrier(operation)
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


@contextlib.asynccontextmanager
async def resilient_stream_guard(name: str, connect: Callable[[], Awaitable[T]]):
    """Circuit breaker + transport retry for a lazy token stream.

    ``connect`` establishes the stream (returns the stream object) and is run
    under the same tenacity transport retry as :func:`call_with_resilience`
    (a 429 / 5xx / timeout at connection time is retried before the stream
    exists).  Unlike ``call_with_resilience`` — which records breaker success
    as soon as the operation returns — success/failure here is recorded
    against the *stream body*: a stream that connects but dies mid-iteration
    is counted as a breaker failure instead of a success.

    Tokens already delivered are never retried: the body is consumed exactly
    once; only connection establishment retries.
    """
    breaker = get_circuit_breaker(name)
    breaker.before_call()
    retrier = AsyncRetrying(**_retry_kwargs())
    try:
        stream: Any = await retrier(connect)
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        raise
    try:
        yield stream
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        raise
    breaker.record_success()
