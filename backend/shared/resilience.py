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
  breaker admits exactly one half-open recovery probe; the probe's outcome
  (success closes, failure re-opens) decides.  In-memory and keyed by
  provider name; a process restart resets it (acceptable for dev /
  single-instance).

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
      callers fail fast.  When the cooldown elapses the breaker moves to
      HALF-OPEN.
    - **HALF-OPEN**: exactly one recovery probe is admitted; every other
      caller keeps failing fast (no probe stampede against a still-down
      provider).  The probe's caller passes the token :meth:`before_call`
      returned to :meth:`record_success`, and that is the *only* call that may
      close the breaker; probe failure (:meth:`record_failure`) puts it back to
      OPEN.  A stale success from a call admitted before the breaker tripped
      carries no token and is ignored, so a late success can't re-close a
      breaker that has since opened (which would oscillate
      OPEN→CLOSED→OPEN while hammering a still-down provider).  A probe that
      never resolves (e.g. its task was cancelled) is treated as stale after
      one cooldown window and a fresh probe may be admitted.

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
        self._probing = False  # half-open: a recovery probe is in flight
        self._probe_deadline = 0.0  # stale-probe guard, time.monotonic()
        self._probe_token: int | None = None  # identity of the admitted probe
        self._probe_seq = 0  # monotonic probe ids (stale tokens never collide)
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_open(self) -> bool:
        with self._lock:
            now = time.monotonic()
            return self._open_until > now or (
                self._probing and self._probe_deadline > now
            )

    def before_call(self) -> int | None:
        """Admit a call or raise :class:`CircuitOpenError`; return a probe token.

        Closed → pass (returns ``None``).  OPEN (cooldown not elapsed) → fail
        fast.  HALF-OPEN (cooldown elapsed, live probe in flight) → only the
        probe's caller was admitted earlier; everyone else fails fast until the
        probe's outcome is recorded.  When the cooldown has elapsed and no live
        probe exists, the next caller is admitted as the single recovery probe
        and gets a fresh probe token — that caller must pass the token back to
        :meth:`record_success` so a stale success from an earlier call cannot
        close the breaker.
        """
        now = time.monotonic()
        with self._lock:
            if self._open_until == 0.0 and not self._probing:
                return None
            if now < self._open_until:
                remaining = self._open_until - now
                inc_circuit_breaker_rejections(self._name)
                raise CircuitOpenError(
                    f"Circuit breaker {self._name!r} is open — provider failed "
                    f"{self._failure_threshold}x consecutively; retry in "
                    f"~{remaining:.0f}s"
                )
            if self._probing and now < self._probe_deadline:
                inc_circuit_breaker_rejections(self._name)
                raise CircuitOpenError(
                    f"Circuit breaker {self._name!r} is half-open — a recovery "
                    f"probe is in flight; retry shortly"
                )
            logger.warning("Circuit breaker %r recovered; allowing a probe", self._name)
            self._open_until = 0.0
            self._failures = 0
            self._probing = True
            self._probe_deadline = now + self._cooldown_seconds
            self._probe_seq += 1
            self._probe_token = self._probe_seq
            set_circuit_breaker_state(self._name, True)  # half-open
            return self._probe_token

    def record_success(self, probe_token: int | None = None) -> None:
        """Reset the consecutive-failure count and, for the admitted probe,
        close the breaker.

        Only the half-open recovery probe may close the breaker: its caller
        passes the token :meth:`before_call` returned when admitting it.  A
        success from a call admitted *before* the breaker tripped (still in
        flight, no token) — or from a stale earlier probe — must not re-close
        an OPEN / HALF-OPEN breaker; doing so would let the breaker oscillate
        OPEN→CLOSED→OPEN against a still-down provider.
        """
        with self._lock:
            if self._probing:
                if probe_token is not None and probe_token == self._probe_token:
                    self._probing = False
                    self._probe_token = None
                    self._open_until = 0.0
                    self._failures = 0
                    set_circuit_breaker_state(self._name, False)
                return
            if self._open_until == 0.0:
                self._failures = 0
                set_circuit_breaker_state(self._name, False)

    def record_failure(self) -> None:
        """Count one retryable failure; trip the breaker at the threshold.

        A failure of the half-open recovery probe always re-opens the breaker
        — one failed probe means the provider is still down — bypassing the
        consecutive-count threshold.
        """
        now = time.monotonic()
        with self._lock:
            if self._probing:
                self._probing = False
                self._open_until = now + self._cooldown_seconds
                inc_circuit_breaker_opens(self._name)
                logger.warning(
                    "Circuit breaker %r probe failed; re-opening for %.0fs",
                    self._name,
                    self._cooldown_seconds,
                )
                set_circuit_breaker_state(self._name, True)
                return
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

    def settle_probe_failure(self) -> None:
        """Re-open a half-open probe that failed with a *non-retryable* error.

        A recovery probe must always resolve: success closes, failure
        re-opens.  ``record_failure`` only runs for retryable errors, so a
        probe raising e.g. a 401 auth error (which the retry loop never
        touches) would otherwise leave the breaker wedged HALF-OPEN: every
        other caller fails fast "probe in flight" for the rest of the
        cooldown, and a fresh probe is then admitted against a provider that
        can never authenticate.  Settling the probe here keeps the state
        machine coherent while the probe's own caller still sees the real
        exception.  No-op when the breaker isn't probing.
        """
        with self._lock:
            if not self._probing:
                return
            self._probing = False
            now = time.monotonic()
            self._open_until = now + self._cooldown_seconds
            inc_circuit_breaker_opens(self._name)
            logger.warning(
                "Circuit breaker %r probe failed with a non-retryable error; "
                "re-opening for %.0fs",
                self._name,
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
    probe_token = breaker.before_call()
    retrier = AsyncRetrying(**_retry_kwargs())
    try:
        result = await retrier(operation)
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        else:
            # This call may have been the half-open recovery probe — a
            # non-retryable probe failure (e.g. a 401) must still settle the
            # probe or the breaker stays wedged HALF-OPEN (see
            # CircuitBreaker.settle_probe_failure).
            breaker.settle_probe_failure()
        raise
    breaker.record_success(probe_token)
    return result


def call_with_resilience_sync(
    name: str,
    operation: Callable[[], T],
) -> T:
    """Sync variant of :func:`call_with_resilience`."""
    breaker = get_circuit_breaker(name)
    probe_token = breaker.before_call()
    retrier = Retrying(**_retry_kwargs())
    try:
        result = retrier(operation)
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        else:
            breaker.settle_probe_failure()
        raise
    breaker.record_success(probe_token)
    return result


@contextlib.asynccontextmanager
async def circuit_breaker_guard(name: str):
    """Async context manager applying the breaker only (no tenacity retry).

    Used by ``chat_json``: its transport retries live in ``structured.py``
    (semantic retry), so this path only needs the breaker to fail fast when
    the provider is down.
    """
    breaker = get_circuit_breaker(name)
    probe_token = breaker.before_call()
    try:
        yield
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        else:
            breaker.settle_probe_failure()
        raise
    breaker.record_success(probe_token)


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
    probe_token = breaker.before_call()
    retrier = AsyncRetrying(**_retry_kwargs())
    try:
        stream = await retrier(connect)
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        else:
            breaker.settle_probe_failure()
        raise
    try:
        yield stream
    except Exception as exc:
        if is_retryable(exc):
            breaker.record_failure()
        else:
            breaker.settle_probe_failure()
        raise
    breaker.record_success(probe_token)
