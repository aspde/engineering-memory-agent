"""Runtime health metrics — Prometheus exposition for the EMA process.

The cost observability already in ``backend/service/usage.py`` answers
"how much did we spend" (persisted to ``llm_usage``).  This module answers
the *health* questions Prometheus is built for — "is the service healthy
right now, how slow is it, where is the pressure" — as in-memory time
series scraped at ``GET /metrics`` (text format, `prometheus_client`).

Metrics are captured at the existing choke points, not by sprinkling new
ones:

- **HTTP layer** — a Starlette middleware records per-route request count,
  latency histogram, and status code distribution.
- **LLM calls** — ``record_llm_call`` is called from
  ``backend/service/usage.record_call`` (the single point every provider
  call passes through), so call count / latency / tokens by scenario and
  status all line up with the cost rows.
- **Circuit breakers** — ``backend/shared/resilience.py`` reports the
  open/half-open state, how many times the breaker tripped, and how many
  calls were rejected while open.
- **Agent concurrency** — the interactive-run slot counter in
  ``backend/service/agent_service.py`` reports in-flight runs and the 503
  rejections (the ``MAX_AGENT_CONCURRENCY`` cap).
- **ReAct loop** — ``backend/api/routes/agent_routes.py`` observes the
  per-run step count, the same over-call signal the task eval measures
  offline, now observable on live traffic.

All metric objects are thread-safe (prometheus_client).  Each record
function is gated on ``config.metrics_enabled`` (default on) and never
raises — health observability must not back-pressure the hot paths it
observes, mirroring the usage-tracing contract.

Naming follows the Prometheus convention: ``ema_``-prefixed, unit-suffixed
(``_total`` for counters, ``_seconds`` for durations, ``_bytes`` not used).
"""

from __future__ import annotations

import time
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from backend.shared.config import config

# A dedicated registry (not the global default) so tests can reset it in
# isolation and /metrics serves exactly EMA's metrics, not library noise.
_REGISTRY = CollectorRegistry()


# ── HTTP ──────────────────────────────────────────────────────────────

HTTP_REQUESTS = Counter(
    "ema_http_requests_total",
    "HTTP requests served, by method / route path / status code.",
    ["method", "path", "status"],
    registry=_REGISTRY,
)
HTTP_DURATION = Histogram(
    "ema_http_request_duration_seconds",
    "HTTP request latency by method / route path.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=_REGISTRY,
)


# ── LLM calls (fed by usage.record_call — the single provider choke point) ──

LLM_CALLS = Counter(
    "ema_llm_calls_total",
    "LLM provider calls, by scenario and status (success|error).",
    ["scenario", "status"],
    registry=_REGISTRY,
)
LLM_DURATION = Histogram(
    "ema_llm_duration_seconds",
    "LLM call latency by scenario.",
    ["scenario"],
    buckets=(0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0),
    registry=_REGISTRY,
)
LLM_TOKENS = Counter(
    "ema_llm_tokens_total",
    "LLM tokens by scenario and kind (input|output|total).",
    ["scenario", "kind"],
    registry=_REGISTRY,
)


# ── Circuit breakers ──────────────────────────────────────────────────

# 1 = open / half-open (not admitting ordinary calls), 0 = closed.
CIRCUIT_STATE = Gauge(
    "ema_circuit_breaker_state",
    "1 when the named circuit breaker is open/half-open, 0 when closed.",
    ["name"],
    registry=_REGISTRY,
)
CIRCUIT_OPENS = Counter(
    "ema_circuit_breaker_opens_total",
    "Number of times the named breaker tripped OPEN.",
    ["name"],
    registry=_REGISTRY,
)
CIRCUIT_REJECTIONS = Counter(
    "ema_circuit_breaker_rejections_total",
    "Calls failed fast against an open breaker.",
    ["name"],
    registry=_REGISTRY,
)


# ── Agent concurrency + loop discipline ───────────────────────────────

AGENT_SLOTS_IN_USE = Gauge(
    "ema_agent_slots_in_use",
    "Interactive agent runs currently in flight (MAX_AGENT_CONCURRENCY cap).",
    registry=_REGISTRY,
)
AGENT_SLOTS_REJECTED = Counter(
    "ema_agent_slots_rejected_total",
    "Chat requests refused with 503 because the concurrency cap was reached.",
    registry=_REGISTRY,
)
AGENT_STEPS = Histogram(
    "ema_agent_steps",
    "ReAct loop steps per completed agent run (call_llm invocations).",
    buckets=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
    registry=_REGISTRY,
)


# ── Record helpers (gated on config.metrics_enabled) ──────────────────


def _enabled() -> bool:
    return bool(config.metrics_enabled)


def record_http_request(method: str, path: str, status: int, latency_s: float) -> None:
    """Record one HTTP request. ``path`` should be the *route* path, not a
    raw URL, so label cardinality stays bounded (never one series per id)."""
    if not _enabled():
        return
    try:
        method = str(method or "").upper() or "UNKNOWN"
        path = str(path or "unmatched")
        HTTP_REQUESTS.labels(method=method, path=path, status=str(int(status))).inc()
        HTTP_DURATION.labels(method=method, path=path).observe(latency_s)
    except Exception:
        pass  # observability must never raise on the request path


def record_llm_call(
    *,
    scenario: str,
    status: str,
    latency_ms: int,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> None:
    """Record one LLM call. Called from usage.record_call so the Prometheus
    series and the persisted ``llm_usage`` rows are fed by the same event."""
    if not _enabled():
        return
    try:
        scenario = str(scenario or "default")
        status = str(status or "success")
        LLM_CALLS.labels(scenario=scenario, status=status).inc()
        LLM_DURATION.labels(scenario=scenario).observe(max(float(latency_ms), 0.0) / 1000.0)
        for kind, value in (
            ("input", input_tokens),
            ("output", output_tokens),
            ("total", total_tokens),
        ):
            if value:
                LLM_TOKENS.labels(scenario=scenario, kind=kind).inc(int(value))
    except Exception:
        pass


def set_circuit_breaker_state(name: str, is_open: bool) -> None:
    """Mirror a breaker's open/closed state to the gauge."""
    if not _enabled():
        return
    try:
        CIRCUIT_STATE.labels(name=str(name)).set(1.0 if is_open else 0.0)
    except Exception:
        pass


def inc_circuit_breaker_opens(name: str) -> None:
    if not _enabled():
        return
    try:
        CIRCUIT_OPENS.labels(name=str(name)).inc()
    except Exception:
        pass


def inc_circuit_breaker_rejections(name: str) -> None:
    if not _enabled():
        return
    try:
        CIRCUIT_REJECTIONS.labels(name=str(name)).inc()
    except Exception:
        pass


def set_agent_slots_in_use(n: int) -> None:
    """Mirror the interactive agent-run slot counter."""
    if not _enabled():
        return
    try:
        AGENT_SLOTS_IN_USE.set(max(int(n), 0))
    except Exception:
        pass


def inc_agent_slots_rejected() -> None:
    if not _enabled():
        return
    try:
        AGENT_SLOTS_REJECTED.inc()
    except Exception:
        pass


def observe_agent_steps(n_steps: int) -> None:
    """Record one completed agent run's ReAct step count."""
    if not _enabled():
        return
    try:
        AGENT_STEPS.observe(max(int(n_steps), 0))
    except Exception:
        pass


# ── ASGI middleware ───────────────────────────────────────────────────


class MetricsMiddleware:
    """Starlette/ASGI middleware that records per-route request metrics.

    Route-path label: prefers the matched route's template path (``scope``
    carries ``route`` after matching) so ids in URLs never explode the label
    cardinality; falls back to the raw path (404s, unmatched routes).
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status_holder: list[int] = [0]
        original_send = send

        async def _send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status_holder[0] = message.get("status", 0)
            await original_send(message)

        try:
            await self.app(scope, receive, _send)
        except Exception:
            # The app raised — a server-side failure even if response.start had
            # already been sent (mid-stream).  Record it as 500, not the partial
            # status already flushed, so the status distribution surfaces the
            # failed request instead of masking it as a 200.  Re-raise so the
            # ASGI server still handles the exception.
            record_http_request(
                str(scope.get("method", "")),
                _route_path(scope),
                500,
                time.perf_counter() - start,
            )
            raise
        record_http_request(
            str(scope.get("method", "")),
            _route_path(scope),
            status_holder[0] or 500,
            time.perf_counter() - start,
        )


def _route_path(scope: dict[str, Any]) -> str:
    """The matched route template path, or the raw path for unmatched routes."""
    route = scope.get("route")
    if route is not None:
        path = getattr(route, "path", None)
        if path:
            return str(path)
    return str(scope.get("path", "unmatched"))


# ── Exposition ────────────────────────────────────────────────────────


def render_metrics() -> str:
    """Render all EMA metrics in the Prometheus text format."""
    return generate_latest(_REGISTRY).decode("utf-8")


def reset_runtime_metrics() -> None:
    """Clear every metric — tests call this for isolation.

    Labeled metrics are cleared via ``clear()`` (drops every label set; the
    parent's ``_value`` is None so ``reset()`` would raise); unlabeled ones
    via ``reset()``.  Each guarded — clearing must never mask a bug in a
    single metric.
    """
    for metric in (
        HTTP_REQUESTS,
        HTTP_DURATION,
        LLM_CALLS,
        LLM_DURATION,
        LLM_TOKENS,
        CIRCUIT_STATE,
        CIRCUIT_OPENS,
        CIRCUIT_REJECTIONS,
        AGENT_SLOTS_IN_USE,
        AGENT_SLOTS_REJECTED,
        AGENT_STEPS,
    ):
        try:
            if getattr(metric, "_labelnames", None):
                metric.clear()
            else:
                metric.reset()
        except Exception:
            pass
    AGENT_SLOTS_IN_USE.set(0.0)
