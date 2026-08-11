"""Process-state reset helpers — test isolation for process singletons.

Several production modules keep process-mutable state as module-level
singletons (caches, throttle counters, circuit-breaker registries, token
buffers).  These helpers reset that state between tests.  They live here —
in a test-support package that production never imports — so the production
modules don't expose test-only reset hooks in their public API.

The helpers reach into production module internals deliberately: resetting
process state is coupled to where that state lives.
"""

from __future__ import annotations

import asyncio

from prometheus_client import Counter, Gauge, Histogram, Summary

import backend.agent.nodes as _nodes
import backend.api.ratelimit as _ratelimit
import backend.service.alerts as _alerts
import backend.service.retrieval as _retrieval
import backend.service.usage as _usage
import backend.shared.resilience as _resilience
import backend.shared.runtime_metrics as _runtime_metrics


def reset_auto_memory_throttle() -> None:
    """Drop auto-memory throttle state (per-thread caps + rolling window)."""
    with _nodes._auto_memory_lock:
        _nodes._auto_memory_last_write.clear()
        _nodes._auto_memory_write_count.clear()
        _nodes._auto_memory_last_content.clear()
        _nodes._auto_memory_recent_writes.clear()


async def wait_auto_memory_tasks() -> None:
    """Wait for all in-flight auto-memory background tasks to finish.

    Tests call this after exercising a node that schedules auto-memory,
    before asserting on its side effects; production never needs it.
    """
    while _nodes._auto_memory_tasks:
        await asyncio.gather(*list(_nodes._auto_memory_tasks), return_exceptions=True)


def reset_circuit_breakers() -> None:
    """Drop all circuit-breaker instances."""
    with _resilience._circuit_breakers_lock:
        _resilience._circuit_breakers.clear()


def reset_usage_buffer() -> None:
    """Drop buffered LLM usage observations and the flush-failure counter."""
    _usage._consecutive_flush_failures = 0
    with _usage._pending_lock:
        _usage._pending.clear()


def reset_alert_state() -> None:
    """Drop alert cooldown and counter baselines."""
    with _alerts._alerts_lock:
        _alerts._last_fired.clear()
        _alerts._prev_structured_failures = {}


def clear_embed_query_cache() -> None:
    """Drop cached query embeddings."""
    with _retrieval._query_embed_cache_lock:
        _retrieval._query_embed_cache.clear()


def reset_rate_limits() -> None:
    """Clear every rate-limiter bucket."""
    _ratelimit._limiter.reset()


def reset_runtime_metrics() -> None:
    """Clear every Prometheus metric.

    Labeled metrics are cleared via ``clear()``.  Label-less metrics keep
    their data in instance attributes and can't go through that path:
    prometheus_client 0.26's ``clear()`` returns early when there are no
    labels, and ``reset()`` exists only on ``Counter`` — so each label-less
    type is zeroed explicitly.  Each guard is independent — clearing must
    never mask a bug in a single metric.
    """
    for metric in (
        _runtime_metrics.HTTP_REQUESTS,
        _runtime_metrics.HTTP_DURATION,
        _runtime_metrics.LLM_CALLS,
        _runtime_metrics.LLM_DURATION,
        _runtime_metrics.LLM_TOKENS,
        _runtime_metrics.CIRCUIT_STATE,
        _runtime_metrics.CIRCUIT_OPENS,
        _runtime_metrics.CIRCUIT_REJECTIONS,
        _runtime_metrics.AGENT_SLOTS_IN_USE,
        _runtime_metrics.AGENT_SLOTS_REJECTED,
        _runtime_metrics.AGENT_STEPS,
    ):
        try:
            if getattr(metric, "_labelnames", None):
                metric.clear()
                continue
            if isinstance(metric, Counter):
                metric.reset()
            elif isinstance(metric, Gauge):
                metric._value.set(0.0)
            elif isinstance(metric, Histogram):
                metric._sum.set(0.0)
                for bucket in metric._buckets:
                    bucket.set(0.0)
                metric._created = 0.0
            elif isinstance(metric, Summary):
                metric._sum.set(0.0)
                metric._count.set(0.0)
                metric._created = 0.0
        except Exception:
            pass
