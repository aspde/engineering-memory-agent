"""Usage tracing query API — reads persisted LLM-usage rows from ``llm_usage``.

Complements the in-memory counters behind ``GET /api/agent/usage`` (which
reset on restart): these endpoints answer "how many calls / tokens / errors
over the last N days, broken down by day / scenario / model", replay a single
trace, and estimate USD cost from a built-in price table.

All queries read raw rows with ``GROUP BY`` in SQL — there is no separate
aggregate table (simple first).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from backend.service.usage import (
    _redact,
    get_daily_summary,
    get_model_summary,
    get_samples,
    get_scenario_summary,
    get_thread_usage,
    get_trace,
)

router = APIRouter(prefix="/usage", tags=["usage"])


def _with_totals(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a summary list with grand totals for quick reading."""
    return {
        "items": items,
        "total_calls": sum(i.get("calls", 0) for i in items),
        "total_tokens": sum(i.get("total_tokens", 0) for i in items),
        "total_cache_read_tokens": sum(i.get("cache_read_tokens", 0) for i in items),
        "total_cache_creation_tokens": sum(
            i.get("cache_creation_tokens", 0) for i in items
        ),
        "est_cost": round(sum(i.get("est_cost", 0.0) for i in items), 4),
    }


def _redact_row_errors(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-redact persisted ``error`` text on read (defense in depth).

    Provider error strings can echo back an Authorization header / ``sk-``
    key; ``record_call`` now redacts before storing, but rows written before
    that landed still never surface a secret.  Mutates the dicts in place and
    returns the list for chaining.
    """
    for c in calls:
        if c.get("error"):
            c["error"] = _redact(c["error"])
    return calls


@router.get("/summary")
async def usage_summary(
    days: int = Query(7, ge=1, le=90),
) -> dict[str, Any]:
    """Per-day totals over the last *days* days."""
    return {"days": days, **_with_totals(await get_daily_summary(days))}


@router.get("/scenarios")
async def usage_scenarios(
    days: int = Query(7, ge=1, le=90),
) -> dict[str, Any]:
    """Per-scenario totals — the cost breakdown per LLM call site."""
    return {"days": days, **_with_totals(await get_scenario_summary(days))}


@router.get("/models")
async def usage_models(
    days: int = Query(7, ge=1, le=90),
) -> dict[str, Any]:
    """Per-(provider/model) totals with per-model estimated cost."""
    return {"days": days, **_with_totals(await get_model_summary(days))}


@router.get("/threads/{thread_id}")
async def usage_thread(thread_id: str) -> dict[str, Any]:
    """Recent LLM calls for one conversation thread."""
    return {
        "thread_id": thread_id,
        "calls": _redact_row_errors(await get_thread_usage(thread_id)),
    }


@router.get("/trace/{trace_id}")
async def usage_trace(
    trace_id: str,
    limit: int = Query(50, ge=1, le=1000),
) -> dict[str, Any]:
    """All LLM calls in one trace — replay a single agent run end-to-end."""
    return {
        "trace_id": trace_id,
        "calls": _redact_row_errors(await get_trace(trace_id, limit=limit)),
    }


@router.get("/samples")
async def usage_samples(
    scenario: str | None = Query(None, description="Filter by scenario tag"),
    status: str | None = Query(None, description="Filter by status (success/error)"),
    trace_id: str | None = Query(None, description="Filter by trace id"),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Sampled prompt/response text for post-hoc quality analysis.

    Only calls where a sample was stored (error calls always, success calls
    at ``USAGE_SAMPLE_RATE``) are returned.  This is the surface for "which
    prompt produced that hallucinated answer" or "which tool call had
    malformed JSON" — the metadata summaries can't answer those.

    Samples are re-redacted on read (defense in depth): rows persisted before
    write-time redaction landed still never surface a secret.
    """
    samples = await get_samples(
        scenario=scenario,
        status=status,
        trace_id=trace_id,
        limit=limit,
    )
    for s in _redact_row_errors(samples):
        if s.get("prompt_sample"):
            s["prompt_sample"] = _redact(s["prompt_sample"])
        if s.get("response_sample"):
            s["response_sample"] = _redact(s["response_sample"])
    return {"samples": samples}
