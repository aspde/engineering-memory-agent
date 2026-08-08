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
    get_daily_summary,
    get_model_summary,
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
        "est_cost": round(sum(i.get("est_cost", 0.0) for i in items), 4),
    }


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
    return {"thread_id": thread_id, "calls": await get_thread_usage(thread_id)}


@router.get("/trace/{trace_id}")
async def usage_trace(trace_id: str) -> dict[str, Any]:
    """All LLM calls in one trace — replay a single agent run end-to-end."""
    return {"trace_id": trace_id, "calls": await get_trace(trace_id)}
