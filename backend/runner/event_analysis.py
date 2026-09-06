"""Event-driven analysis runner — agent reacts to an ingested event.

Phase 3's missing layer: the webhook path ingests events into memory but
never interprets them.  After a delivery reaches its terminal state, this
module runs one agent invocation that searches the memory store for
similar historical events and produces a structured verdict (known issue?
similar incidents? recommendation?), persisted to ``webhook_logs.analysis``
and — above the configured severity threshold — pushed to Feishu.

Design constraints:

* **Read-only tool surface** — the analysis verdict is a conclusion, not a
  new memory.  The LLM only sees retrieval tools, so analysis traffic can
  never pollute the knowledge base (and no write-conflict interrupt can
  occur; the interrupt check below is defensive).
* **Delivery semantics untouched** — the analysis runs in its own task,
  wrapped in try/except: any failure is logged and recorded in the
  ``analysis`` column, never reflected back into the delivery status.
* **Cooldown gate** — repeated events of the same entity (CI: job_name)
  within the window are analysed once, so a CI storm doesn't burn one
  agent run per failed build.  In-memory, like the auto-memory throttle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from backend.agent.tool_envelope import envelope_display
from backend.runner.agent_service import get_agent
from backend.service.json_extraction import extract_json_object
from backend.service.notification import send_feishu_message
from backend.shared.config import (
    SEVERITY_LEVELS,
    config,
    current_thread_id,
    current_trace_id,
)
from backend.shared.slots import SlotLimiter

logger = logging.getLogger(__name__)

# Analysis needs one retrieval pass + maybe an entity lookup — one step
# above the interactive budget of 5.  Bounded so a confused model cannot
# loop.
_EVENT_MAX_STEPS = 6

# LangGraph default recursion limit fits: 6 steps × ~4 nodes ≈ 24 << 50.
_EVENT_RECURSION_LIMIT = 50

_VALID_SEVERITIES = SEVERITY_LEVELS
_SEVERITY_RANK = {s: i for i, s in enumerate(_VALID_SEVERITIES)}

_REQUIRED_KEYS = {
    "is_known_issue",
    "similar_incidents",
    "root_cause_hypothesis",
    "recommendation",
    "severity",
}

# ── Concurrency cap ────────────────────────────────────────────────────
# Independent of the webhook extraction slots (_WEBHOOK_MAX_CONCURRENCY):
# intake and analysis are separate budgets, and an analysis backlog must
# never block ingestion.  Same plain-counter approach as scenarios/agent.
_event_slots = SlotLimiter(lambda: config.event_analysis.max_concurrency)


def _try_acquire_event_slot() -> bool:
    return _event_slots.try_acquire()


def _release_event_slot() -> None:
    _event_slots.release()


# ── Cooldown gate ──────────────────────────────────────────────────────

_cooldowns: dict[str, float] = {}


def _cooldown_key(source: str, metadata: dict, field_name: str | None) -> str | None:
    """The gate key for this event, or None when it cannot be determined."""
    if not field_name:
        return None
    value = metadata.get(field_name)
    if not isinstance(value, str) or not value.strip():
        return None
    return f"{source}:{field_name}:{value.strip()}"


def _cooldown_gate(key: str | None) -> str:
    """Return ``"skipped_cooldown"`` when *key* is inside the window, else
    record ``key`` as analysed-now and return ``"pass"``.

    A None key passes ungated (no cooldown field configured or absent from
    metadata — better to analyse than to silently drop).
    """
    if key is None:
        return "pass"
    now = time.monotonic()
    last = _cooldowns.get(key)
    if last is not None and (now - last) < config.event_analysis.cooldown_seconds:
        return "skipped_cooldown"
    _cooldowns[key] = now
    # Bound the dict: keys are source:field:value strings, one per distinct
    # entity ever seen.  Prune expired entries when it grows past a few
    # hundred — far beyond any realistic job-name cardinality between
    # restarts, and cheap because pruning is O(n) on an in-memory dict.
    if len(_cooldowns) > 512:
        cutoff = now - config.event_analysis.cooldown_seconds
        for k in [k for k, v in _cooldowns.items() if v < cutoff]:
            del _cooldowns[k]
    return "pass"


def reset_cooldowns_for_tests() -> None:
    """Clear the cooldown table (pytest isolation)."""
    _cooldowns.clear()


# ── Tool surface ───────────────────────────────────────────────────────


def _event_llm_tools() -> list:
    """Retrieval-only tool schemas for the analysis agent.

    Imported lazily to avoid an import cycle (tools.py → nodes.py → …).
    """
    from backend.agent.tools import (
        query_entity_tool,
        retrieve_chunks_tool,
        search_memories_tool,
    )

    # query_rewrite_and_search_tool is deliberately excluded: concept
    # rewriting adds an LLM call with little value for error-log matching,
    # where concrete symbols/tokens already surface in search_memories_tool.
    return [search_memories_tool, query_entity_tool, retrieve_chunks_tool]


# ── Prompt assembly ────────────────────────────────────────────────────


def build_event_user_message(content: str, metadata: dict, context_fields: list[str]) -> str:
    """Assemble the analysis user message from the enriched content.

    ``context_fields`` is the connector-declared list of metadata fields to
    surface as a trailing context line (CI: ``["branch", "source_url"]``) —
    the runner itself stays source-agnostic.
    """
    parts = ["Analyze the following event against our engineering history:", "", content]
    context_line = []
    for field in [f.strip() for f in context_fields if f.strip()]:
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            context_line.append(f"{field}: {value.strip()}")
    if context_line:
        parts.append("")
        parts.append(" / ".join(context_line))
    return "\n".join(parts)


# ── Result handling ────────────────────────────────────────────────────


def validate_analysis(analysis: dict | None) -> tuple[dict | None, str | None]:
    """Check the parsed JSON against the output contract.

    Returns ``(analysis, error)`` — exactly one is non-None.  A malformed
    verdict must not be treated as "no known issue": downstream severity
    gating would silently skip notification on garbage input.
    """
    if analysis is None:
        return None, "empty response"
    if not isinstance(analysis, dict):
        return None, f"expected JSON object, got {type(analysis).__name__}"
    missing = sorted(_REQUIRED_KEYS - set(analysis.keys()))
    if missing:
        return None, f"missing required keys: {', '.join(missing)}"
    severity = analysis.get("severity")
    if severity not in _VALID_SEVERITIES:
        return None, f"invalid severity: {severity!r} (must be info|warning|critical)"
    return analysis, None


def should_notify(analysis: dict) -> bool:
    """Severity-threshold and switch check for the Feishu push.

    ``config.event_analysis.notify_severity`` is enum-validated at startup
    (``validate_config``), so the threshold rank is always resolvable.
    An analysis with a missing/unknown severity never notifies — it can
    only reach here through the completed path, which validate_analysis
    already gated.
    """
    if not config.event_analysis.notify_enabled:
        return False
    threshold = _SEVERITY_RANK[config.event_analysis.notify_severity]
    severity = analysis.get("severity")
    # validate_analysis already guaranteed a valid severity on the completed
    # path; an unexpected shape reads as below-threshold rather than crashing.
    return _SEVERITY_RANK.get(severity, -1) >= threshold if isinstance(severity, str) else False


def format_feishu_card(
    source: str, title_entity: str | None, metadata: dict, analysis: dict
) -> tuple[str, str]:
    """Render the analysis as ``(title, markdown)`` for a Feishu card.

    ``title_entity`` is the connector-declared metadata field shown after
    the source name (CI: ``job_name``); without it the title is just the
    source — no connector-specific key is hardcoded here.
    """
    entity = ""
    if title_entity:
        value = metadata.get(title_entity)
        if isinstance(value, str) and value.strip():
            entity = f": {value.strip()}"
    title = f"EMA 事件分析 · {source}{entity}".rstrip(": ")
    lines: list[str] = []
    known = analysis.get("is_known_issue")
    lines.append(f"**已知问题重演**: {'是' if known else '否（首次出现）'}")
    similar = analysis.get("similar_incidents") or []
    if similar:
        lines.append("**相似历史事件**:")
        for inc in similar[:5]:
            mid = str(inc.get("memory_id", ""))[:8]
            summary = str(inc.get("summary", ""))[:120]
            sim = inc.get("similarity")
            sim_text = f", 相似度 {sim:.2f}" if isinstance(sim, (int, float)) else ""
            lines.append(f"- ({mid}{sim_text}) {summary}")
    hypothesis = analysis.get("root_cause_hypothesis", "")
    if hypothesis:
        lines.append(f"**根因假设**: {hypothesis}")
    recommendation = analysis.get("recommendation", "")
    if recommendation:
        lines.append(f"**建议**: {recommendation}")
    lines.append(f"**严重级别**: {analysis.get('severity')}")
    return title, "\n".join(lines)


async def persist_analysis(delivery_id: str, record: dict) -> None:
    """Write the analysis record onto the delivery row.

    Failure here is logged, never raised — the delivery row itself is
    already terminal; losing the analysis column must not crash the task
    after the work was done.
    """
    from sqlalchemy import text

    from backend.db import get_session_factory

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text("UPDATE webhook_logs SET analysis = :analysis WHERE id = :id"),
                {"id": delivery_id, "analysis": json.dumps(record, ensure_ascii=False)},
            )
            await session.commit()
    except Exception:
        logger.exception(
            "Failed to persist analysis for delivery %s", delivery_id
        )


# ── Event context ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class EventContext:
    """Everything the analysis pipeline needs about one delivered event.

    The five fields travel together through gate → runner, so they are one
    value object rather than a clump of parallel parameters.  ``connector``
    carries the per-source analysis declarations (prompt key, cooldown
    field, display hints), keeping the runner source-agnostic.
    """

    source: str
    delivery_id: str
    content: str
    metadata: dict
    connector: Any  # backend.connectors.base.Connector (avoided at runtime)
    # Memory id the ingestion wrote for this event, when it landed as a new
    # memory — lets the postmortem auto-trigger anchor its draft on the
    # concrete incident record.  Empty on conflict/failed deliveries.
    memory_id: str = ""


# ── Runner ─────────────────────────────────────────────────────────────


async def run_event_analysis(event: EventContext) -> None:
    """Analyse one delivered event and persist/push the verdict.

    Called by ``maybe_analyze_event`` after the gates pass.  Never
    raises — every failure path lands in ``webhook_logs.analysis`` with
    ``status: failed`` (or is logged when persistence itself fails).
    """
    source = event.source
    delivery_id = event.delivery_id
    from backend.service.prompts import get_prompt

    connector = event.connector
    display = connector.event_analysis_display or {}
    try:
        _, system_prompt = get_prompt(connector.event_analysis_prompt_key)

        user_message = build_event_user_message(
            event.content,
            event.metadata,
            list(display.get("context_fields", [])),
        )
        thread_id = f"{source}-event-{delivery_id}"

        token = current_thread_id.set(thread_id)
        trace_token = current_trace_id.set(f"webhook:{delivery_id}")
        try:
            agent = get_agent(
                approval_required_tools=frozenset(),
                max_steps=_EVENT_MAX_STEPS,
                llm_tools=_event_llm_tools(),
            )
            async with asyncio.timeout(config.event_analysis.timeout_seconds):
                result = await agent.ainvoke(
                    input={
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ]
                    },
                    config={
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": _EVENT_RECURSION_LIMIT,
                    },
                )
        finally:
            current_thread_id.reset(token)
            current_trace_id.reset(trace_token)

        # Defensive: a read-only tool surface cannot pause on approval or
        # conflict today, but if someone later widens the tool set without
        # revisiting this runner, surface the pause instead of fabricating
        # a completed verdict from whatever message happens to be last.
        interrupts = result.get("__interrupt__")
        if interrupts:
            payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
            logger.warning(
                "Event analysis %s (%s) interrupted for human review: %r",
                delivery_id, source, payload,
            )
            await persist_analysis(
                delivery_id,
                {"status": "interrupted", "interrupt": repr(payload)[:500]},
            )
            return

        raw_text = str(result.get("final_response", "") or "")
        if not raw_text:
            messages = result.get("messages", [])
            for m in reversed(messages):
                if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None):
                    raw_text = envelope_display(str(m.content))
                    break

        graph_error = result.get("error")
        analysis, validation_error = validate_analysis(extract_json_object(raw_text))
        if analysis is None:
            reason = (
                f"agent error: {graph_error}" if graph_error else validation_error
            )
            logger.warning("Event analysis %s failed: %s", delivery_id, reason)
            await persist_analysis(
                delivery_id,
                {
                    "status": "failed",
                    "error": reason,
                    "raw_output": raw_text[:5000],
                },
            )
            return

        notified = False
        if should_notify(analysis):
            title, markdown = format_feishu_card(
                source, display.get("title_entity"), event.metadata, analysis
            )
            ok, detail = await send_feishu_message(markdown, msg_type="interactive", title=title)
            notified = ok
            if not ok:
                logger.warning(
                    "Event analysis %s: Feishu push failed: %s", delivery_id, detail
                )

        await persist_analysis(
            delivery_id,
            {"status": "completed", "notified": notified, **analysis},
        )
        logger.info(
            "Event analysis %s (%s) completed — known_issue=%s severity=%s notified=%s",
            delivery_id,
            source,
            analysis.get("is_known_issue"),
            analysis.get("severity"),
            notified,
        )

        maybe_trigger_postmortem(analysis, event)

    except asyncio.CancelledError:
        raise
    except TimeoutError:
        logger.warning(
            "Event analysis %s (%s) timed out after %ss",
            delivery_id, source, config.event_analysis.timeout_seconds,
        )
        await persist_analysis(
            delivery_id,
            {
                "status": "failed",
                "error": f"timeout after {config.event_analysis.timeout_seconds}s",
            },
        )
    except Exception as exc:
        logger.exception("Event analysis %s (%s) failed unexpectedly", delivery_id, source)
        await persist_analysis(
            delivery_id, {"status": "failed", "error": str(exc)}
        )


# ── Postmortem auto-trigger (Phase 4) ──────────────────────────────────

# Repeated known-issue events of the same cooldown key within this window
# spawn one postmortem draft, not one per recurrence — the scenario costs a
# full agent run (up to SCENARIO_TIMEOUT_SECONDS), and the draft for an
# event analysed an hour ago adds nothing.  Independent of the analysis
# cooldown above: analysis re-runs on every non-cooldown event, the
# postmortem fires far less often.
_POSTMORTEM_TRIGGER_COOLDOWN_SECONDS = 6 * 3600

_postmortem_triggers: dict[str, float] = {}


def _postmortem_gate(key: str | None) -> bool:
    """True when this cooldown key may spawn a postmortem run now."""
    if key is None:
        return True
    now = time.monotonic()
    last = _postmortem_triggers.get(key)
    if last is not None and (now - last) < _POSTMORTEM_TRIGGER_COOLDOWN_SECONDS:
        return False
    _postmortem_triggers[key] = now
    return True


def reset_postmortem_triggers_for_tests() -> None:
    """Clear the postmortem trigger table (pytest isolation)."""
    _postmortem_triggers.clear()


def maybe_trigger_postmortem(analysis: dict, event: EventContext) -> None:
    """Spawn a postmortem scenario run for a qualifying analysis verdict.

    Gate order: scenarios module (ADR-011) → switch → known-issue recurrence
    → severity threshold → per-key trigger cooldown.  The scenarios-module
    gate mirrors the route mounting in ``backend/api/router.py``: with
    ``SCENARIOS_ENABLED=false`` the event path must not burn scenario runs
    either (deployment.md documents the same contract for the manual route).
    Fire-and-forget — the analysis task must never block on (or fail with)
    the scenario; execute_scenario records its own outcome in
    ``scenario_runs``.
    """
    if not config.scenarios_active:
        return
    if not config.event_analysis.postmortem_enabled:
        return
    if not analysis.get("is_known_issue"):
        return
    threshold = _SEVERITY_RANK[config.event_analysis.postmortem_min_severity]
    if _SEVERITY_RANK.get(str(analysis.get("severity")), -1) < threshold:
        return

    connector = event.connector
    key = _cooldown_key(
        event.source, event.metadata, connector.event_analysis_cooldown_key
    )
    if not _postmortem_gate(key):
        logger.info(
            "Postmortem trigger suppressed for delivery %s (%s) — recent "
            "trigger active",
            event.delivery_id,
            event.source,
        )
        return

    params = {
        "trigger_event": {
            "source": event.source,
            "content": event.content,
            "memory_id": event.memory_id,
        }
    }

    async def _run() -> None:
        # Module-attribute access (not from-import) so tests can patch
        # ``execute_scenario`` at its source module.
        from backend.runner import scenarios as scenarios_mod

        token = current_thread_id.set(f"event-postmortem-{event.delivery_id}")
        trace_token = current_trace_id.set(f"webhook:{event.delivery_id}")
        try:
            outcome = await scenarios_mod.execute_scenario(
                "postmortem", params=params, trigger="event"
            )
            logger.info(
                "Event-triggered postmortem %s for delivery %s (%s)",
                outcome["run_id"], event.delivery_id, event.source,
            )
            await _notify_postmortem(event, outcome)
        except Exception:
            logger.exception(
                "Event-triggered postmortem failed for delivery %s (%s)",
                event.delivery_id, event.source,
            )
        finally:
            current_thread_id.reset(token)
            current_trace_id.reset(trace_token)

    task = asyncio.create_task(_run())
    # Hold a strong reference — a fire-and-forget task dropped by the GC at
    # its first await would silently lose the postmortem (same pattern as
    # auto-memory capture in nodes.py).
    _postmortem_tasks.add(task)
    task.add_done_callback(_postmortem_tasks.discard)


_postmortem_tasks: set[asyncio.Task] = set()


async def _notify_postmortem(event: EventContext, outcome: dict) -> None:
    """Push the finished draft to Feishu as the unattended-run surface.

    Manual runs are visible in their conversation; event-triggered ones
    would otherwise complete invisibly.  A missing webhook URL is a no-op,
    not an error.
    """
    from backend.service.notification import send_feishu_message

    if not config.feishu_webhook_url:
        return
    source = event.source
    title_entity = event.connector.event_analysis_display.get("title_entity")
    entity = ""
    if title_entity:
        value = event.metadata.get(title_entity)
        if isinstance(value, str) and value.strip():
            entity = f": {value.strip()}"
    title = f"EMA 自动复盘 · {source}{entity}".rstrip(": ")
    status = outcome.get("status")
    lines: list[str] = []
    findings = outcome.get("findings") or {}
    overview = str(findings.get("overview", "")).strip()
    if status != "completed":
        error = outcome.get("error") or "未知原因"
        lines.append(f"复盘草稿生成失败: {error[:200]}")
    elif overview:
        lines.append(f"**概述**: {overview[:300]}")
        root_cause = str(findings.get("root_cause", "")).strip()
        if root_cause:
            lines.append(f"**根因假设**: {root_cause[:300]}")
        lines.append(f"运行记录: {outcome['run_id']}")
        if not outcome.get("contract_ok"):
            lines.append("(结构化契约未通过，草稿以 Markdown 为准)")
    else:
        # Completed but no structured findings — a contract miss, not a
        # failure: the markdown draft exists (spec: 契约未过不判死) and the
        # user should go read it, not read "生成失败".
        lines.append("复盘草稿已生成，但结构化契约未通过——草稿以 Markdown 为准。")
        lines.append(f"运行记录: {outcome['run_id']}")
    ok, detail = await send_feishu_message(
        "\n".join(lines), msg_type="interactive", title=title
    )
    if not ok:
        logger.warning(
            "Postmortem Feishu push failed for delivery %s: %s",
            event.delivery_id, detail,
        )


# ── Trigger entry point (called by the webhook path) ──────────────────


async def maybe_analyze_event(event: EventContext) -> None:
    """Gate and dispatch the analysis run for one delivered event.

    Runs inline within the webhook background task (after the delivery
    reached its terminal state), but acquires its own concurrency slot —
    release happens here regardless of outcome.  Skips are recorded in
    the analysis column so silence is distinguishable from "not enabled".
    """
    if not config.event_analysis.enabled:
        return

    connector = event.connector
    if not connector.triggers_event_analysis:
        return

    if not connector.event_analysis_prompt_key:
        logger.error(
            "Connector %s opted into event analysis but declared no "
            "event_analysis_prompt_key — skipping",
            event.source,
        )
        return

    key = _cooldown_key(
        event.source, event.metadata, connector.event_analysis_cooldown_key
    )
    gate = _cooldown_gate(key)
    if gate == "skipped_cooldown":
        logger.info(
            "Event analysis skipped for delivery %s (%s) — cooldown active for %s",
            event.delivery_id, event.source, key,
        )
        await persist_analysis(event.delivery_id, {"status": "skipped_cooldown"})
        return

    if not _try_acquire_event_slot():
        logger.warning(
            "Event analysis skipped for delivery %s (%s) — concurrency cap "
            "(%d) reached",
            event.delivery_id, event.source, config.event_analysis.max_concurrency,
        )
        # Slot exhaustion is transient load, not a property of the event:
        # don't burn the cooldown window on a run that never happened.
        if key is not None:
            _cooldowns.pop(key, None)
        await persist_analysis(event.delivery_id, {"status": "skipped_concurrency"})
        return

    try:
        await run_event_analysis(event)
    finally:
        _release_event_slot()
