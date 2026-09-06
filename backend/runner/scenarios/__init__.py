"""Scenario registry — explicit dict of available vertical scenarios.

Each scenario is a compose function that assembles a specialised system
prompt + user message, calls the agent, and returns formatted results.
Scenarios are pure consumers of the existing infrastructure — no new
tools, no changes to the agent graph.  Runs are persisted to the
``scenario_runs`` table (migration 0005) via :func:`execute_scenario`,
the single execution path shared by the manual API route and the
event-trigger gate.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.shared.config import config
from backend.shared.slots import SlotLimiter

logger = logging.getLogger(__name__)

# Context variable to pass the scenario thread_id from the API layer into
# invoke_scenario_agent without threading it through every compose function.
scenario_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "scenario_thread_id", default=""
)


# ── Shared agent-invocation helper ─────────────────────────────────────


async def invoke_scenario_agent(
    system_prompt: str,
    user_message: str,
) -> str:
    """Invoke the default agent with a scenario-specific prompt.

    All four compose functions delegate to this helper so the agent-
    invocation boilerplate (build messages, call ``ainvoke()``, extract
    ``final_response``, error handling) lives in one place.

    Uses ``scenario_thread_id`` ContextVar when set (from the API layer),
    falling back to a generated UUID for background / patrol-triggered runs.
    """

    from backend.runner.agent_service import get_agent

    # Scenario runs are unattended (manual trigger / scheduled scan) — no
    # human can approve a paused write/ingest call, so pass an empty approval
    # set.  The conflict HITL gate still pauses on a write conflict; that
    # interrupt is surfaced below instead of being swallowed.  The ReAct
    # budget uses config.scenario_max_steps (12) rather than the interactive
    # 5 — a scenario compose chains several retrieval rounds before writing
    # the report, and force-terminating mid-search degrades the final answer
    # to the last tool envelope (verified live 2026-09-05, run 291807b5).
    agent = get_agent(
        approval_required_tools=frozenset(),
        max_steps=config.scenario_max_steps,
    )
    # Recursion limit counts graph-node executions (~4 per ReAct step, plus
    # the final node): steps × 4 + headroom, comfortably under 50 for the
    # default budget; scale with the configured budget so a raised
    # SCENARIO_MAX_STEPS doesn't trip GraphRecursionError.
    recursion_limit = max(50, config.scenario_max_steps * 4 + 30)
    tid = scenario_thread_id.get() or f"scenario-{uuid.uuid4()}"
    # Mirror the thread into the shared current_thread_id contextvar: the
    # synthesis path (generate_final_node) reads it to recognise scenario
    # threads and apply report-mode instructions + the SCENARIO_MAX_TOKENS
    # output ceiling.  Without this the synthesis ran with the interactive
    # 4096 budget and truncated mid-report (verified 2026-09-05, runs
    # 6c9945da / c1ff35bf — agent_final output_tokens pinned at 4096,
    # response degraded to a raw tool envelope).
    from backend.shared.config import current_thread_id as shared_thread_id

    thread_token = shared_thread_id.set(tid)
    try:
        result = await agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message),
                ]
            },
            config={
                "configurable": {"thread_id": tid},
                "recursion_limit": recursion_limit,
            },
        )
    except Exception as exc:
        logger.exception("Scenario agent invocation failed")
        return f"场景执行失败: {exc}"
    finally:
        shared_thread_id.reset(thread_token)

    # A HITL gate paused the run — ``ainvoke`` returns normally with
    # ``__interrupt__`` set, it is not an exception.  Without this check the
    # pause fell through to the last-message fallback and the scenario
    # reported a fabricated result.  Surface it truthfully instead.
    interrupts = result.get("__interrupt__")
    if interrupts:
        payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        logger.warning(
            "Scenario interrupted for human review (thread=%s): %r", tid, payload,
        )
        return f"场景执行被中断，等待人工审批: {payload}"

    final = result.get("final_response", "") or ""
    if not final:
        for m in reversed(result.get("messages", [])):
            if (
                hasattr(m, "content")
                and m.content
                and not getattr(m, "tool_calls", None)
            ):
                final = str(m.content)
                break

    return final or "(Agent 未返回内容)"


# ── Scenario-run concurrency cap ───────────────────────────────────────
# Each scenario run invokes the full agent (recursion_limit=50) for its
# whole compose chain — up to SCENARIO_TIMEOUT_SECONDS — so an unbounded
# number of concurrent scenarios would together saturate the provider rate
# limit and can only be stopped by a restart.  Same plain-counter approach
# as the interactive-agent cap in ``agent_service`` — event-loop-agnostic,
# safe across pytest's function-scoped event loops.
_scenario_slots = SlotLimiter(lambda: config.max_scenario_concurrency)


def _try_acquire_scenario_slot() -> bool:
    """Reserve one in-flight scenario run; False when the cap is hit."""
    return _scenario_slots.try_acquire()


def _release_scenario_slot() -> None:
    """Release a scenario slot acquired by :func:`_try_acquire_scenario_slot`."""
    _scenario_slots.release()


# ── Shared execution path (manual route + event trigger) ──────────────


class ScenarioBusyError(Exception):
    """Raised when the scenario concurrency cap is reached."""


class ScenarioErrorKind:
    """Typed error classification recorded on failed scenario runs.

    The manual route maps these onto HTTP statuses (504/422) — matching on
    a code instead of substring-matching the human-readable ``error`` text,
    which changes with every rewording.
    """

    TIMEOUT = "timeout"
    INVALID_PARAMS = "invalid_params"
    CANCELLED = "cancelled"
    EXECUTION = "execution"


def _dump_json(value: Any) -> str | None:
    """Serialise *value* for a JSONB parameter; None stays NULL."""
    if value is None:
        return None
    import json

    return json.dumps(value, ensure_ascii=False)


# Per-scenario contract parsers: (result_md) → (findings | None, contract_ok).
# Only scenarios that instruct the model to emit a trailing JSON block are
# listed — the others persist findings=NULL, contract_ok=NULL.
def _make_contract_parser(required_keys: set[str]):
    """Build a contract parser for one scenario's required-key set.

    All four scenarios share the same shape: `extract_json_object` tolerates
    fenced/wrapped JSON, the `raw_output` wrapper counts as a miss, and a
    missing key counts as a miss — findings stay NULL either way (the
    markdown still reaches the user).
    """
    from backend.service.json_extraction import extract_json_object

    def _parse(result_text: str) -> tuple[dict[str, Any] | None, bool]:
        findings = extract_json_object(result_text or "")
        # extract_json_object degrades unparseable output to
        # {"raw_output": ...}; treat that wrapper as a contract miss rather
        # than real findings.
        if not isinstance(findings, dict) or "raw_output" in findings:
            return None, False
        missing = required_keys - set(findings.keys())
        if missing:
            return None, False
        return findings, True

    return _parse


_CONTRACT_PARSERS: dict[str, Any] = {
    "postmortem": _make_contract_parser({
        "overview",
        "timeline",
        "similar_incidents",
        "root_cause",
        "recommendations",
        "related_entities",
    }),
    "code_review": _make_contract_parser({
        "pr_overview",
        "risk_files",
        "decision_conflicts",
        "review_points",
    }),
    "onboarding": _make_contract_parser({
        "project_overview",
        "core_modules",
        "reading_order",
        "key_decisions",
        "incident_patterns",
    }),
    "tech_debt": _make_contract_parser({
        "summary",
        "unresolved_workarounds",
        "doc_gaps",
        "auto_resolved",
        "priorities",
    }),
}


async def execute_scenario(
    name: str,
    params: dict[str, Any] | None = None,
    trigger: str = "manual",
    thread_id: str = "",
) -> dict[str, Any]:
    """Run one scenario end-to-end and persist it to ``scenario_runs``.

    Single execution path for both callers — the manual API route maps
    the typed errors onto HTTP statuses, while the event-trigger gate
    treats them as log-and-skip.  Steps: acquire a concurrency slot,
    insert a ``running`` row, dynamically import and await the compose
    function under the scenario deadline, parse the postmortem JSON
    contract when present, then update the row to its terminal state.

    Returns a dict with keys ``run_id``, ``status``, ``result``,
    ``findings`` and ``contract_ok``.  Raises :class:`ScenarioBusyError`
    before anything is written; every other failure is recorded on the
    row and reported through ``status='failed'`` rather than raised.
    """
    from sqlalchemy import text

    from backend.db import get_session_factory

    scenario = SCENARIOS.get(name)
    if scenario is None or scenario.get("status") == "inactive":
        raise KeyError(name)

    params = params or {}
    if not _try_acquire_scenario_slot():
        logger.warning(
            "execute_scenario refused — concurrency cap reached (max=%d) scenario=%s",
            config.max_scenario_concurrency, name,
        )
        raise ScenarioBusyError(name)

    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            result = await session.execute(
                text(
                    "INSERT INTO scenario_runs (scenario_key, trigger, status, params) "
                    "VALUES (:key, :trigger, 'running', CAST(:params AS JSONB)) "
                    "RETURNING id"
                ),
                {"key": name, "trigger": trigger, "params": _dump_json(params)},
            )
            run_id = str(result.scalar_one())
            await session.commit()

        if thread_id:
            # Persist the client thread_id through the compose chain via
            # ContextVar so invoke_scenario_agent reuses the conversation.
            scenario_thread_id.set(thread_id)
        elif not scenario_thread_id.get():
            scenario_thread_id.set(f"scenario-run-{run_id}")

        status = "completed"
        error_msg: str | None = None
        error_kind: str | None = None
        started_at = datetime.now(UTC)
        result_text = ""
        cancelled = False
        try:
            module_path, func_name = scenario["compose"].rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_path)
            compose_func = getattr(module, func_name)

            async with asyncio.timeout(config.scenario_timeout):
                result_text = await compose_func(**params)
        except TimeoutError:
            logger.warning(
                "Scenario '%s' run %s timed out after %ds",
                name, run_id, config.scenario_timeout,
            )
            status = "failed"
            error_msg = f"timed out after {config.scenario_timeout}s"
            error_kind = ScenarioErrorKind.TIMEOUT
        except TypeError as exc:
            logger.warning("Scenario '%s' rejected params: %s", name, exc)
            status = "failed"
            error_msg = f"invalid parameters: {exc}"
            error_kind = ScenarioErrorKind.INVALID_PARAMS
        except asyncio.CancelledError:
            # Task cancelled (shutdown).  Persist the terminal state below,
            # then re-raise so the task ends cancelled — same contract as
            # run_patrol: a stuck 'running' row would mislead readers.
            status = "failed"
            error_msg = "cancelled mid-run"
            error_kind = ScenarioErrorKind.CANCELLED
            cancelled = True
        except Exception as exc:
            logger.exception("Scenario '%s' execution failed", name)
            status = "failed"
            error_msg = f"{type(exc).__name__}: {exc}"
            error_kind = ScenarioErrorKind.EXECUTION
            result_text = ""

        findings: dict[str, Any] | None = None
        contract_ok: bool | None = None
        parser = _CONTRACT_PARSERS.get(name)
        if parser is not None and status == "completed":
            findings, contract_ok = parser(result_text)

        completed_at = datetime.now(UTC)
        async with session_factory() as session:
            await session.execute(
                text(
                    "UPDATE scenario_runs "
                    "SET status = :status, result_md = :result_md, "
                    "findings = CAST(:findings AS JSONB), contract_ok = :contract_ok, "
                    "error = :error, completed_at = :completed_at "
                    "WHERE id = :id"
                ),
                {
                    "id": run_id,
                    "status": status,
                    "result_md": result_text,
                    "findings": _dump_json(findings),
                    "contract_ok": contract_ok,
                    "error": error_msg,
                    "completed_at": completed_at,
                },
            )
            await session.commit()

        logger.info(
            "Scenario '%s' run %s %s (%.1fs, trigger=%s)",
            name, run_id, status,
            (completed_at - started_at).total_seconds(),
            trigger,
        )
        if cancelled:
            raise asyncio.CancelledError()
        return {
            "run_id": run_id,
            "status": status,
            "result": result_text,
            "findings": findings,
            "contract_ok": contract_ok,
            "error": error_msg,
            "error_kind": error_kind,
        }
    finally:
        _release_scenario_slot()


# ── Typed scenario descriptor ──────────────────────────────────────────


@dataclass
class ScenarioMeta:
    """Typed descriptor for a vertical scenario."""

    key: str
    name: str
    description: str = ""
    compose: str = ""
    triggers: list[str] = field(default_factory=lambda: ["manual"])
    status: str = "active"  # active | beta | inactive


# ── Registry ───────────────────────────────────────────────────────────

SCENARIOS: dict[str, dict[str, Any]] = {
    "postmortem": {
        "name": "故障复盘",
        "description": "从故障记录自动生成复盘草稿，包含时间线、相似故障匹配和根因分析",
        "compose": "backend.runner.scenarios.postmortem.compose_postmortem",
        "triggers": ["manual"],
        "status": "active",
    },
    "code_review": {
        "name": "代码审查助手",
        "description": "分析 PR 变更文件，标记高风险代码和历史故障关联，检查与已有决策的一致性",
        "compose": "backend.runner.scenarios.code_review.compose_review_context",
        "triggers": ["manual"],
        "status": "active",
    },
    "onboarding": {
        "name": "新人 Onboarding",
        "description": "生成项目结构化概览、推荐阅读顺序和决策溯源",
        "compose": "backend.runner.scenarios.onboarding.compose_onboarding_guide",
        "triggers": ["manual"],
        "status": "active",
    },
    "tech_debt": {
        "name": "技术债雷达",
        "description": "扫描未解决的临时方案、文档缺口，自动检测已解决的 workaround",
        "compose": "backend.runner.scenarios.tech_debt.compose_tech_debt_report",
        "triggers": ["weekly_patrol", "manual"],
        "status": "active",
    },
}


def visible_scenarios(include_beta: bool = False) -> dict[str, dict[str, Any]]:
    """Return scenarios that should be exposed in the UI.

    Args:
        include_beta: When True, also include beta-status scenarios.
    """
    valid = {"active"}
    if include_beta:
        valid.add("beta")
    return {
        k: v for k, v in SCENARIOS.items()
        if v.get("status", "inactive") in valid
    }
