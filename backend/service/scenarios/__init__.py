"""Scenario registry — explicit dict of available vertical scenarios.

Each scenario is a compose function that assembles a specialised system
prompt + user message, calls the agent, and returns formatted results.
Scenarios are pure consumers of the existing infrastructure — no new
tools, no new tables, no changes to the agent graph.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from backend.shared.config import config

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
    import uuid

    from backend.service.agent_service import get_agent

    # Scenario runs are unattended (manual trigger / scheduled scan) — no
    # human can approve a paused write/ingest call, so pass an empty approval
    # set.  The conflict HITL gate still pauses on a write conflict; that
    # interrupt is surfaced below instead of being swallowed.
    agent = get_agent(approval_required_tools=frozenset())
    tid = scenario_thread_id.get() or f"scenario-{uuid.uuid4()}"
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
                "recursion_limit": 50,
            },
        )
    except Exception as exc:
        logger.exception("Scenario agent invocation failed")
        return f"场景执行失败: {exc}"

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
# limit and can only be stopped by a restart.  Backed by a plain counter
# (not an ``asyncio.Semaphore``) so it stays event-loop-agnostic like the
# interactive-agent cap in ``agent_service`` — safe across pytest's
# function-scoped event loops.
_scenario_active = 0
_scenario_slots_lock = threading.Lock()


def _try_acquire_scenario_slot() -> bool:
    """Reserve one in-flight scenario run; False when the cap is hit."""
    global _scenario_active
    with _scenario_slots_lock:
        if _scenario_active >= config.max_scenario_concurrency:
            return False
        _scenario_active += 1
        return True


def _release_scenario_slot() -> None:
    """Release a scenario slot acquired by :func:`_try_acquire_scenario_slot`."""
    global _scenario_active
    with _scenario_slots_lock:
        _scenario_active = max(_scenario_active - 1, 0)


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
        "compose": "backend.service.scenarios.postmortem.compose_postmortem",
        "triggers": ["manual"],
        "status": "active",
    },
    "code_review": {
        "name": "代码审查助手",
        "description": "分析 PR 变更文件，标记高风险代码和历史故障关联，检查与已有决策的一致性",
        "compose": "backend.service.scenarios.code_review.compose_review_context",
        "triggers": ["manual"],
        "status": "active",
    },
    "onboarding": {
        "name": "新人 Onboarding",
        "description": "生成项目结构化概览、推荐阅读顺序和决策溯源",
        "compose": "backend.service.scenarios.onboarding.compose_onboarding_guide",
        "triggers": ["manual"],
        "status": "active",
    },
    "tech_debt": {
        "name": "技术债雷达",
        "description": "扫描未解决的临时方案、文档缺口，自动检测已解决的 workaround",
        "compose": "backend.service.scenarios.tech_debt.compose_tech_debt_report",
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
