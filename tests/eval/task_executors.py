"""Default executor for the task-level end-to-end eval — drives the agent graph.

The task eval's whole point is measuring the *real agent*: this executor
builds the production ``build_agent_graph`` (real tools, real HITL gates,
real checkpointer) and drives it like the API does, with one difference —
every human gate is *automatically decided* so the eval measures the agent's
capability to complete a task, isolated from the human-in-the-loop decision
(a task "succeeds" here if the agent would do the right thing when a human
always approves).

- **Tool execution** — tools actually run (``ToolNode``), so a task that
  writes a memory or notifies 飞书 really performs the side effect.  Write
  tasks are tagged by their content (see ``task_ground_truth``).
- **HITL** — ``check_approval_node`` (approve writes / notification) and
  ``check_conflict_node`` (resolve memory conflicts) pause via ``interrupt()``;
  the runner resumes each with a policy (default :func:`auto_approve_resume`:
  approve tools, keep the existing memory on conflict, approve every call in
  a batch).  A custom ``resume`` callable lets tests simulate a rejection.
- **Loop discipline** — ``n_steps`` and ``within_budget`` come out of the
  run: did the agent reach the final answer before ``max_steps`` forced
  termination?
- **Trajectory** — every ``AIMessage.tool_calls`` across the run is the
  called-tool list; every ``ToolMessage`` contributes its envelope display
  (what the model saw) to ``context_text`` and its source ids to
  ``source_ids`` for citation / grounding judgement.

Auto-memory is disabled for the run: ``generate_final_node`` would otherwise
schedule background extraction + write after each task (3-7 LLM calls each)
and pollute the eval corpus.

``make_task_runner`` accepts ``tools`` for test injection; unit tests drive
the real graph with fake ``@tool`` functions and a patched
``backend.agent.nodes.get_llm_provider`` (the same convention as
``tests.eval.llm_executors.make_tool_selector``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from backend.agent.tool_envelope import parse_tool_envelope, truncate_tool_content

# query → the agent's trajectory + answer for one task
TaskRunner = Callable[[str], Awaitable["TaskOutcome"]]


@dataclass
class TaskOutcome:
    """Result of one full agent run: the trajectory and what the model saw."""

    answer: str
    tool_calls: list[dict[str, Any]]  # [{"name", "args"}] in call order
    n_steps: int  # call_llm invocations in this run (ReAct loop count)
    within_budget: bool  # False when max_steps or the wall-clock timeout
    #                    # force-terminated the loop before the final answer
    had_error: bool  # provider error / timeout / graph exception
    error: str = ""
    #: Concatenated tool-result display text (what the model actually saw),
    #: truncated per-message the same way the agent truncates before resending.
    context_text: str = ""
    #: Source ids the model saw (memory ids / document ids) — for citation.
    source_ids: list[str] = field(default_factory=list)


def auto_approve_resume(payload: Any) -> dict[str, Any]:
    """Resume policy: approve everything, keep the existing memory on conflict.

    ``check_approval_node`` interrupt payloads are either a single tool
    (``{"tool_name", ...}``) or a batch (``{"type": "batch", "calls": [...]}``);
    ``check_conflict_node`` payloads carry ``{"type": "conflict", ...}``.
    """
    if isinstance(payload, dict) and payload.get("type") == "conflict":
        return {"resolution": "keep_existing"}
    if isinstance(payload, dict) and payload.get("type") == "batch":
        return {
            "calls": [
                {"id": c.get("id"), "approved": True} for c in payload.get("calls", [])
            ]
        }
    return {"approved": True}


def _tool_call_name(call: dict) -> str:
    """Tool name from a raw tool_call dict (LangChain or OpenAI shape)."""
    return str(call.get("name", call.get("function", {}).get("name", "")))


def _tool_call_args(call: dict) -> dict[str, Any]:
    args = call.get("args", call.get("function", {}).get("arguments", {}))
    if isinstance(args, str):
        import json

        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}
    return args if isinstance(args, dict) else {}


def _collect_sources(envelope: dict[str, Any]) -> list[str]:
    """Source ids from a tool envelope: memory ids or chunk document ids."""
    ids: list[str] = []
    for s in envelope.get("sources") or []:
        if not isinstance(s, dict):
            continue
        sid = s.get("id") or s.get("document_id")
        if sid:
            ids.append(str(sid))
    return ids


def make_task_runner(
    tools: list | None = None,
    *,
    max_steps: int | None = None,
    approval_required_tools: frozenset[str] | None = None,
    resume: Callable[[Any], dict[str, Any]] | None = None,
    timeout: float | None = None,
) -> TaskRunner:
    """Executor that runs one task through the real agent graph.

    Args:
        tools: tool roster (default: ``ALL_TOOLS``).  Tests pass fake tools.
        max_steps: ReAct loop budget (default: ``config.max_agent_steps``).
        approval_required_tools: the human-gate set.  Default is the
            production chat set (``CHAT_APPROVAL_TOOLS``) so notification
            tools are gated like a real chat turn — the runner auto-approves.
            Pass ``frozenset()`` to bypass the approval gate entirely.
        resume: interrupt-resume policy (default :func:`auto_approve_resume`).
        timeout: per-task wall-clock budget in seconds (default:
            ``config.agent_timeout``).  A timeout is recorded as
            ``had_error`` with ``error="timeout"``.
    """
    from backend.agent.nodes import CHAT_APPROVAL_TOOLS
    from backend.agent.tools import ALL_TOOLS

    from backend.shared.config import config

    # A task run must not schedule background auto-memory capture: each
    # capture costs 3-7 LLM calls and writes to the eval corpus.  The eval is
    # a standalone process, so disabling it process-wide is safe here.
    config.auto_memory_enabled = False

    roster = list(tools) if tools is not None else ALL_TOOLS
    step_budget = max_steps if max_steps is not None else config.max_agent_steps
    approval_set = (
        approval_required_tools
        if approval_required_tools is not None
        else CHAT_APPROVAL_TOOLS
    )
    resume_policy = resume or auto_approve_resume
    task_timeout = timeout if timeout is not None else config.agent_timeout

    from backend.agent.graph import build_agent_graph

    graph = build_agent_graph(
        tools=roster,
        max_steps=step_budget,
        approval_required_tools=frozenset(approval_set),
    )

    _seq = [0]

    async def _run(query: str) -> TaskOutcome:
        _seq[0] += 1
        thread_id = f"task-eval-{_seq[0]}"
        run_config = {"configurable": {"thread_id": thread_id}}
        initial = {"messages": [HumanMessage(content=query)]}

        state: dict[str, Any] = {}
        try:
            async with asyncio.timeout(task_timeout):
                state = await graph.ainvoke(initial, config=run_config)
                while "__interrupt__" in state:
                    payload = state["__interrupt__"][0].value
                    state = await graph.ainvoke(
                        Command(resume=resume_policy(payload)),
                        config=run_config,
                    )
        except TimeoutError:
            # A wall-clock timeout is a loop-discipline failure too: the run
            # consumed the whole budget window without reaching a final answer,
            # so it did NOT finish within budget (unlike a plain error abort,
            # which is not a budget matter).
            return TaskOutcome(
                answer="",
                tool_calls=[],
                n_steps=int(state.get("step_count") or 0),
                within_budget=False,
                had_error=True,
                error="timeout",
            )
        except Exception as exc:
            return TaskOutcome(
                answer="",
                tool_calls=[],
                n_steps=int(state.get("step_count") or 0),
                within_budget=True,
                had_error=True,
                error=str(exc),
            )

        # ── Harvest the trajectory ──
        tool_calls: list[dict[str, Any]] = []
        context_parts: list[str] = []
        source_ids: list[str] = []
        for m in state.get("messages", []):
            if isinstance(m, AIMessage):
                for tc in getattr(m, "tool_calls", None) or []:
                    tool_calls.append(
                        {"name": _tool_call_name(tc), "args": _tool_call_args(tc)}
                    )
            elif isinstance(m, ToolMessage):
                raw = str(m.content or "")
                display = truncate_tool_content(raw)  # what the model saw
                if display.strip():
                    context_parts.append(display)
                envelope = parse_tool_envelope(raw)
                if envelope is not None:
                    source_ids.extend(_collect_sources(envelope))

        # ── Final answer (mirror the API layer's extraction) ──
        answer = str(state.get("final_response") or "").strip()
        if not answer:
            for m in reversed(state.get("messages", [])):
                if (
                    isinstance(m, AIMessage)
                    and getattr(m, "content", "")
                    and not getattr(m, "tool_calls", None)
                ):
                    answer = str(m.content).strip()
                    break

        n_steps = int(state.get("step_count") or 0)
        return TaskOutcome(
            answer=answer,
            tool_calls=tool_calls,
            n_steps=n_steps,
            within_budget=n_steps < step_budget,
            had_error=bool(state.get("error")),
            error=str(state.get("error") or ""),
            context_text="\n\n".join(context_parts),
            source_ids=source_ids,
        )

    return _run
