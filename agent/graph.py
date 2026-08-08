"""Agent graph construction — builds the StateGraph and compiles it.

The graph implements a ReAct loop with two human-in-the-loop gates:

    START -> call_llm -> (tools?) -> check_approval -> tools
                                                       |
                                                       v
                                              check_conflict
                                                       |
                                                       v
                                                    call_llm (loop)
                       \\-> (!tools) -> generate_final -> END

check_approval pauses *before* sensitive tool execution (write / ingest)
and uses ``Command(goto=...)`` to route approved/rejected paths.

check_conflict pauses *after* write_memory_tool when it returns a conflict
and similarly uses ``Command(goto=...)``.  The edges declared here are
only exercised for the non-interrupt pass-through cases; on interrupt the
``Command`` returned by the node takes precedence.
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import (
    APPROVAL_REQUIRED_TOOLS,
    call_llm_node,
    check_approval_node,
    check_conflict_node,
    generate_final_node,
)
from agent.state import AgentState


def _make_route_after_call_llm(max_steps: int):
    """Return a routing function that respects *tools_condition* but
    force-terminates to ``generate_final`` after *max_steps* iterations."""

    def _route_after_call_llm(state: AgentState) -> str:
        steps = state.get("step_count", 0) or 0
        if steps >= max_steps:
            return "generate_final"
        return tools_condition(state)

    return _route_after_call_llm


def _route_after_approval(state: AgentState) -> str:
    """Route to tools after check_approval pass-through."""
    return "tools"


def build_agent_graph(
    tools: list,
    checkpointer: object | None = None,
    max_steps: int = 5,
    approval_required_tools: frozenset[str] = APPROVAL_REQUIRED_TOOLS,
) -> CompiledStateGraph:
    """Build and compile the EMA agent graph.

    Args:
        tools: List of ``@tool``-decorated async functions.
        checkpointer: Checkpointer for state persistence
            (InMemorySaver, PostgresSaver, etc.).  Defaults to InMemorySaver.
        max_steps: Maximum ReAct loop iterations before the graph
            forces a final answer.  Defaults to 5.
        approval_required_tools: Tool names that pause for human approval
            before execution.  Defaults to ``APPROVAL_REQUIRED_TOOLS`` (the
            write/ingest set); the interactive chat path passes
            ``CHAT_APPROVAL_TOOLS`` to also gate the notification tool.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``ainvoke()``.
    """
    _call_llm = partial(call_llm_node, tools=tools)
    _check_approval = partial(
        check_approval_node,
        approval_required_tools=approval_required_tools,
    )

    builder = StateGraph(AgentState)

    builder.add_node("call_llm", _call_llm)
    builder.add_node("check_approval", _check_approval)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_node("check_conflict", check_conflict_node)
    builder.add_node("generate_final", generate_final_node)

    builder.add_edge(START, "call_llm")

    # call_llm → if tool_calls AND step_count < max_steps → check_approval
    #            else → generate_final
    builder.add_conditional_edges(
        "call_llm",
        _make_route_after_call_llm(max_steps),
        {
            "tools": "check_approval",
            "__end__": "generate_final",
            "generate_final": "generate_final",
        },
    )

    # check_approval → tools (approved / pass-through)
    # Rejected cases use Command(goto="call_llm") — skips this edge.
    builder.add_conditional_edges(
        "check_approval",
        _route_after_approval,
        {"tools": "tools"},
    )

    # tools → check_conflict → call_llm (all routing via Command.goto)
    builder.add_edge("tools", "check_conflict")
    builder.add_edge("check_conflict", "call_llm")

    builder.add_edge("generate_final", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


# ── Convenience: pre-built with default tools ────────────────────────


def get_default_agent() -> CompiledStateGraph:
    """Return a compiled agent with all default tools and InMemorySaver."""
    from agent.tools import ALL_TOOLS

    return build_agent_graph(tools=ALL_TOOLS)
