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
    call_llm_node,
    check_approval_node,
    check_conflict_node,
    generate_final_node,
)
from agent.state import AgentState


def _route_after_approval(state: AgentState) -> str:
    """Route to tools after check_approval pass-through."""
    return "tools"


def build_agent_graph(
    tools: list,
    checkpointer: object | None = None,
) -> CompiledStateGraph:
    """Build and compile the EMA agent graph.

    Args:
        tools: List of ``@tool``-decorated async functions.
        checkpointer: Checkpointer for state persistence
            (InMemorySaver, PostgresSaver, etc.).  Defaults to InMemorySaver.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``ainvoke()``.
    """
    _call_llm = partial(call_llm_node, tools=tools)

    builder = StateGraph(AgentState)

    builder.add_node("call_llm", _call_llm)
    builder.add_node("check_approval", check_approval_node)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_node("check_conflict", check_conflict_node)
    builder.add_node("generate_final", generate_final_node)

    builder.add_edge(START, "call_llm")

    # call_llm → if tool_calls → check_approval (gate) else → generate_final
    builder.add_conditional_edges(
        "call_llm",
        tools_condition,
        {
            "tools": "check_approval",
            "__end__": "generate_final",
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
