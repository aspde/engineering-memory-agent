"""Agent graph construction — builds the StateGraph and compiles it.

The graph implements a simple ReAct loop:

    START -> call_llm -> (tools?) -> tools -> call_llm (loop)
                       \\-> (!tools) -> generate_final -> END
"""

from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.nodes import call_llm_node, generate_final_node
from agent.state import AgentState


def build_agent_graph(
    tools: list,
    checkpointer: InMemorySaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the EMA agent graph.

    Args:
        tools: List of ``@tool``-decorated async functions.
        checkpointer: Checkpointer for state persistence (default: InMemorySaver).

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``ainvoke()``.
    """
    # Bind tools to the call_llm node at construction time
    _call_llm = partial(call_llm_node, tools=tools)

    builder = StateGraph(AgentState)

    builder.add_node("call_llm", _call_llm)
    builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    builder.add_node("generate_final", generate_final_node)

    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges(
        "call_llm",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "generate_final",
        },
    )
    builder.add_edge("tools", "call_llm")
    builder.add_edge("generate_final", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


# ── Convenience: pre-built with default tools ────────────────────────


def get_default_agent() -> CompiledStateGraph:
    """Return a compiled agent with all default tools and InMemorySaver."""
    from agent.tools import ALL_TOOLS

    return build_agent_graph(tools=ALL_TOOLS)
