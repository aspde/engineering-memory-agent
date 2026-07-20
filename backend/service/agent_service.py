"""Agent service — manages the compiled agent graph lifecycle.

Provides factory functions that the API layer uses to obtain agent
instances.  Each call to ``get_agent_for_thread()`` creates a fresh
agent with its own ``InMemorySaver`` for thread-isolated state.
"""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from agent.graph import build_agent_graph
from agent.tools import ALL_TOOLS


def get_agent() -> CompiledStateGraph:
    """Return a singleton agent graph (shared InMemorySaver).

    Suitable for development and testing where a single conversation
    thread is sufficient.  For production with multiple concurrent
    users, use ``get_agent_for_thread()`` instead.
    """
    return build_agent_graph(tools=ALL_TOOLS)


def get_agent_for_thread() -> CompiledStateGraph:
    """Return a fresh agent with its own InMemorySaver.

    Each call creates an independent checkpointer, so different
    thread_ids are fully isolated.  In production, replace
    ``InMemorySaver`` with ``PostgresSaver`` for durability.
    """
    return build_agent_graph(tools=ALL_TOOLS)
