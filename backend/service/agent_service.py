"""Agent service — manages the compiled agent graph lifecycle.

Provides factory functions that the API layer uses to obtain agent
instances.  Each call to ``get_agent_for_thread()`` creates a fresh
agent with its own ``InMemorySaver`` for thread-isolated state.

Eagerly initialises the embedding provider singleton on import so
that the offline-mode environment variables take effect before
any SentenceTransformer/transformers network access is attempted.
"""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from agent.graph import build_agent_graph
from agent.tools import ALL_TOOLS

# ── Eager-init embedding provider (ensures offline flags are set) ───
try:
    from backend.service.embedding_service import get_embedding_provider

    get_embedding_provider()
except Exception:
    pass  # will be retried lazily at first tool call


def get_agent() -> CompiledStateGraph:
    """Return a compiled agent graph with all default tools.

    Each call returns a *new* graph instance with its own
    ``InMemorySaver`` — callers should cache the result if they
    need a shared instance.  For production with multiple
    concurrent users, use ``get_agent_for_thread()``.
    """
    return build_agent_graph(tools=ALL_TOOLS)


def get_agent_for_thread() -> CompiledStateGraph:
    """Return a fresh agent with its own InMemorySaver.

    Each call creates an independent checkpointer, so different
    thread_ids are fully isolated.  In production, replace
    ``InMemorySaver`` with ``PostgresSaver`` for durability.
    """
    return build_agent_graph(tools=ALL_TOOLS)
