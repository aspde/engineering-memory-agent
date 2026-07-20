"""Agent service — manages the compiled agent graph lifecycle.

Provides factory functions that the API layer uses to obtain agent
instances.  Uses ``AsyncPostgresSaver`` for durable checkpoint
persistence across restarts; falls back to ``InMemorySaver`` when
the database is unreachable.

Eagerly initialises the embedding provider singleton on import so
that the offline-mode environment variables take effect before
any SentenceTransformer/transformers network access is attempted.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agent.graph import build_agent_graph
from agent.tools import ALL_TOOLS
from backend.shared.config import config

logger = logging.getLogger(__name__)

_checkpointer: object | None = None

# ── Eager-init embedding provider (ensures offline flags are set) ───
try:
    from backend.service.embedding_service import get_embedding_provider

    get_embedding_provider()
except Exception:
    pass  # will be retried lazily at first tool call


def _get_checkpointer() -> InMemorySaver | object:
    """Return AsyncPostgresSaver singleton, or InMemorySaver as fallback."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async_url = config.database_url.replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
        _checkpointer = AsyncPostgresSaver.from_conn_string(async_url)
        logger.info("Using AsyncPostgresSaver for checkpoint persistence")
    except Exception:
        logger.info(
            "AsyncPostgresSaver unavailable, falling back to InMemorySaver"
        )
        _checkpointer = InMemorySaver()

    return _checkpointer


async def _setup_checkpointer() -> None:
    """Create checkpoint tables lazily — safe to call more than once."""
    checkpointer = _get_checkpointer()
    if hasattr(checkpointer, "setup"):
        await checkpointer.setup()  # type: ignore[union-attr]


def get_agent() -> CompiledStateGraph:
    """Return a compiled agent graph with all default tools.

    Uses PostgresSaver when the database is available, InMemorySaver
    otherwise.
    """
    return build_agent_graph(tools=ALL_TOOLS, checkpointer=_get_checkpointer())


def get_agent_for_thread() -> CompiledStateGraph:
    """Return a compiled agent with PostgresSaver for durable state.

    Different ``thread_id`` values in ``ainvoke()`` config are fully
    isolated.
    """
    return build_agent_graph(tools=ALL_TOOLS, checkpointer=_get_checkpointer())
