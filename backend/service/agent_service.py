"""Agent service — manages the compiled agent graph lifecycle.

Provides factory functions that the API layer uses to obtain agent
instances.  Uses ``AsyncPostgresSaver`` with a psycopg 3 connection
pool for durable checkpoint persistence across restarts; falls back
to ``InMemorySaver`` when the database is unreachable.

Eagerly initialises the embedding provider singleton on import so
that the offline-mode environment variables take effect before
any SentenceTransformer/transformers network access is attempted.

Notes
-----
On Windows ``AsyncPostgresSaver`` requires ``SelectorEventLoop``
because psycopg 3 is incompatible with ``ProactorEventLoop``.
``backend.main`` sets the event loop before importing this module.
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from agent.graph import build_agent_graph
from agent.tools import ALL_TOOLS
from backend.shared.config import config

logger = logging.getLogger(__name__)

_checkpointer: InMemorySaver | object | None = None
_pool = None  # psycopg AsyncConnectionPool, closed on shutdown


# ── Eager-init embedding provider (ensures offline flags are set) ───
try:
    from backend.service.embedding_service import get_embedding_provider

    get_embedding_provider()
except Exception:
    pass  # will be retried lazily at first tool call


def _get_checkpointer() -> InMemorySaver | object:
    """Return the active checkpointer, or InMemorySaver as fallback."""
    if _checkpointer is not None:
        return _checkpointer

    logger.warning("No checkpointer set — using InMemorySaver")
    return InMemorySaver()


async def _setup_checkpointer() -> None:
    """Initialise the checkpointer singleton with a connection pool.

    Falls back to ``InMemorySaver`` if the database is unreachable.
    """
    global _checkpointer, _pool

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        # config.database_url is postgresql://..., psycopg 3 natively supports it.
        # Remove +asyncpg if present (legacy compatibility).
        conninfo = config.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

        _pool = AsyncConnectionPool(
            conninfo=conninfo,
            open=False,
            min_size=1,
            max_size=5,
        )
        await _pool.open()
        await _pool.wait()

        _checkpointer = AsyncPostgresSaver(_pool)
        await _checkpointer.setup()
        logger.info("AsyncPostgresSaver setup complete")
    except Exception as exc:
        logger.warning(
            "Failed to setup AsyncPostgresSaver (%s) — using InMemorySaver",
            exc,
        )
        _checkpointer = InMemorySaver()


async def _close_checkpointer() -> None:
    """Close the connection pool on shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Checkpoint pool closed")


def get_agent() -> CompiledStateGraph:
    """Return a compiled agent graph with all default tools.

    Uses ``AsyncPostgresSaver`` when the database is reachable;
    falls back to ``InMemorySaver``.  Different ``thread_id`` values
    in ``ainvoke()`` config are fully isolated.
    """
    return build_agent_graph(tools=ALL_TOOLS, checkpointer=_get_checkpointer())


# Alias kept for compatibility — agents are now always per-thread-safe.
get_agent_for_thread = get_agent
