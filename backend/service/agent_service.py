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

# ── Tool filtering based on MEMORY_ENABLED ──────────────────────────


def _active_tools() -> list:
    """Return the tool list appropriate for the current config.

    When ``MEMORY_ENABLED=false`` the agent runs without any memory tools —
    a lightweight pure-chat mode suitable for quick questions where the
    memory pipeline overhead isn't warranted.
    """
    if config.memory_enabled:
        return ALL_TOOLS
    logger.info("MEMORY_ENABLED=false — agent running in chat-only mode (no tools)")
    return []

_checkpointer: InMemorySaver | object | None = None
_pool = None  # psycopg AsyncConnectionPool, closed on shutdown


# ── Trigger offline flags before SentenceTransformer sees them ──────
# Importing embedding_service sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
# at module level; the expensive model load is deferred to first use.
try:
    import backend.service.embedding_service  # noqa: F401
except Exception:
    pass


def _get_checkpointer() -> InMemorySaver | object:
    """Return the active checkpointer, or InMemorySaver as fallback."""
    global _checkpointer
    if _checkpointer is not None:
        return _checkpointer

    logger.warning("No checkpointer set — using InMemorySaver")
    _checkpointer = InMemorySaver()
    return _checkpointer


async def _setup_checkpointer() -> None:
    """Initialise the checkpointer singleton with a connection pool.

    Falls back to ``InMemorySaver`` if the database is unreachable.
    """
    global _checkpointer, _pool

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

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
        async with _pool.connection() as conn:
            await conn.set_autocommit(True)
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoint_migrations (v INTEGER PRIMARY KEY)"
            )
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints ("
                "  thread_id TEXT NOT NULL,"
                "  checkpoint_ns TEXT NOT NULL DEFAULT '',"
                "  checkpoint_id TEXT NOT NULL,"
                "  parent_checkpoint_id TEXT,"
                "  type TEXT,"
                "  checkpoint JSONB NOT NULL,"
                "  metadata JSONB NOT NULL DEFAULT '{}',"
                "  PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id))"
            )
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoint_blobs ("
                "  thread_id TEXT NOT NULL,"
                "  checkpoint_ns TEXT NOT NULL DEFAULT '',"
                "  channel TEXT NOT NULL,"
                "  version TEXT NOT NULL,"
                "  type TEXT NOT NULL,"
                "  blob BYTEA,"
                "  PRIMARY KEY (thread_id, checkpoint_ns, channel, version))"
            )
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS checkpoint_writes ("
                "  thread_id TEXT NOT NULL,"
                "  checkpoint_ns TEXT NOT NULL DEFAULT '',"
                "  checkpoint_id TEXT NOT NULL,"
                "  task_id TEXT NOT NULL,"
                "  idx INTEGER NOT NULL,"
                "  channel TEXT NOT NULL,"
                "  type TEXT,"
                "  blob BYTEA NOT NULL,"
                "  task_path TEXT NOT NULL DEFAULT '',"
                "  PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx))"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx "
                "ON checkpoints(thread_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS checkpoint_blobs_thread_id_idx "
                "ON checkpoint_blobs(thread_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx "
                "ON checkpoint_writes(thread_id)"
            )
            for v in range(10):
                await conn.execute(
                    "INSERT INTO checkpoint_migrations (v) VALUES (%s) ON CONFLICT DO NOTHING",
                    [v],
                )
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


def get_agent(
    approval_required_tools: frozenset[str] | None = None,
) -> CompiledStateGraph:
    """Return a compiled agent graph with active tools per config.

    Uses ``AsyncPostgresSaver`` when the database is reachable;
    falls back to ``InMemorySaver``.  Different ``thread_id`` values
    in ``ainvoke()`` config are fully isolated.

    Tool selection is controlled by ``MEMORY_ENABLED`` — when ``false``
    the agent runs without memory tools (pure chat mode).

    ``approval_required_tools`` overrides the human-approval gate set
    (default ``APPROVAL_REQUIRED_TOOLS``); the interactive chat routes pass
    ``CHAT_APPROVAL_TOOLS`` so the notification tool also requires approval,
    while automated patrol/scenario runs keep the default and notify
    autonomously.
    """
    return build_agent_graph(
        tools=_active_tools(),
        checkpointer=_get_checkpointer(),
        max_steps=config.max_agent_steps,
        approval_required_tools=approval_required_tools,
    )


# Alias kept for compatibility — agents are now always per-thread-safe.
get_agent_for_thread = get_agent
