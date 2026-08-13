"""Integration test — production checkpoint persistence path.

The unit suite deliberately tests only the InMemorySaver *fallback*
(``test_checkpointer_fallback.py``); the production path — a real psycopg
connection pool, ``AsyncPostgresSaver`` writing LangGraph checkpoints to
PostgreSQL, and recovering them across a process restart — has no test at
all.  This module closes that gap.

What "production path" means here, and what the test proves:

1. ``agent_service._setup_checkpointer()`` builds an ``AsyncPostgresSaver``
   backed by a real psycopg ``AsyncConnectionPool`` — not the fallback.
2. A compiled agent writes checkpoint state for a ``thread_id`` to the
   database during ``ainvoke()``.
3. A *second* saver instance — fresh pool, same database, exactly what a
   restarted process creates — recovers that thread's history
   (``aget_state``) and continues the conversation from it.
4. Thread isolation survives the restart: an untouched ``thread_id`` has no
   recovered history.

Skipped when PostgreSQL is unreachable (e.g. the Docker ``ema-postgres``
container is down, or ``TEST_DATABASE_URL`` points at the wrong
credentials) — the test must not fail on an environment, only on a broken
persistence path.

Windows note: psycopg 3's async driver refuses to run on the default
``ProactorEventLoop``.  The test therefore drives everything through an
explicit ``SelectorEventLoop`` (``asyncio.run(..., loop_factory=...)``)
instead of relying on pytest-asyncio's function-scoped loop, so it runs on
both Windows and Linux CI unchanged.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from backend.service import agent_service as svc
from tests._fake_llm import content_stream, sequential_stream

pytest.importorskip("psycopg_pool")

_THREAD_ID = "integration-persistence-thread"
_OTHER_THREAD_ID = "integration-persistence-other-thread"

# The loop_factory that psycopg 3 needs on Windows (a no-op elsewhere).
_SELECTOR_LOOP = asyncio.SelectorEventLoop


def _db_reachable() -> bool:
    """True when a psycopg async connection to the configured DB succeeds."""
    from backend.shared.config import config

    conninfo = config.database_url.replace(
        "postgresql+asyncpg://", "postgresql://", 1
    )

    async def _probe() -> bool:
        from psycopg import AsyncConnection

        try:
            conn = await AsyncConnection.connect(conninfo)
        except Exception:
            return False
        await conn.close()
        return True

    return asyncio.run(_probe(), loop_factory=_SELECTOR_LOOP)


def test_checkpoint_persistence_across_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    """A conversation survives a process restart via PostgreSQL checkpoints.

    The real production pipeline is exercised end-to-end: a psycopg pool and
    ``AsyncPostgresSaver`` write a conversation, then a second pool + saver
    (the "restarted process") reads the same thread back and keeps going.
    """
    if not _db_reachable():
        pytest.skip("No reachable PostgreSQL — persistence path not exercised")

    # Fake streaming LLM — two plain-chat turns, one per ainvoke below.
    fake = AsyncMock()
    fake.chat_raw_stream = sequential_stream(
        content_stream("reply before restart"),
        content_stream("reply after restart"),
    )
    monkeypatch.setattr("backend.agent.nodes.get_llm_provider", lambda: fake)
    # Auto-memory capture is a fire-and-forget background task that would
    # otherwise spawn its own LLM/gate calls on top of this test.
    monkeypatch.setattr("backend.agent.nodes._schedule_auto_memory", lambda state: None)

    async def _scenario() -> None:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        try:
            # ── First "process": build the production checkpointer ──
            await svc._setup_checkpointer()
            assert isinstance(svc._checkpointer, AsyncPostgresSaver), (
                "production checkpointer must be AsyncPostgresSaver, got "
                f"{type(svc._checkpointer).__name__}"
            )

            agent1 = svc.get_agent()
            result1 = await agent1.ainvoke(
                {"messages": [HumanMessage(content="hello before restart")]},
                config={"configurable": {"thread_id": _THREAD_ID}},
            )
            assert result1["final_response"] == "reply before restart"

            # ── Simulate a process restart: fresh pool, fresh saver ──
            await svc._close_checkpointer()
            svc._checkpointer = None
            svc._pool = None
            await svc._setup_checkpointer()
            assert isinstance(svc._checkpointer, AsyncPostgresSaver)

            # The *new* saver must recover the old thread's history from the DB.
            agent2 = svc.get_agent()
            snap = await agent2.aget_state(
                config={"configurable": {"thread_id": _THREAD_ID}}
            )
            msgs = snap.values.get("messages", [])
            assert [m.content for m in msgs] == [
                "hello before restart",
                "reply before restart",
            ]

            # And continue the conversation on top of the recovered history.
            result2 = await agent2.ainvoke(
                {"messages": [HumanMessage(content="second question after restart")]},
                config={"configurable": {"thread_id": _THREAD_ID}},
            )
            assert result2["final_response"] == "reply after restart"

            # Thread isolation survives the restart too.
            other = await agent2.aget_state(
                config={"configurable": {"thread_id": _OTHER_THREAD_ID}}
            )
            assert not other.values.get("messages")
        finally:
            # Restore the module globals so later tests see a clean state.
            await svc._close_checkpointer()
            svc._checkpointer = None
            svc._pool = None

    asyncio.run(_scenario(), loop_factory=_SELECTOR_LOOP)
