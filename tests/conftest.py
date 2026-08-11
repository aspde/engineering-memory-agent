"""Pytest fixtures for EMA project."""

import os
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

# Set test environment before any imports
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://ema:ema123@localhost:5432/ema_test"
)

# Force offline before SentenceTransformer imports are triggered
# during collection of any test module.  Must be in conftest because
# pytest imports this before any test module, and test collection can
# trigger chain-imports that reach SentenceTransformer.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _noop_conversation_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from writing to the ``conversations`` table.

    The API tests use ``ASGITransport`` which exercises real route
    handlers.  Without this fixture every ``POST /api/agent/chat``
    would call ``_upsert_conversation()`` and pollute the production
    database with test ``thread_id`` values.
    """

    async def _noop(*args: object, **kwargs: object) -> None:
        pass

    monkeypatch.setattr(
        "backend.api.routes.agent_routes._upsert_conversation",
        _noop,
    )


@pytest.fixture(autouse=True)
def _reset_auto_memory_throttle() -> None:
    """Reset auto-memory throttle state so tests never throttle each other.

    Auto-memory captures are rate-limited in-process (see
    ``backend.agent.nodes._auto_memory_throttled``); without a reset, a
    write in one test would suppress the next within the minimum interval.
    """
    from tests.support.process_state import reset_auto_memory_throttle

    reset_auto_memory_throttle()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Reset the test database once per pytest run (hermetic tests).

    Drops and recreates the ``public`` schema, then recreates every table
    via ``init_db()``.  Without this, API tests that write to the real
    ``ema_test`` database accumulate rows across runs — a previous run's
    inserts leak into the next, making tests order- and history-dependent.

    The ``ema_test`` database itself is created if missing (fresh CI
    services only provision ``ema_dev``, e.g. via POSTGRES_DB), so a
    one-liner ``pytest`` works everywhere.

    Runs before collection, so a fresh DB is guaranteed for the whole
    session regardless of which module imports what first.
    """
    del session  # unused
    import asyncio
    from urllib.parse import urlsplit, urlunsplit

    from sqlalchemy import text

    from backend.db import get_engine
    from backend.db.schema import init_db
    from backend.shared.config import config

    async def _ensure_test_db() -> None:
        """Create the test database if it does not exist."""
        import asyncpg

        parts = urlsplit(config.database_url)
        dbname = parts.path.lstrip("/")
        if not dbname or dbname == "postgres":
            return  # already the maintenance DB — nothing to create
        admin_url = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))
        conn = await asyncpg.connect(admin_url)
        try:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1", dbname
            )
            if not exists:
                await conn.execute(f'CREATE DATABASE "{dbname}"')
        finally:
            await conn.close()

    async def _reset() -> None:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
        await init_db()

    # asyncio.run uses a fresh event loop; the NullPool in test mode closes
    # every connection before returning, so no connection survives into the
    # pytest-asyncio function-scoped loops below.
    asyncio.run(_ensure_test_db())
    asyncio.run(_reset())


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient]:
    """Create an async HTTP client for testing FastAPI endpoints.

    Import the FastAPI app lazily so APP_ENV is set first.
    """
    from backend.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
