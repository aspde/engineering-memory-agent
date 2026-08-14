"""Pytest fixtures for EMA project."""

import os
from collections.abc import AsyncGenerator
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from httpx import ASGITransport, AsyncClient

# Test-DB URL precedence: explicit TEST_DATABASE_URL > derived from the
# project .env (real ema credentials) > ema123 placeholder (matches a dev box
# whose ema password was left at the .env.example default, or a CI postgres
# service whose password is the placeholder).
_PLACEHOLDER_TEST_URL = "postgresql://ema:ema123@127.0.0.1:5432/ema_test"


def _default_test_database_url() -> str:
    """Test-DB URL derived from the project ``.env`` DATABASE_URL.

    A pytest session always talks to the local/CI PostgreSQL on
    127.0.0.1:5432 with the fixed database name ``ema_test``; only the
    ``user:password`` pair is taken from ``.env``.  Hard-coding the ema123
    placeholder here (as this did before) breaks local pytest on any box whose
    ema password differs from the .env.example default — the password lives in
    ``.env``, so the test bootstrap should read it from there instead of
    guessing.  Falls back to the placeholder when no ``.env`` DATABASE_URL
    exists (fresh clone, CI without a .env).
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    dsn = ""
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line.startswith("DATABASE_URL="):
                continue
            dsn = line.split("=", 1)[1].strip().strip("\"'")
            break
    except OSError:
        dsn = ""
    if not dsn:
        return _PLACEHOLDER_TEST_URL
    netloc = urlsplit(dsn).netloc
    userinfo, _, _hostport = netloc.rpartition("@")
    if not userinfo or "@" in userinfo:
        # A raw @ inside the password would make rpartition ambiguous — fall
        # back rather than connect with a mis-split credential.
        return _PLACEHOLDER_TEST_URL
    return f"postgresql://{userinfo}@127.0.0.1:5432/ema_test"


# Set test environment before any imports
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL", _default_test_database_url()
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
