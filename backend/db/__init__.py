"""Async database engine and session management.

Provides a single async engine and session factory using asyncpg + SQLAlchemy.
Uses the DATABASE_URL from config — no hardcoded connection strings.
"""

from __future__ import annotations

import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.shared.config import config

# Convert postgresql:// to postgresql+asyncpg:// for async driver
_sync_url = config.database_url
if _sync_url.startswith("postgresql://"):
    _async_url = _sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _sync_url.startswith("postgresql+asyncpg://"):
    _async_url = _sync_url
else:
    _async_url = _sync_url

# In test mode, use NullPool so every connection is created fresh and
# destroyed immediately.  This avoids the classic Windows ProactorEventLoop +
# asyncpg issue where pooled connections survive across event-loop
# boundaries and crash with "Event loop is closed".
_is_test = os.environ.get("APP_ENV") == "test"
_kwargs: dict = {}
if _is_test:
    _kwargs["poolclass"] = NullPool
else:
    _kwargs.update(pool_size=5, max_overflow=10, pool_pre_ping=True)

engine = create_async_engine(_async_url, echo=False, **_kwargs)
_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory (reuse across calls)."""
    return _session_factory


def get_engine():
    """Return the current async engine.

    Always use this instead of importing ``engine`` directly — the engine
    may be recreated by ``close_db()`` between tests, and a direct import
    would hold a stale reference.
    """
    return engine


async def close_db() -> None:
    """Dispose the engine and re-create it.

    On Windows (ProactorEventLoop), asyncpg connections are bound to the
    event loop that created them.  Disposing the engine between tests and
    re-creating a fresh one ensures the next test gets a clean connection
    pool with no stale connections from a closed loop.
    """
    global engine, _session_factory
    await engine.dispose()
    engine = create_async_engine(_async_url, echo=False, **_kwargs)
    _session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
