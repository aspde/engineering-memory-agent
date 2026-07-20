"""Async database engine and session management.

Provides a single async engine and session factory using asyncpg + SQLAlchemy.
Uses the DATABASE_URL from config — no hardcoded connection strings.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.shared.config import config

# Convert postgresql:// to postgresql+asyncpg:// for async driver
_sync_url = config.database_url
if _sync_url.startswith("postgresql://"):
    _async_url = _sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _sync_url.startswith("postgresql+asyncpg://"):
    _async_url = _sync_url
else:
    _async_url = _sync_url

_engine = create_async_engine(_async_url, echo=False, pool_size=5, max_overflow=10, pool_pre_ping=True)
_session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the session factory (reuse across calls)."""
    return _session_factory


async def close_db() -> None:
    """Dispose the engine. Call on app shutdown."""
    await _engine.dispose()
