"""Alembic migration environment for EMA.

The database URL comes from the app's own config
(``backend.shared.config.config.database_url``), so there is one source of
truth and no duplicated credentials.  EMA's schema is managed with raw SQL
(``op.execute``), not ORM metadata, so ``target_metadata`` is ``None`` and
autogenerate is not used — migrations are hand-written.

The async app URL (``postgresql+asyncpg://``) is translated to the sync
psycopg 3 driver here, because Alembic runs migrations with a synchronous
engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable even when alembic runs from another
# cwd (e.g. via the programmatic API from backend/db/schema.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from alembic import context  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

from backend.shared.config import config  # noqa: E402

# Schema is raw SQL, not ORM metadata — autogenerate has nothing to compare.
target_metadata = None


def _sync_url() -> str:
    url = config.database_url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_sync_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_sync_url())
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
