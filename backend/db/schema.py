"""Database schema — owned entirely by Alembic migrations.

Tables (chunks / memories / entities / conversations / webhook_logs /
patrol_logs / pending_conflicts / llm_usage) are defined in
``migrations/versions/0001_baseline.py``; every schema change is a new
versioned migration.  ``init_db()`` is the single entry point every caller
uses (FastAPI lifespan, tests, eval scripts): it runs ``alembic upgrade
head`` — creating missing tables on a fresh database, skipping
already-present tables on an existing one, and stamping the version table
either way.

The only piece that stays *outside* Alembic is verifying the embedding-column
dimension.  The configured embedding model's dimension
(``config.embedding.dimension``) cannot live in a static migration; the
baseline migration uses 1024 as a placeholder, and :func:`init_db` checks the
live columns against the configured dimension and refuses to start on a
mismatch.  A dev database whose embedding model changed is recreated
(``python -m scripts.recreate_db``) and re-ingested — it is never silently
emptied by an auto-migration.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from backend.shared.config import config

# Project root = backend/db/schema.py → backend/db → backend → root.
_ALEMBIC_INI = Path(__file__).resolve().parent.parent.parent / "alembic.ini"
_MIGRATIONS_DIR = _ALEMBIC_INI.parent / "migrations"


def run_alembic_upgrade() -> None:
    """Apply all pending Alembic migrations (``upgrade head``).

    Runs with a *synchronous* engine (psycopg) on a plain thread — Alembic
    needs a sync connection, and calling it from an async context would fight
    the running event loop.  The URL comes from ``env.py``, which reads the
    app's own ``config.database_url``.
    """
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig(str(_ALEMBIC_INI))
    # Absolute path so the ini's relative script_location resolves regardless
    # of the caller's working directory.
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    command.upgrade(cfg, "head")


def _config_async_url() -> str:
    """The configured database URL in asyncpg form (``postgresql+asyncpg://``)."""
    url = config.database_url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


async def _dimension_mismatches(dimension: int) -> list[str]:
    """Tables whose embedding column is not ``vector(dimension)``.

    The pgvector column type (``vector(N)``) is fixed by the schema; a
    configured model with a different dimension makes every embedding write
    fail against that column, so the mismatch is surfaced before any data is
    touched.  Returns the mismatched table names, empty when all match.
    """
    engine = create_async_engine(_config_async_url(), poolclass=NullPool)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    """\
                    SELECT c.relname AS tbl,
                           format_type(a.atttypid, a.atttypmod) AS col_type
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    JOIN pg_attribute a ON a.attrelid = c.oid
                    WHERE n.nspname = 'public'
                      AND c.relname IN ('chunks', 'memories', 'entities')
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped
                    """
                )
            )
            rows = result.fetchall()
    finally:
        await engine.dispose()

    expected = f"vector({dimension})"
    return [row[0] for row in rows if row[1] != expected]


async def init_db(dimension: int | None = None) -> None:
    """Bring the database schema up to date.

    Applies all Alembic migrations (``upgrade head``) — creating tables on a
    fresh database, skipping already-present ones on an existing database —
    then *verifies* the embedding columns match the configured dimension.

    *dimension* defaults to ``config.embedding.dimension`` — tests may pass an
    explicit value to exercise other dimensions without touching config.

    A dimension mismatch raises ``RuntimeError`` instead of auto-migrating:
    resizing a pgvector column requires emptying it (recomputed embeddings),
    and silently doing that on startup is a destructive surprise a dev tool
    must not hide.  A database whose embedding model changed is recreated with
    ``python -m scripts.recreate_db`` and re-ingested.

    The check runs on a short-lived engine built from the *current*
    ``config.database_url`` rather than the module-level ``get_engine()``
    singleton (which is bound to the URL at import time).  Alembic's ``env.py``
    reads ``config.database_url`` live, so both halves of ``init_db`` must
    target the same database even when the URL is switched at runtime — the
    migration tests point it at a scratch database.
    """
    await asyncio.to_thread(run_alembic_upgrade)
    dimension = dimension if dimension is not None else config.embedding.dimension
    mismatched = await _dimension_mismatches(dimension)
    if mismatched:
        raise RuntimeError(
            f"Embedding dimension mismatch on {', '.join(mismatched)} "
            f"(expected vector({dimension}) for "
            f"EMBEDDING_MODEL={config.embedding.model}). Refusing to "
            f"auto-migrate — recreate the database with "
            f"`python -m scripts.recreate_db`, or align EMBEDDING_MODEL with "
            f"the schema."
        )
    print("Database initialized (schema via alembic upgrade head)")
