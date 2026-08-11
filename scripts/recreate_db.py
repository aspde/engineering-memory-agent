"""Recreate the dev database: drop the public schema, rebuild via Alembic.

Use when ``init_db`` refuses to start because the embedding dimension
changed — ``EMBEDDING_MODEL`` and the pgvector columns are out of sync
(see ``docs/memory-system.md`` 维度).  All data is lost; re-ingest your
sources afterwards (re-run connectors / re-import git repos).

    python -m scripts.recreate_db

Deliberately destructive and never called automatically: the dimension check
in ``init_db`` raises instead of auto-migrating precisely so this decision
stays explicit.  ``DROP SCHEMA public CASCADE`` also removes the pgvector
extension (installed in ``public``); the Alembic baseline re-creates it with
``CREATE EXTENSION IF NOT EXISTS vector``.
"""

from __future__ import annotations


def main() -> None:
    import psycopg

    from backend.db.schema import run_alembic_upgrade
    from backend.shared.config import config

    # psycopg.connect cannot parse SQLAlchemy driver suffixes — strip them.
    url = config.database_url
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if url.startswith(prefix):
            url = url.replace(prefix, "postgresql://", 1)
            break

    conn = psycopg.connect(url)
    try:
        conn.autocommit = True
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    finally:
        conn.close()

    run_alembic_upgrade()
    print("Database recreated; re-ingest your sources.")


if __name__ == "__main__":
    main()
