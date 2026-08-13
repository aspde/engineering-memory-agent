"""Migration tests — Alembic baseline lifecycle against a throwaway database.

These run against a dedicated temporary database (never the shared
``ema_test`` the rest of the suite uses), because they deliberately
``downgrade`` the schema to empty and rebuild it — that would wreck the
tables other test modules rely on if run in place.

Covered:
  - ``upgrade head`` on an empty database creates exactly the 9 business
    tables (the ``EXPECTED_TABLES`` set below).
  - the version table is stamped at head.
  - ``downgrade base`` clears the tables; ``upgrade head`` rebuilds.
  - running ``upgrade head`` twice is a no-op.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig

from backend.db.schema import _ALEMBIC_INI, _MIGRATIONS_DIR
from backend.shared.config import config

EXPECTED_TABLES = {"chunks", "memories", "entities", "memory_entities",
                   "conversations", "webhook_logs", "patrol_logs",
                   "pending_conflicts", "llm_usage"}


def _migration_config() -> AlembicConfig:
    cfg = AlembicConfig(str(_ALEMBIC_INI))
    cfg.set_main_option("script_location", str(_MIGRATIONS_DIR))
    return cfg


def _admin_conn_string() -> str:
    parts = urlsplit(config.database_url)
    admin = urlunsplit((parts.scheme, parts.netloc, "/postgres", "", ""))
    return _psycopg_url(admin)


def _psycopg_url(url: str) -> str:
    """Strip the SQLAlchemy driver suffix psycopg.connect cannot parse."""
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg://"):
        if url.startswith(prefix):
            return url.replace(prefix, "postgresql://", 1)
    return url


def _schema_tables(conn) -> set[str]:
    cur = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
    )
    return {row[0] for row in cur.fetchall()}


@pytest.fixture()
def temp_migration_db():
    """A scratch database Alembic may freely drop/rebuild, restored after."""
    name = f"ema_mig_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    admin = psycopg.connect(_admin_conn_string())
    admin.autocommit = True
    admin.execute(f'CREATE DATABASE "{name}"')
    admin.close()

    original_url = config.database_url
    parts = urlsplit(original_url)
    # 继承原 URL 的用户/密码/主机——测试不应假设数据库密码固定为
    # 某个值（本地改强密码后硬编码会认证失败；通过 TEST_DATABASE_URL
    # 覆盖即可）。
    config.database_url = f"postgresql://{parts.netloc}/{name}"
    try:
        yield name
    finally:
        config.database_url = original_url
        admin = psycopg.connect(_admin_conn_string())
        admin.autocommit = True
        admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        admin.close()


class TestBaselineMigration:
    def test_upgrade_creates_all_business_tables(self, temp_migration_db) -> None:
        command.upgrade(_migration_config(), "head")

        conn = psycopg.connect(
            _psycopg_url(config.database_url)
        )
        try:
            tables = _schema_tables(conn)
            assert EXPECTED_TABLES <= tables, (
                f"missing tables: {EXPECTED_TABLES - tables}"
            )
            # And nothing beyond the expected business tables + version table.
            extra = tables - EXPECTED_TABLES - {"alembic_version"}
            assert not extra, f"unexpected tables: {extra}"
        finally:
            conn.close()

    def test_version_stamped_at_head(self, temp_migration_db) -> None:
        command.upgrade(_migration_config(), "head")

        conn = psycopg.connect(
            _psycopg_url(config.database_url)
        )
        try:
            version = conn.execute(
                "SELECT version_num FROM alembic_version"
            ).fetchone()
            assert version == ("0002_drop_decay_factor",)
        finally:
            conn.close()

    def test_downgrade_clears_then_upgrade_rebuilds(self, temp_migration_db) -> None:
        command.upgrade(_migration_config(), "head")

        command.downgrade(_migration_config(), "base")
        conn = psycopg.connect(
            _psycopg_url(config.database_url)
        )
        try:
            tables = _schema_tables(conn)
            assert EXPECTED_TABLES.isdisjoint(tables), (
                f"business tables survived downgrade: {EXPECTED_TABLES & tables}"
            )
        finally:
            conn.close()

        command.upgrade(_migration_config(), "head")
        conn = psycopg.connect(
            _psycopg_url(config.database_url)
        )
        try:
            assert EXPECTED_TABLES <= _schema_tables(conn)
        finally:
            conn.close()

    def test_upgrade_is_idempotent(self, temp_migration_db) -> None:
        command.upgrade(_migration_config(), "head")
        # Second run: no pending revisions, no error, schema unchanged.
        command.upgrade(_migration_config(), "head")
        conn = psycopg.connect(
            _psycopg_url(config.database_url)
        )
        try:
            assert EXPECTED_TABLES <= _schema_tables(conn)
        finally:
            conn.close()

    def test_upgrade_backfills_columns_on_partially_migrated_db(
        self, temp_migration_db,
    ) -> None:
        """A live database that predates a column addition is patched.

        ``CREATE TABLE IF NOT EXISTS`` skips existing tables, so the
        historical ``ADD COLUMN IF NOT EXISTS`` clauses must backfill the
        columns — and do so *before* the indexes that reference them
        (``uq_pending_conflicts_patrol_pair`` reads ``conflict_type`` /
        ``peer_id``; the alerts query reads ``llm_usage.attempts``).
        """
        conn = psycopg.connect(_psycopg_url(config.database_url))
        try:
            conn.autocommit = True
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                """
                CREATE TABLE chunks (
                    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    document_id TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    embedding   vector(1024),
                    meta        JSONB DEFAULT '{}',
                    created_at  TIMESTAMPTZ DEFAULT now(),
                    chunk_index INT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE pending_conflicts (
                    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source           TEXT NOT NULL,
                    source_type      TEXT,
                    existing_id      UUID NOT NULL,
                    existing_summary TEXT NOT NULL,
                    new_summary      TEXT NOT NULL,
                    deferred         JSONB NOT NULL,
                    status           TEXT NOT NULL DEFAULT 'pending',
                    resolution       TEXT,
                    created_at       TIMESTAMPTZ DEFAULT now(),
                    resolved_at      TIMESTAMPTZ
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE llm_usage (
                    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    seq            BIGSERIAL,
                    trace_id       TEXT,
                    thread_id      TEXT,
                    scenario       TEXT NOT NULL,
                    provider       TEXT NOT NULL,
                    model          TEXT NOT NULL,
                    input_tokens   INT,
                    output_tokens  INT,
                    total_tokens   INT,
                    cache_read_tokens     INT,
                    cache_creation_tokens INT,
                    latency_ms     INT,
                    status         TEXT NOT NULL DEFAULT 'success',
                    error          TEXT,
                    prompt_chars   INT,
                    response_chars INT,
                    created_at     TIMESTAMPTZ DEFAULT now()
                )
                """
            )
        finally:
            conn.close()

        command.upgrade(_migration_config(), "head")

        conn = psycopg.connect(_psycopg_url(config.database_url))
        try:
            def _columns(table: str) -> set[str]:
                rows = conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = %s",
                    (table,),
                ).fetchall()
                return {r[0] for r in rows}

            assert {"tokens", "content_hash"} <= _columns("chunks")
            assert {"conflict_type", "peer_id"} <= _columns("pending_conflicts")
            assert {"attempts"} <= _columns("llm_usage")

            # The partial patrol-pair index depends on the backfilled
            # columns — its presence proves the ALTERs ran before the
            # index creation.
            pair_idx = conn.execute(
                "SELECT 1 FROM pg_indexes WHERE indexname = "
                "'uq_pending_conflicts_patrol_pair'"
            ).fetchone()
            assert pair_idx is not None
        finally:
            conn.close()


class TestInitDbIntegration:
    """init_db() wires Alembic into the app's startup path.

    Runs against the temp database: the alembic upgrade inside init_db reads
    ``config.database_url`` (pointed at the temp db by the fixture) and
    creates the schema there.  Verification uses a direct psycopg connection
    to the temp db — ``get_engine()`` is a module-level singleton bound to
    the *original* DATABASE_URL at import, so it must not be used here.
    """

    @pytest.mark.asyncio
    async def test_init_db_upgrades_fresh_database(self, temp_migration_db) -> None:
        from backend.db.schema import init_db

        await init_db()

        conn = psycopg.connect(
            _psycopg_url(config.database_url)
        )
        try:
            tables = _schema_tables(conn)
            version = conn.execute(
                "SELECT version_num FROM alembic_version"
            ).fetchone()
        finally:
            conn.close()
        assert EXPECTED_TABLES <= tables
        assert version[0] == "0002_drop_decay_factor"

    @pytest.mark.asyncio
    async def test_init_db_rejects_dimension_mismatch(self, temp_migration_db) -> None:
        """A configured dimension different from the migration's vector(1024)
        is refused — init_db verifies, it never auto-migrates."""
        from backend.db.schema import init_db

        with pytest.raises(RuntimeError, match=r"vector\(1536\)"):
            await init_db(dimension=1536)
