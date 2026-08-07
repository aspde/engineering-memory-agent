"""Tests for schema DDL generation — embedding dimension parameterization.

These are pure-function tests: ``build_schema_statements(dimension)`` must emit
``vector(<dimension>)`` for every embedding column and a resize migration that
clears + resizes when the live column dimension differs.  No database is
touched.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.shared.config import config


def _statements(dimension: int) -> list[str]:
    from backend.db.schema import build_schema_statements

    return build_schema_statements(dimension)


def _create_tables(statements: list[str]) -> dict[str, str]:
    """Map table name → its CREATE TABLE statement."""
    out: dict[str, str] = {}
    for s in statements:
        for table in ("chunks", "memories", "entities"):
            if f"CREATE TABLE IF NOT EXISTS {table}" in s:
                out[table] = s
    return out


class TestDimensionParameterization:
    def test_default_dimension_used(self) -> None:
        s = _statements(1024)
        assert any("vector(1024)" in x for x in s)
        assert not any("vector(1536)" in x for x in s)

    def test_other_dimension_used(self) -> None:
        s = _statements(3072)
        assert any("vector(3072)" in x for x in s)
        assert not any("vector(1024)" in x for x in s)

    def test_all_embedding_tables_share_dimension(self) -> None:
        for dim in (1024, 1536, 3072):
            tables = _create_tables(_statements(dim))
            assert set(tables) == {"chunks", "memories", "entities"}
            for name, stmt in tables.items():
                assert f"vector({dim})" in stmt, f"{name} CREATE TABLE lacks vector({dim})"

    def test_non_dimension_statements_identical_across_dimensions(self) -> None:
        s1024 = _statements(1024)
        s3072 = _statements(3072)

        def _plain(statements: list[str]) -> list[str]:
            # Drop statements whose text legitimately embeds the dimension:
            # CREATE TABLEs and the resize DO blocks.
            return [x for x in statements if "vector(" not in x]

        assert _plain(s1024) == _plain(s3072)


class TestResizeMigration:
    def test_resize_statement_clears_then_resizes(self) -> None:
        from backend.db.schema import _resize_statement

        sql = _resize_statement("memories", "idx_memories_embedding", 1536)
        assert "UPDATE memories SET embedding = NULL" in sql
        assert "ALTER TABLE memories ALTER COLUMN embedding TYPE vector(1536)" in sql
        assert "DROP INDEX IF EXISTS idx_memories_embedding" in sql
        assert (
            "CREATE INDEX IF NOT EXISTS idx_memories_embedding "
            "ON memories USING hnsw" in sql
        )

    def test_resize_only_guards_on_different_vector_dimension(self) -> None:
        from backend.db.schema import _resize_statement

        sql = _resize_statement("chunks", "idx_chunks_embedding", 1024)
        # no-op when the column already matches…
        assert "cur IS DISTINCT FROM 'vector(1024)'" in sql
        # …and only ever touches columns that are actually vector-typed
        # (never a plain TEXT/other column, and never a missing column).
        assert "cur LIKE 'vector%'" in sql

    def test_each_embedding_table_gets_a_resize_block(self) -> None:
        s = _statements(1536)
        for table in ("chunks", "memories", "entities"):
            assert any(
                f"WHERE a.attrelid = '{table}'::regclass" in x for x in s
            ), f"no resize block for {table}"

    def test_embedding_indexes_use_hnsw(self) -> None:
        """Embedding indexes are HNSW (pgvector >= 0.5), not ivfflat.

        ivfflat's lists=100 spread probes over mostly-empty clusters on small
        corpora (the seed-010 lesson); HNSW has no cluster dependency.
        """
        s = _statements(1024)
        for table, index in (
            ("chunks", "idx_chunks_embedding"),
            ("memories", "idx_memories_embedding"),
            ("entities", "idx_entities_embedding"),
        ):
            create = next(x for x in s if f"CREATE INDEX IF NOT EXISTS {index}" in x)
            assert f"ON {table} USING hnsw" in create
        # ivfflat survives only inside the migration block's LIKE predicate,
        # never as an actual CREATE INDEX / resize-rebuild statement.
        non_migration = [x for x in s if "indexdef LIKE '%ivfflat%'" not in x]
        assert not any("ivfflat" in x for x in non_migration)

    def test_ivfflat_to_hnsw_migration_block_present(self) -> None:
        """Existing ivfflat indexes are swapped for HNSW once, then a no-op."""
        s = _statements(1024)
        migration = next(
            x for x in s
            if "FOREACH t IN ARRAY ARRAY['chunks', 'memories', 'entities']" in x
        )
        assert "indexdef LIKE '%ivfflat%'" in migration
        assert "USING hnsw (embedding vector_cosine_ops)" in migration
        assert "DROP INDEX" in migration


class TestInitDbDimensionFlow:
    """init_db() wires the dimension into the generated DDL."""

    class _FakeConn:
        def __init__(self) -> None:
            self.executed: list[str] = []

        async def execute(self, stmt, *args, **kwargs) -> None:
            self.executed.append(str(stmt))

    class _FakeEngine:
        def __init__(self) -> None:
            self.conn = TestInitDbDimensionFlow._FakeConn()

        def begin(self):
            return _AsyncCM(self.conn)

    @pytest.mark.asyncio
    async def test_defaults_to_config_dimension(self, monkeypatch) -> None:
        from backend.db import schema as schema_mod

        engine = TestInitDbDimensionFlow._FakeEngine()
        monkeypatch.setattr(schema_mod, "get_engine", lambda: engine)

        await schema_mod.init_db()

        chunks = [s for s in engine.conn.executed if "CREATE TABLE IF NOT EXISTS chunks" in s]
        assert chunks
        assert f"vector({config.embedding.dimension})" in chunks[0]

    @pytest.mark.asyncio
    async def test_explicit_dimension_overrides_config(self, monkeypatch) -> None:
        from backend.db import schema as schema_mod

        engine = TestInitDbDimensionFlow._FakeEngine()
        monkeypatch.setattr(schema_mod, "get_engine", lambda: engine)

        await schema_mod.init_db(dimension=3072)

        chunks = [s for s in engine.conn.executed if "CREATE TABLE IF NOT EXISTS chunks" in s]
        assert chunks
        assert "vector(3072)" in chunks[0]


class _AsyncCM:
    """Async context manager yielding a connection, for the fake engine."""

    def __init__(self, conn: TestInitDbDimensionFlow._FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc) -> bool:
        return False
