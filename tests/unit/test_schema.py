"""Tests for schema init — embedding dimension verification.

``build_schema_statements`` was removed along with the auto-migrate behaviour
it served; ``init_db`` now runs Alembic migrations and *verifies* the live
embedding columns match the configured dimension, raising instead of
silently emptying them.  These tests exercise the check with a fake engine;
no database is touched.
"""

from __future__ import annotations

import pytest

from backend.shared.config import config


class _FakeResult:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[str, str]]:
        return self._rows


class _FakeConn:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._rows = rows

    async def execute(self, stmt, *args, **kwargs) -> _FakeResult:
        return _FakeResult(self._rows)


class _FakeEngine:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._conn = _FakeConn(rows)

    def connect(self):
        return _AsyncCM(self._conn)

    async def dispose(self) -> None:
        pass


class _AsyncCM:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc) -> bool:
        return False


def _all_match(dimension: int) -> list[tuple[str, str]]:
    return [
        ("chunks", f"vector({dimension})"),
        ("memories", f"vector({dimension})"),
        ("entities", f"vector({dimension})"),
    ]


class TestDimensionMismatch:
    @pytest.mark.asyncio
    async def test_all_match_returns_empty(self, monkeypatch) -> None:
        from backend.db import schema as schema_mod

        monkeypatch.setattr(
            schema_mod, "create_async_engine",
            lambda *a, **k: _FakeEngine(_all_match(1024)),
        )
        assert await schema_mod._dimension_mismatches(1024) == []

    @pytest.mark.asyncio
    async def test_wrong_dimension_reported(self, monkeypatch) -> None:
        from backend.db import schema as schema_mod

        rows = [
            ("chunks", "vector(1024)"),
            ("memories", "vector(1536)"),
            ("entities", "vector(1024)"),
        ]
        monkeypatch.setattr(
            schema_mod, "create_async_engine",
            lambda *a, **k: _FakeEngine(rows),
        )
        # The two 1024 columns are the mismatch when 1536 is expected.
        assert await schema_mod._dimension_mismatches(1536) == ["chunks", "entities"]

    @pytest.mark.asyncio
    async def test_init_db_passes_when_matching(self, monkeypatch) -> None:
        from backend.db import schema as schema_mod

        monkeypatch.setattr(schema_mod, "run_alembic_upgrade", lambda: None)
        monkeypatch.setattr(
            schema_mod, "create_async_engine",
            lambda *a, **k: _FakeEngine(_all_match(config.embedding.dimension)),
        )
        await schema_mod.init_db()

    @pytest.mark.asyncio
    async def test_init_db_raises_on_mismatch(self, monkeypatch) -> None:
        from backend.db import schema as schema_mod

        monkeypatch.setattr(schema_mod, "run_alembic_upgrade", lambda: None)
        monkeypatch.setattr(
            schema_mod, "create_async_engine",
            lambda *a, **k: _FakeEngine(_all_match(1024)),
        )
        with pytest.raises(RuntimeError, match="chunks, memories, entities"):
            await schema_mod.init_db(dimension=1536)
