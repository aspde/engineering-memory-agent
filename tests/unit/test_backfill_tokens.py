"""Tests for scripts.backfill_tokens.backfill — tokens column backfill for old rows.

DB is mocked at the session-factory boundary: verify dry-run writes nothing,
and the real path issues exactly one executemany UPDATE for all rows.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _mock_factory(session):
    """Return a callable standing in for ``get_session_factory``.

    backfill does ``factory = get_session_factory(); async with factory() as s:``
    — a two-step call.  So get_session_factory() must yield a callable whose
    call yields the async context manager.
    """
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=ctx)
    return lambda: factory


def _row(id_: str, content: str) -> MagicMock:
    r = MagicMock()
    r._mapping = {"id": id_, "content": content}
    return r


class TestBackfill:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_write(self, monkeypatch) -> None:
        from scripts.backfill_tokens import backfill

        session = AsyncMock()
        result = MagicMock()
        result.__iter__ = lambda self: iter([_row("uuid-1", "pgvector 向量检索")])
        session.execute.return_value = result
        monkeypatch.setattr("scripts.backfill_tokens.get_session_factory", _mock_factory(session))

        n = await backfill(dry_run=True)

        assert n == 0
        session.execute.assert_awaited_once()  # SELECT only, no UPDATE
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_backfills_rows_in_one_executemany(self, monkeypatch) -> None:
        from scripts.backfill_tokens import backfill

        session = AsyncMock()
        rows = [
            _row("uuid-1", "pgvector 向量检索"),
            _row("uuid-2", "BGE-M3 CPU 推理"),
        ]
        select_result = MagicMock()
        select_result.__iter__ = lambda self: iter(rows)
        session.execute.side_effect = [select_result, AsyncMock()]
        monkeypatch.setattr("scripts.backfill_tokens.get_session_factory", _mock_factory(session))

        n = await backfill(dry_run=False)

        assert n == 2
        session.commit.assert_awaited_once()
        assert len(session.execute.await_args_list) == 2
        # Second call is one executemany UPDATE carrying a param list.
        stmt, params = session.execute.await_args_list[1].args
        assert "UPDATE chunks SET tokens" in str(stmt)
        assert len(params) == 2
        assert all(p["tokens"] for p in params)  # jieba produced tokens

    @pytest.mark.asyncio
    async def test_no_rows_to_backfill(self, monkeypatch) -> None:
        from scripts.backfill_tokens import backfill

        session = AsyncMock()
        result = MagicMock()
        result.__iter__ = lambda self: iter([])
        session.execute.return_value = result
        monkeypatch.setattr("scripts.backfill_tokens.get_session_factory", _mock_factory(session))

        n = await backfill(dry_run=False)

        assert n == 0
        session.execute.assert_awaited_once()  # SELECT, no UPDATE
        session.commit.assert_not_awaited()
