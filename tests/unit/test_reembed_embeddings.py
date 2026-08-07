"""Tests for reembed_embeddings.backfill — NULL-embedding re-embed after a
dimension migration.

DB and embedding provider are mocked: verify dry-run writes nothing (and
loads no model), and the real path issues one executemany UPDATE per batch.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _mock_factory(session):
    """Stand in for ``get_session_factory`` (two-step call — see backfill_tokens).

    ``factory = get_session_factory(); async with factory() as s:``
    """
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=ctx)
    return lambda: factory


def _row(id_: str, text: str) -> MagicMock:
    r = MagicMock()
    r._mapping = {"id": id_, "content": text}
    return r


def _empty_result() -> MagicMock:
    r = MagicMock()
    r.__iter__ = lambda self: iter([])
    return r


class _FakeProvider:
    """Embedding provider stub — one fixed-dim vector per input text."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]):
        self.calls.append(list(texts))
        return [[0.1, 0.2, 0.3]] * len(texts)


class TestBackfill:
    @pytest.mark.asyncio
    async def test_dry_run_counts_without_loading_provider(self, monkeypatch) -> None:
        from reembed_embeddings import backfill

        session = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 3
        session.execute.return_value = count_result
        monkeypatch.setattr("reembed_embeddings.get_session_factory", _mock_factory(session))

        def _fail(*args, **kwargs):
            raise AssertionError("dry-run must not instantiate the embedding provider")

        monkeypatch.setattr("reembed_embeddings.get_embedding_provider", _fail)

        counts = await backfill(["chunks"], dry_run=True)

        assert counts == {"chunks": 3}
        session.execute.assert_awaited_once()
        assert "COUNT(*)" in str(session.execute.await_args.args[0])
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reembeds_rows_in_one_executemany_per_batch(self, monkeypatch) -> None:
        from reembed_embeddings import backfill

        session = AsyncMock()
        rows = [_row("uuid-1", "pgvector 向量检索"), _row("uuid-2", "BGE-M3 CPU 推理")]
        select_result = MagicMock()
        select_result.__iter__ = lambda self: iter(rows)
        # SELECT rows → UPDATE → SELECT empty (loop terminates).
        session.execute.side_effect = [select_result, AsyncMock(), _empty_result()]
        monkeypatch.setattr("reembed_embeddings.get_session_factory", _mock_factory(session))

        provider = _FakeProvider()
        monkeypatch.setattr("reembed_embeddings.get_embedding_provider", lambda: provider)

        counts = await backfill(["chunks"], dry_run=False)

        assert counts == {"chunks": 2}
        session.commit.assert_awaited_once()
        assert len(session.execute.await_args_list) == 3
        update_stmt, params = session.execute.await_args_list[1].args
        assert "UPDATE chunks SET embedding" in str(update_stmt)
        assert len(params) == 2
        assert all(p["vec"].startswith("[") for p in params)
        assert provider.calls == [["pgvector 向量检索", "BGE-M3 CPU 推理"]]

    @pytest.mark.asyncio
    async def test_no_rows_to_reembed(self, monkeypatch) -> None:
        from reembed_embeddings import backfill

        session = AsyncMock()
        session.execute.return_value = _empty_result()
        monkeypatch.setattr("reembed_embeddings.get_session_factory", _mock_factory(session))
        monkeypatch.setattr("reembed_embeddings.get_embedding_provider", _FakeProvider)

        counts = await backfill(["memories"], dry_run=False)

        assert counts == {"memories": 0}
        session.execute.assert_awaited_once()  # one SELECT, no UPDATE
        session.commit.assert_not_awaited()
