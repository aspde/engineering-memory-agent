"""Tests for recall tracking — the metadata recorded on every memory search.

Ranking is pure similarity (see ``retrieval.search_memories``); this module
only verifies the recall write is atomic, single-batch, and safe to skip.
"""

from __future__ import annotations

import pytest


class TestRecordRecallsUnit:
    """record_recalls — one atomic UPDATE for many memories (the N+1 fix)."""

    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self, monkeypatch) -> None:
        from backend.service import recall as mod

        monkeypatch.setattr(
            mod, "get_session_factory", lambda: (_ for _ in ()).throw(AssertionError("must not touch DB"))
        )
        assert await mod.record_recalls([]) is None

    @pytest.mark.asyncio
    async def test_records_are_bumped(self, monkeypatch) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from backend.service import recall as mod

        session = MagicMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        # session_factory() is an async context manager yielding the session.
        factory = MagicMock()
        factory.return_value.__aenter__.return_value = session
        factory.return_value.__aexit__ = AsyncMock(return_value=False)
        monkeypatch.setattr(mod, "get_session_factory", lambda: factory)

        await mod.record_recalls(["m1", "m2"])

        session.execute.assert_awaited_once()
        call_args, _ = session.execute.call_args
        stmt = call_args[0].text
        assert "recall_count = recall_count + 1" in stmt
        assert "recalled_at = NOW()" in stmt
        assert "WHERE id = ANY(:ids)" in stmt
        assert call_args[1]["ids"] == ["m1", "m2"]
        session.commit.assert_awaited_once()


class TestSearchMemoriesRanking:
    """search_memories ranks by raw similarity — no decay weighting."""

    def test_module_has_no_decay_formula(self) -> None:
        """The decay Python mirror is gone; only the similarity query remains."""
        import inspect

        from backend.service import retrieval

        source = inspect.getsource(retrieval.search_memories)
        assert "exp(-" not in source
        assert "decay_factor" not in source
        assert "similarity" in source
