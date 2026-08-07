"""Tests for the memory write-path's correctness-critical structured call.

``_detect_conflict`` is correctness-critical: after retries are exhausted it
must raise (so the write fails) rather than silently assume no conflict.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from backend.model.llm import LLMStructuredError
from backend.service.memory import _detect_conflict


class TestDetectConflict:
    @pytest.mark.asyncio
    async def test_returns_true_when_conflict(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat_json.return_value = json.dumps({"conflict": True})
        monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: mock_llm)

        assert await _detect_conflict({"summary": "old"}, {"summary": "new"}) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_conflict(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat_json.return_value = json.dumps({"conflict": False})
        monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: mock_llm)

        assert await _detect_conflict({"summary": "old"}, {"summary": "new"}) is False

    @pytest.mark.asyncio
    async def test_propagates_on_exhaustion(self, monkeypatch) -> None:
        """Never silently assumes no-conflict when structured output fails."""

        async def _raise(*args, **kwargs):
            raise LLMStructuredError("no schema-valid JSON after retries")

        monkeypatch.setattr("backend.service.structured.chat_structured", _raise)

        with pytest.raises(LLMStructuredError):
            await _detect_conflict({"summary": "old"}, {"summary": "new"})


def _make_session_factory(mock_session: AsyncMock, side_effect: list | None = None):
    """Build a mock session factory that returns *mock_session* inside
    ``async with`` blocks (mirrors the real ``get_session_factory()`` shape).
    """
    if side_effect is not None:
        mock_session.execute.side_effect = side_effect

    mock_sess = AsyncMock()
    mock_sess.__aenter__ = AsyncMock(return_value=mock_session)
    mock_sess.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock()
    factory.return_value = mock_sess
    return factory


class TestMergeMemory:
    """Regression tests for merge-path correctness (_merge_memory)."""

    @pytest.mark.asyncio
    async def test_merge_reembeds_merged_summary_and_dedups_relations(self) -> None:
        """Stored vector must be computed over the merged summary, and
        relations deduplicated by (from, to, type) — not the new content's
        embedding, and not a (subject, predicate, object) key that collapses
        every relation to the first one.
        """
        from backend.service.memory import _merge_memory

        existing = {
            "id": "11111111-1111-1111-1111-111111111111",
            "summary": "Existing summary.",
            "entities": [{"name": "A", "type": "concept"}],
            "relations": [{"from": "A", "to": "B", "type": "depends_on"}],
        }
        extracted = {
            "summary": "New summary.",
            "entities": [{"name": "B", "type": "concept"}],
            "relations": [
                {"from": "A", "to": "B", "type": "depends_on"},  # duplicate
                {"from": "B", "to": "C", "type": "causes"},      # new
            ],
        }

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "Merged summary."
        mock_emb = AsyncMock()
        mock_emb.embed.return_value = [[0.42] * 1024]
        mock_session = AsyncMock()

        with (
            patch(
                "backend.service.llm_service.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "backend.service.memory.get_embedding_provider",
                return_value=mock_emb,
            ),
            patch(
                "backend.service.memory.get_session_factory",
                return_value=_make_session_factory(mock_session),
            ),
            patch("backend.service.memory._schedule_normalization"),
        ):
            result = await _merge_memory(
                existing, extracted, [0.1] * 1024, "doc", None, "abc123hash"
            )

        assert result["action"] == "merged"

        # Re-embedded the merged summary (not the incoming embedding)
        embed_texts = mock_emb.embed.call_args[0][0]
        assert embed_texts == ["Merged summary."]

        # UPDATE stored the merged-summary embedding + content hash
        sql, params = mock_session.execute.call_args[0]
        assert params["summary"] == "Merged summary."
        assert params["embedding"] == str([0.42] * 1024)
        assert params["content_hash"] == "abc123hash"

        # Relations deduplicated by (from, to, type); the duplicate is kept once
        stored_relations = json.loads(params["relations"])
        assert stored_relations == [
            {"from": "A", "to": "B", "type": "depends_on"},
            {"from": "B", "to": "C", "type": "causes"},
        ]

    @pytest.mark.asyncio
    async def test_merge_content_hash_collision_falls_back_to_duplicate(self) -> None:
        """If the merged UPDATE hits the content_hash unique index (a concurrent
        identical write stored the same hash on another live memory), return
        that memory as a duplicate instead of crashing."""
        from backend.service.memory import _merge_memory

        existing = {
            "id": "11111111-1111-1111-1111-111111111111",
            "summary": "Existing summary.",
            "entities": [],
            "relations": [],
        }
        extracted = {
            "summary": "New summary.",
            "entities": [],
            "relations": [],
        }
        winner = {"id": "33333333-3333-3333-3333-333333333333", "summary": "Winner summary."}
        select_result = MagicMock()
        select_result.fetchone.return_value = MagicMock(_mapping=winner)

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "Merged summary."
        mock_emb = AsyncMock()
        mock_emb.embed.return_value = [[0.42] * 1024]
        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            IntegrityError("UPDATE ...", {}, Exception("unique_violation")),
            select_result,
        ]

        with (
            patch(
                "backend.service.llm_service.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "backend.service.memory.get_embedding_provider",
                return_value=mock_emb,
            ),
            patch(
                "backend.service.memory.get_session_factory",
                return_value=_make_session_factory(mock_session),
            ),
            patch("backend.service.memory._schedule_normalization"),
        ):
            result = await _merge_memory(
                existing, extracted, [0.1] * 1024, "doc", None, "abc123hash"
            )

        assert result["action"] == "duplicate"
        assert result["id"] == winner["id"]
        assert result["summary"] == winner["summary"]


class TestResolveConflictMerge:
    """Regression tests for resolve_conflict(..., resolution="merge")."""

    @pytest.mark.asyncio
    async def test_merge_keeps_multiple_relations_and_reembeds(self) -> None:
        """The (subject, predicate, object) dedup key collapsed every relation
        to the first one; (from, to, type) must preserve all unique relations.
        """
        from backend.service.memory import resolve_conflict

        existing_id = "11111111-1111-1111-1111-111111111111"
        existing_row = (
            "Existing summary.",
            json.dumps([{"name": "A", "type": "concept"}]),
            json.dumps([{"from": "A", "to": "B", "type": "depends_on"}]),
        )
        mock_select = MagicMock()
        mock_select.fetchone.return_value = existing_row
        mock_update = MagicMock()

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "Merged by human."
        mock_emb = AsyncMock()
        mock_emb.embed.return_value = [[0.5] * 1024]
        mock_session = AsyncMock()

        deferred = {
            "extracted": {
                "summary": "New summary.",
                "entities": [{"name": "B", "type": "concept"}],
                "relations": [
                    {"from": "B", "to": "C", "type": "causes"},
                    {"from": "A", "to": "B", "type": "depends_on"},  # dup of existing
                ],
            },
            "embedding": str([0.1] * 1024),
            "source_type": "conversation",
            "metadata": {"conflicts_with": existing_id},
            "content_hash": "def456hash",
        }

        with (
            patch(
                "backend.service.llm_service.get_llm_provider",
                return_value=mock_llm,
            ),
            patch(
                "backend.service.memory.get_embedding_provider",
                return_value=mock_emb,
            ),
            patch(
                "backend.service.memory.get_session_factory",
                return_value=_make_session_factory(
                    mock_session, side_effect=[mock_select, mock_update]
                ),
            ),
        ):
            result = await resolve_conflict("merge", existing_id, deferred)

        assert result["resolution"] == "merge"

        # Second execute is the UPDATE (first is the SELECT of existing row)
        sql, update_params = mock_session.execute.call_args_list[1][0]
        stored_relations = json.loads(update_params["relations"])
        assert stored_relations == [
            {"from": "A", "to": "B", "type": "depends_on"},
            {"from": "B", "to": "C", "type": "causes"},
        ]

        # Stored vector matches the merged summary, not the deferred one
        assert mock_emb.embed.call_args[0][0] == ["Merged by human."]
        assert update_params["embedding"] == str([0.5] * 1024)
        # Content hash threaded through the deferred payload
        assert update_params["content_hash"] == "def456hash"


class TestResolveConflictHashCollision:
    """Overwrite/merge resolutions whose content_hash collides on the unique
    index (the new content got stored elsewhere while the conflict sat queued)
    report the stored memory as a duplicate instead of crashing."""

    @pytest.mark.asyncio
    async def test_overwrite_collision_reports_duplicate(self) -> None:
        from backend.service.memory import resolve_conflict

        existing_id = "11111111-1111-1111-1111-111111111111"
        winner = {"id": "33333333-3333-3333-3333-333333333333", "summary": "Winner summary."}
        select_result = MagicMock()
        select_result.fetchone.return_value = MagicMock(_mapping=winner)

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            IntegrityError("UPDATE ...", {}, Exception("unique_violation")),
            select_result,
        ]

        deferred = {
            "extracted": {"summary": "New summary.", "entities": [], "relations": []},
            "embedding": str([0.1] * 1024),
            "source_type": "conversation",
            "metadata": {"conflicts_with": existing_id},
            "content_hash": "abc123hash",
        }

        with patch(
            "backend.service.memory.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            result = await resolve_conflict("overwrite", existing_id, deferred)

        assert result["action"] == "duplicate"
        assert result["id"] == winner["id"]
        assert result["resolution"] == "overwrite"


class TestWriteMemoryIdempotency:
    """The content-hash gate in write_memory skips exact-duplicate ingestion."""

    @pytest.mark.asyncio
    async def test_duplicate_content_skips_extraction(self) -> None:
        """Same raw content → action='duplicate', extraction never runs."""
        from backend.service.memory import write_memory

        existing_row = {
            "id": "22222222-2222-2222-2222-222222222222",
            "summary": "Already stored summary.",
        }
        mock_find = AsyncMock(return_value=existing_row)
        mock_extract = AsyncMock()

        with (
            patch("backend.service.memory._find_by_content_hash", mock_find),
            patch("backend.service.memory.extract_memory", mock_extract),
            patch("backend.service.memory.get_session_factory"),
            patch("backend.service.memory.get_embedding_provider"),
        ):
            result = await write_memory(
                "identical content", source_type="doc", metadata={}
            )

        assert result["action"] == "duplicate"
        assert result["id"] == existing_row["id"]
        assert result["summary"] == existing_row["summary"]
        mock_extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fresh_content_proceeds_to_extraction(self) -> None:
        """Different raw content → not a duplicate, extraction runs."""
        from backend.service.memory import write_memory

        mock_find = AsyncMock(return_value=None)
        mock_extract = AsyncMock(
            return_value={"summary": "s", "entities": [], "relations": []}
        )
        mock_provider = AsyncMock()
        mock_provider.embed.return_value = [[0.1] * 1024]

        with (
            patch("backend.service.memory._find_by_content_hash", mock_find),
            patch("backend.service.memory.extract_memory", mock_extract),
            patch(
                "backend.service.memory.get_embedding_provider",
                return_value=mock_provider,
            ),
            patch(
                "backend.service.memory._find_similar",
                AsyncMock(return_value=(0.0, None)),
            ),
            patch(
                "backend.service.memory._insert_memory",
                AsyncMock(return_value={"id": "x", "action": "inserted"}),
            ),
        ):
            result = await write_memory(
                "fresh content", source_type="doc", metadata={}
            )

        assert result["action"] == "inserted"
        mock_extract.assert_awaited_once()
