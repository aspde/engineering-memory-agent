"""Tests for the memory write-path's correctness-critical structured call.

``_detect_conflict`` is correctness-critical: after retries are exhausted it
must raise (so the caller decides how to degrade) rather than silently assume
no conflict.  ``write_memory`` catches that error and degrades to a supplement
write — the content is kept without asserting a contradiction.
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
            "meta": json.dumps({"thread_id": "t-1", "commit_id": "c-1"}),
            "content_hash": "old-hash-abc",
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

        # Existing meta preserved (provenance) + superseded hash recorded for
        # the idempotency gate — merge must not wipe them (see prior_hashes).
        assert json.loads(params["meta"]) == {
            "thread_id": "t-1",
            "commit_id": "c-1",
            "prior_hashes": ["old-hash-abc"],
        }

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
            json.dumps({"thread_id": "t-1", "commit_id": "c-1"}),  # existing meta
            "old-hash-abc",
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
            patch("backend.service.memory._schedule_normalization"),
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
        # Existing meta preserved + conflict tag + superseded hash recorded
        # (same merge semantics as _merge_memory — not a wipe).
        assert json.loads(update_params["meta"]) == {
            "thread_id": "t-1",
            "commit_id": "c-1",
            "conflicts_with": existing_id,
            "prior_hashes": ["old-hash-abc"],
        }


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
        # resolve_conflict(overwrite) reads the existing meta + content_hash
        # before the UPDATE; then the UPDATE collides on the unique index.
        meta_select_result = MagicMock()
        meta_select_result.fetchone.return_value = ("{}", "old-hash-abc")

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            meta_select_result,
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
                AsyncMock(return_value=[]),
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


class TestInsertMemoryOnConflict:
    @pytest.mark.asyncio
    async def test_on_conflict_names_partial_index_predicate(self) -> None:
        """The INSERT's ON CONFLICT clause must name the partial index predicate.

        ``uq_memories_content_hash_live`` is a *partial* unique index
        (``WHERE deleted_at IS NULL``).  A bare ``ON CONFLICT (content_hash)``
        cannot be inferred against it — PostgreSQL raises "no unique or
        exclusion constraint matching the ON CONFLICT specification" on every
        real insert.  This regression was masked by fully-mocked sessions
        (the mock never parses the SQL), so it is locked here by asserting
        the exact clause.
        """
        from backend.service.memory import _insert_memory

        mock_session = AsyncMock()
        # execute() is an AsyncMock child, so its return_value would also be an
        # AsyncMock (fetchone() → coroutine).  Pin a plain MagicMock for the
        # result row so fetchone() returns the tuple synchronously.
        mock_session.execute.return_value = MagicMock()
        mock_session.execute.return_value.fetchone.return_value = ("new-id",)

        with (
            patch(
                "backend.service.memory.get_session_factory",
                return_value=_make_session_factory(mock_session),
            ),
            patch("backend.service.memory._schedule_normalization"),
        ):
            result = await _insert_memory(
                {"summary": "s", "entities": [], "relations": []},
                [0.1] * 1024,
                "doc",
                {},
                "abc123hash",
            )

        assert result["id"] == "new-id"
        sql, _ = mock_session.execute.call_args[0]
        assert (
            "ON CONFLICT (content_hash) WHERE deleted_at IS NULL DO NOTHING"
            in str(sql)
        )


class TestNormalizationBackpressure:
    """Fire-and-forget entity normalisation is bounded by a semaphore.

    Without the cap, ingesting a batch of memories spawns one background
    task per memory, each opening embedding + vector-search + LLM calls —
    hundreds of concurrent provider calls against a small connection pool.
    """

    @pytest.mark.asyncio
    async def test_concurrent_runs_are_bounded(self, monkeypatch) -> None:
        import asyncio

        from backend.service import memory as mod

        active = 0
        peak = 0
        done: list[str] = []

        async def fake_normalize(memory_id, entities):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.05)
            active -= 1
            done.append(memory_id)

        monkeypatch.setattr(mod, "normalize_entities", fake_normalize)
        # Fresh semaphore so this test doesn't inherit stale state from other
        # runs; the cap itself stays the module's configured constant.
        monkeypatch.setattr(
            mod,
            "_normalization_semaphore",
            asyncio.Semaphore(mod._NORMALIZATION_MAX_CONCURRENCY),
        )

        for i in range(8):
            mod._schedule_normalization(
                f"m-{i}", [{"name": "X", "type": "concept"}]
            )

        # Wait for all 8 queued tasks to finish (4 run at a time).
        await asyncio.sleep(0.5)

        # Concurrency never exceeds the cap, and excess runs queue rather
        # than being dropped.
        assert peak <= mod._NORMALIZATION_MAX_CONCURRENCY
        assert sorted(done) == [f"m-{i}" for i in range(8)]


class TestWriteMemoryConflictFailSafe:
    """write_memory must not drop content when conflict detection fails.

    Previously LLMStructuredError escaped write_memory, so ingestion (returns
    None on error) and auto-memory (swallows errors) silently lost the commit
    or turn.  The fix degrades to a supplement write — content preserved, no
    contradiction assumed — and surfaces the degradation via
    ``record_structured_failure("conflict_detection")``.
    """

    @pytest.mark.asyncio
    async def test_conflict_detection_failure_writes_supplement(self) -> None:
        from backend.service.memory import write_memory

        existing = {
            "id": "11111111-1111-1111-1111-111111111111",
            "summary": "Existing summary.",
        }
        similar = [(0.8, existing)]  # CONFLICT_CHECK range [0.72, 0.85)

        async def _fake_find_similar(embedding, session_factory):
            return similar

        async def _raise_structured(*args, **kwargs):
            raise LLMStructuredError("no schema-valid JSON after retries")

        captured = {}

        async def _fake_supplement(existing, extracted, embedding, source_type, metadata, content_hash):
            captured["existing_id"] = str(existing["id"])
            captured["metadata"] = metadata
            return {"id": "new-id", "action": "inserted", "summary": "New summary."}

        failures: list[str] = []

        def _record_failure(scenario: str) -> None:
            failures.append(scenario)

        with (
            patch("backend.service.memory._find_by_content_hash", AsyncMock(return_value=None)),
            patch(
                "backend.service.memory.extract_memory",
                AsyncMock(return_value={"summary": "New summary.", "entities": [], "relations": []}),
            ),
            patch("backend.service.memory.get_embedding_provider", return_value=AsyncMock()),
            patch("backend.service.memory._find_similar", _fake_find_similar),
            patch("backend.service.memory._detect_conflict", _raise_structured),
            patch("backend.service.memory._supplement_memory", _fake_supplement),
            patch("backend.service.memory.record_structured_failure", _record_failure),
        ):
            result = await write_memory("new content", source_type="conversation")

        # Degraded to a supplement write (action shape stays compatible for
        # the agent tool / resolve_conflict consumers).
        assert result["action"] == "inserted"
        assert captured["existing_id"] == existing["id"]
        assert failures == ["conflict_detection"]


class TestWriteMemoryFanOut:
    """A new memory similar to several existing ones records all of them.

    Previously ``_find_similar`` returned a single candidate (LIMIT 1), so a
    new memory 0.93-similar to A and B only ever merged with A; B never heard
    of it.  Now top-N candidates are graded and the lower ones are carried on
    the new memory's meta as ``supplements``.
    """

    @pytest.mark.asyncio
    async def test_supplement_branch_fans_out_other_candidates(self) -> None:
        from backend.service.memory import write_memory

        a = {"id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "summary": "A"}
        b = {"id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "summary": "B"}
        similar = [(0.70, a), (0.65, b)]  # below conflict, both ≥ supplement

        async def _fake_find_similar(embedding, session_factory):
            return similar

        captured = {}

        async def _fake_supplement(existing, extracted, embedding, source_type, metadata, content_hash):
            captured["existing_id"] = str(existing["id"])
            captured["metadata"] = metadata
            return {"id": "new-id", "action": "inserted", "summary": "New summary."}

        with (
            patch("backend.service.memory._find_by_content_hash", AsyncMock(return_value=None)),
            patch(
                "backend.service.memory.extract_memory",
                AsyncMock(return_value={"summary": "New summary.", "entities": [], "relations": []}),
            ),
            patch("backend.service.memory.get_embedding_provider", return_value=AsyncMock()),
            patch("backend.service.memory._find_similar", _fake_find_similar),
            patch("backend.service.memory._supplement_memory", _fake_supplement),
        ):
            result = await write_memory("new content", source_type="conversation")

        assert result["action"] == "inserted"
        assert captured["existing_id"] == a["id"]
        # The second candidate rides along on the new memory's meta.
        assert captured["metadata"]["supplements"] == [b["id"]]

    @pytest.mark.asyncio
    async def test_merge_branch_carries_other_candidates_on_meta(self) -> None:
        from backend.service.memory import write_memory

        a = {"id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "summary": "A"}
        b = {"id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "summary": "B"}
        similar = [(0.95, a), (0.70, b)]  # merge with A, supplement B

        async def _fake_find_similar(embedding, session_factory):
            return similar

        captured = {}

        async def _fake_merge(existing, extracted, embedding, source_type, metadata, content_hash):
            captured["existing_id"] = str(existing["id"])
            captured["metadata"] = metadata
            return {"id": a["id"], "action": "merged", "summary": "Merged summary."}

        with (
            patch("backend.service.memory._find_by_content_hash", AsyncMock(return_value=None)),
            patch(
                "backend.service.memory.extract_memory",
                AsyncMock(return_value={"summary": "New summary.", "entities": [], "relations": []}),
            ),
            patch("backend.service.memory.get_embedding_provider", return_value=AsyncMock()),
            patch("backend.service.memory._find_similar", _fake_find_similar),
            patch("backend.service.memory._merge_memory", _fake_merge),
        ):
            result = await write_memory("new content", source_type="conversation")

        assert result["action"] == "merged"
        assert captured["existing_id"] == a["id"]
        # The merge writes B's id onto the merged memory's meta (fan-out).
        assert captured["metadata"]["supplements"] == [b["id"]]

    @pytest.mark.asyncio
    async def test_conflict_band_candidate_no_longer_flagged_unchecked(self) -> None:
        """A supplement candidate sitting in the conflict-detection band is no
        longer tagged with the dead ``supplements_unchecked`` meta marker.

        Regression: that field was written but never read anywhere (backend +
        tests) — a 2nd/3rd near-relative was flagged-and-ignored instead of
        vetted or honestly documented.  Only the *closest* match runs
        ``_detect_conflict``; band candidates ride along as supplements and the
        gap is documented in code, not silently flagged.
        """
        from backend.service.memory import write_memory

        a = {"id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "summary": "A"}
        b = {"id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "summary": "B"}
        similar = [(0.70, a), (0.80, b)]  # closest below band; b in [0.72, 0.85)

        async def _fake_find_similar(embedding, session_factory):
            return similar

        captured = {}

        async def _fake_supplement(existing, extracted, embedding, source_type, metadata, content_hash):
            captured["existing_id"] = str(existing["id"])
            captured["metadata"] = metadata
            return {"id": "new-id", "action": "inserted", "summary": "New summary."}

        with (
            patch("backend.service.memory._find_by_content_hash", AsyncMock(return_value=None)),
            patch(
                "backend.service.memory.extract_memory",
                AsyncMock(return_value={"summary": "New summary.", "entities": [], "relations": []}),
            ),
            patch("backend.service.memory.get_embedding_provider", return_value=AsyncMock()),
            patch("backend.service.memory._find_similar", _fake_find_similar),
            patch("backend.service.memory._supplement_memory", _fake_supplement),
        ):
            result = await write_memory("new content", source_type="conversation")

        assert result["action"] == "inserted"
        assert captured["existing_id"] == a["id"]
        # The band candidate still rides along as a supplement…
        assert captured["metadata"]["supplements"] == [b["id"]]
        # …but the dead marker is gone from the written meta.
        assert "supplements_unchecked" not in captured["metadata"]


class TestSupplementMemoryFanOut:
    @pytest.mark.asyncio
    async def test_supplements_list_keeps_primary_first(self) -> None:
        from backend.service.memory import _supplement_memory

        existing = {"id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "summary": "A"}
        inserted = {}

        async def _fake_insert(extracted, embedding, source_type, metadata, content_hash):
            inserted["meta"] = metadata
            return {"id": "new-id", "action": "inserted", "summary": "s"}

        with patch("backend.service.memory._insert_memory", _fake_insert):
            await _supplement_memory(
                existing,
                {"summary": "s", "entities": [], "relations": []},
                [0.1] * 8,
                "doc",
                {"supplements": ["bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"]},
                "hash",
            )

        # Primary parent first, then the fan-out candidate.
        assert inserted["meta"]["supplements"] == [
            existing["id"],
            "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        ]
        assert inserted["meta"]["parent_summary"] == "A"


class TestPriorHashes:
    """Rewriting a memory's content_hash must keep the old hash reachable, so
    re-ingesting the original content is still gated as a duplicate instead of
    re-running extract → grade → merge (version drift)."""

    def test_record_prior_hash_is_idempotent(self) -> None:
        from backend.service.memory import _record_prior_hash

        meta = _record_prior_hash({}, "hash-1")
        assert meta["prior_hashes"] == ["hash-1"]
        assert _record_prior_hash(meta, "hash-1")["prior_hashes"] == ["hash-1"]
        assert _record_prior_hash(meta, "hash-2")["prior_hashes"] == ["hash-1", "hash-2"]

    def test_record_prior_hash_ignores_missing(self) -> None:
        from backend.service.memory import _record_prior_hash

        assert _record_prior_hash({"a": 1}, None) == {"a": 1}

    @pytest.mark.asyncio
    async def test_find_by_content_hash_checks_prior_hashes(self) -> None:
        from backend.service.memory import _find_by_content_hash

        row = MagicMock()
        row._mapping = {"id": "x", "summary": "s"}
        exec_result = MagicMock()
        exec_result.fetchone.return_value = row
        mock_session = AsyncMock()
        mock_session.execute.return_value = exec_result
        factory = _make_session_factory(mock_session)

        result = await _find_by_content_hash("old-hash", factory)

        assert result == {"id": "x", "summary": "s"}
        sql, params = mock_session.execute.call_args[0]
        assert "meta->'prior_hashes' ? :hash" in str(sql)
        assert params["hash"] == "old-hash"


class TestResolveConflictOverwrite:
    @pytest.mark.asyncio
    async def test_overwrite_preserves_meta_and_schedules_normalization(self) -> None:
        from backend.service.memory import resolve_conflict

        existing_id = "11111111-1111-1111-1111-111111111111"
        meta_select_result = MagicMock()
        meta_select_result.fetchone.return_value = (
            json.dumps({"thread_id": "t-1", "commit_id": "c-1"}),
            "old-hash-abc",
        )
        mock_session = AsyncMock()
        mock_session.execute.side_effect = [meta_select_result, MagicMock()]

        deferred = {
            "extracted": {
                "summary": "New summary.",
                "entities": [{"name": "B", "type": "concept"}],
                "relations": [],
            },
            "embedding": str([0.1] * 1024),
            "source_type": "conversation",
            "metadata": {"conflicts_with": existing_id},
            "content_hash": "abc123hash",
        }

        scheduled = {}

        def _fake_schedule(memory_id, entities):
            scheduled["id"] = memory_id
            scheduled["entities"] = entities

        with (
            patch(
                "backend.service.memory.get_session_factory",
                return_value=_make_session_factory(mock_session),
            ),
            patch("backend.service.memory._schedule_normalization", _fake_schedule),
        ):
            result = await resolve_conflict("overwrite", existing_id, deferred)

        assert result["action"] == "conflict_resolved"
        sql, params = mock_session.execute.call_args_list[1][0]
        stored_meta = json.loads(params["meta"])
        # Existing provenance preserved + conflict tag + superseded hash kept.
        assert stored_meta["thread_id"] == "t-1"
        assert stored_meta["commit_id"] == "c-1"
        assert stored_meta["conflicts_with"] == existing_id
        assert stored_meta["prior_hashes"] == ["old-hash-abc"]
        assert params["content_hash"] == "abc123hash"
        # The overwritten content's entities are written and normalisation is
        # scheduled for the memory_entities link table.
        assert json.loads(params["entities"]) == [{"name": "B", "type": "concept"}]
        assert scheduled["id"] == existing_id
        assert scheduled["entities"] == [{"name": "B", "type": "concept"}]
