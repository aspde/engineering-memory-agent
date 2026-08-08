"""Unit tests for the patrol-conflict queue service.

Patrol contradictions are two *already-stored* memories (A, B) queued for
HITL arbitration.  DB access is mocked at the session-factory boundary; the
LLM and embedding providers are mocked so these stay hermetic.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_session_factory(mock_session: AsyncMock, side_effect: list | None = None):
    """Mirror the shape of ``get_session_factory()`` (async context manager)."""
    if side_effect is not None:
        mock_session.execute.side_effect = side_effect

    mock_sess = AsyncMock()
    mock_sess.__aenter__ = AsyncMock(return_value=mock_session)
    mock_sess.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock()
    factory.return_value = mock_sess
    return factory


A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def _memory_row(
    mid: str,
    summary: str,
    embedding: str = "[0.1, 0.2]",
    content_hash: str | None = "hash-b",
    source_type: str = "conversation",
):
    """A memories row as returned by asyncpg (`.id` + `._mapping`)."""
    r = MagicMock()
    r.id = mid
    r._mapping = {
        "id": mid,
        "summary": summary,
        "entities": json.dumps([{"name": "postgres", "type": "tech"}]),
        "relations": json.dumps([{"from": "a", "to": "b", "type": "uses"}]),
        "embedding": embedding,
        "source_type": source_type,
        "meta": {},
        "content_hash": content_hash,
    }
    return r


def _patrol_finding() -> dict:
    """A patrol.weekly contradiction finding as produced by the LLM."""
    return {
        "memory_a_id": A_ID,
        "memory_a_summary": "Use PostgreSQL for storage",
        "memory_b_id": B_ID,
        "memory_b_summary": "Migrate away from PostgreSQL",
        "conflict_description": "opposite recommendations",
        "severity": "warning",
    }


def _patrol_deferred(extra_meta: dict | None = None) -> dict:
    """The deferred payload persist_patrol_conflict builds for (A, B)."""
    meta = {
        "conflicts_with": A_ID,
        "conflicting_summary": "Use PostgreSQL for storage",
        "peer_id": B_ID,
        "peer_summary": "Migrate away from PostgreSQL",
        "conflict_description": "opposite recommendations",
        "severity": "warning",
        "patrol_log_id": "log-1",
    }
    if extra_meta:
        meta.update(extra_meta)
    return {
        "kind": "patrol",
        "extracted": {
            "summary": "Migrate away from PostgreSQL",
            "entities": [{"name": "postgres", "type": "tech"}],
            "relations": [{"from": "a", "to": "b", "type": "uses"}],
        },
        "embedding": "[0.1, 0.2]",
        "source_type": "conversation",
        "metadata": meta,
        "content_hash": "hash-b",
    }


def _memory_query_result(*rows) -> MagicMock:
    result = MagicMock()
    result.fetchall.return_value = list(rows)
    return result


class TestPersistPatrolConflict:
    @pytest.mark.asyncio
    async def test_builds_deferred_payload(self) -> None:
        """Both memories are re-read; extracted/embedding/hash come from B."""
        from backend.service.conflicts import persist_patrol_conflict

        new_id = "22222222-2222-2222-2222-222222222222"
        created = datetime(2026, 8, 8, 10, 0, 0)
        insert_result = MagicMock()
        insert_result.fetchone.return_value = [new_id, created]
        resolved_result = MagicMock()
        resolved_result.fetchone.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            _memory_query_result(
                _memory_row(A_ID, "Use PostgreSQL for storage", content_hash="hash-a"),
                _memory_row(B_ID, "Migrate away from PostgreSQL", content_hash="hash-b"),
            ),
            resolved_result,
            insert_result,
        ]

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            result = await persist_patrol_conflict("log-1", _patrol_finding())

        assert result["id"] == new_id
        assert result["status"] == "queued"
        assert result["queued"] is True

        # INSERT params: A is the surviving side (existing_id), B the peer.
        sql, params = mock_session.execute.call_args_list[2][0]
        assert params["existing_id"] == A_ID
        assert params["peer_id"] == B_ID
        assert params["existing_summary"] == "Use PostgreSQL for storage"
        assert params["new_summary"] == "Migrate away from PostgreSQL"

        deferred = json.loads(params["deferred"])
        assert deferred["kind"] == "patrol"
        assert deferred["extracted"]["summary"] == "Migrate away from PostgreSQL"
        assert deferred["embedding"] == "[0.1, 0.2]"
        assert deferred["content_hash"] == "hash-b"
        assert deferred["metadata"]["peer_id"] == B_ID
        assert deferred["metadata"]["severity"] == "warning"
        assert deferred["metadata"]["patrol_log_id"] == "log-1"

    @pytest.mark.asyncio
    async def test_reverse_pair_dedups_to_one_pending(self) -> None:
        """Inserting (B, A) after (A, B) reuses the pending row — one queue entry."""
        from backend.service.conflicts import persist_patrol_conflict

        existing_id = "33333333-3333-3333-3333-333333333333"
        created = datetime(2026, 8, 8, 11, 0, 0)
        insert_result = MagicMock()
        insert_result.fetchone.return_value = None  # pair index rejected the insert
        select_result = MagicMock()
        select_result.fetchone.return_value = [existing_id, created]
        resolved_result = MagicMock()
        resolved_result.fetchone.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            _memory_query_result(
                _memory_row(A_ID, "Use PostgreSQL for storage"),
                _memory_row(B_ID, "Migrate away from PostgreSQL"),
            ),
            resolved_result,
            insert_result,
            select_result,
        ]

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            result = await persist_patrol_conflict(
                "log-1",
                {**_patrol_finding(), "memory_a_id": B_ID, "memory_b_id": A_ID},
            )

        assert result["id"] == existing_id
        assert result["status"] == "already_pending"
        assert result["queued"] is False

        # The fallback SELECT matches on the unordered pair (LEAST/GREATEST).
        sql, params = mock_session.execute.call_args_list[3][0]
        assert params["lo"] == min(A_ID, B_ID)
        assert params["hi"] == max(A_ID, B_ID)
        assert "LEAST(existing_id, peer_id)" in str(sql)

    @pytest.mark.asyncio
    async def test_already_arbitrated_skips_insert(self) -> None:
        """A resolved keep_both record suppresses re-queueing (both memories live)."""
        from backend.service.conflicts import persist_patrol_conflict

        resolved_id = "44444444-4444-4444-4444-444444444444"
        resolved_result = MagicMock()
        resolved_result.fetchone.return_value = [resolved_id]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            _memory_query_result(
                _memory_row(A_ID, "Use PostgreSQL for storage"),
                _memory_row(B_ID, "Migrate away from PostgreSQL"),
            ),
            resolved_result,
        ]

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            result = await persist_patrol_conflict("log-1", _patrol_finding())

        assert result["id"] == resolved_id
        assert result["status"] == "already_resolved"
        assert result["queued"] is False
        # Only the memories read + the resolved check ran — no INSERT.
        assert mock_session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_stale_finding_raises_when_memory_deleted(self) -> None:
        """A memory that is soft-deleted (or missing) makes the finding stale."""
        from backend.service.conflicts import persist_patrol_conflict

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            _memory_query_result(_memory_row(A_ID, "Use PostgreSQL for storage")),  # B missing
        ]

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            with pytest.raises(ValueError, match="not found or already deleted"):
                await persist_patrol_conflict("log-1", _patrol_finding())

    @pytest.mark.asyncio
    async def test_missing_ids_raise(self) -> None:
        from backend.service.conflicts import persist_patrol_conflict

        with pytest.raises(ValueError, match="memory_a_id or memory_b_id"):
            await persist_patrol_conflict("log-1", {"memory_a_id": A_ID})
        with pytest.raises(ValueError, match="must differ"):
            await persist_patrol_conflict("log-1", {"memory_a_id": A_ID, "memory_b_id": A_ID})


class TestResolvePatrolConflict:
    @pytest.mark.asyncio
    async def test_keep_existing_soft_deletes_peer_only(self) -> None:
        """keep_existing drops B; A is untouched."""
        from backend.service.conflicts import resolve_patrol_conflict

        mock_session = AsyncMock()
        mock_session.execute.return_value = MagicMock()

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            outcome = await resolve_patrol_conflict(
                "keep_existing", A_ID, B_ID, _patrol_deferred()
            )

        assert outcome["resolution"] == "keep_existing"
        # Call 0: surviving-memory (A) liveness check.
        sql0, params0 = mock_session.execute.call_args_list[0][0]
        assert "deleted_at" in str(sql0)
        assert params0["id"] == A_ID
        # Call 1: soft-delete B.
        sql1, params1 = mock_session.execute.call_args_list[1][0]
        assert "deleted_at" in str(sql1)
        assert params1["id"] == B_ID
        assert mock_session.execute.await_count == 2

    @pytest.mark.asyncio
    async def test_overwrite_soft_deletes_peer_then_updates_a(self) -> None:
        """A adopts B's content; B is soft-deleted first, in the same session."""
        from backend.service.conflicts import resolve_patrol_conflict

        update_result = MagicMock()
        update_result.rowcount = 1
        mock_session = AsyncMock()
        mock_session.execute.side_effect = [MagicMock(), MagicMock(), update_result]

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            outcome = await resolve_patrol_conflict(
                "overwrite", A_ID, B_ID, _patrol_deferred()
            )

        assert outcome["action"] == "conflict_resolved"
        # Call 0: surviving-memory (A) liveness check.
        sql0, params0 = mock_session.execute.call_args_list[0][0]
        assert "deleted_at" in str(sql0)
        assert params0["id"] == A_ID
        # Call 1: soft-delete B.
        sql1, params1 = mock_session.execute.call_args_list[1][0]
        assert "deleted_at" in str(sql1)
        assert params1["id"] == B_ID
        # Call 2: update A with B's content.
        sql2, params2 = mock_session.execute.call_args_list[2][0]
        assert params2["id"] == A_ID
        assert params2["summary"] == "Migrate away from PostgreSQL"
        assert params2["content_hash"] == "hash-b"

    @pytest.mark.asyncio
    async def test_merge_writes_merged_summary_and_soft_deletes_peer(self) -> None:
        """LLM merges both summaries; merged row lands on A, B is soft-deleted."""
        from backend.service.conflicts import resolve_patrol_conflict

        select_result = MagicMock()
        select_result.fetchone.return_value = ("Use PostgreSQL for storage", '[]', '[]')
        update_result = MagicMock()
        update_result.rowcount = 1

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [
            MagicMock(),  # call 0: surviving-memory (A) liveness check
            select_result,  # call 1: read A's current content
            MagicMock(),  # call 2: soft-delete B
            update_result,  # call 3: write merged summary to A
        ]

        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="PostgreSQL with a migration exit plan")
        mock_embed = AsyncMock()
        mock_embed.embed = AsyncMock(return_value=[[0.9, 0.8]])

        with (
            patch(
                "backend.service.conflicts.get_session_factory",
                return_value=_make_session_factory(mock_session),
            ),
            patch("backend.service.llm_service.get_llm_provider", return_value=mock_llm),
            patch("backend.service.conflicts.get_embedding_provider", return_value=mock_embed),
        ):
            outcome = await resolve_patrol_conflict(
                "merge", A_ID, B_ID, _patrol_deferred()
            )

        assert outcome["action"] == "conflict_resolved"
        # Call 1: read A's current content.
        assert mock_session.execute.call_args_list[1][0][1]["id"] == A_ID
        # Call 2: soft-delete B.
        sql2, params2 = mock_session.execute.call_args_list[2][0]
        assert "deleted_at" in str(sql2)
        assert params2["id"] == B_ID
        # Call 3: write merged summary to A.
        sql3, params3 = mock_session.execute.call_args_list[3][0]
        assert params3["id"] == A_ID
        assert params3["summary"] == "PostgreSQL with a migration exit plan"
        # Re-embedded merged text.
        assert mock_embed.embed.await_args.args == (["PostgreSQL with a migration exit plan"],)

    @pytest.mark.asyncio
    async def test_keep_both_touches_no_memories(self) -> None:
        """keep_both leaves both rows alone — arbitration is a keep-both record."""
        from backend.service.conflicts import resolve_patrol_conflict

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [MagicMock()]  # liveness check passes

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            outcome = await resolve_patrol_conflict(
                "keep_both", A_ID, B_ID, _patrol_deferred()
            )

        assert outcome["resolution"] == "keep_both"
        # Only the surviving-memory liveness check runs — no memory is touched.
        assert mock_session.execute.await_count == 1
        sql0, params0 = mock_session.execute.call_args_list[0][0]
        assert params0["id"] == A_ID

    @pytest.mark.asyncio
    async def test_keep_existing_refused_when_survivor_deleted(self) -> None:
        """A deleted while the conflict sat queued → refuse, not silently resolve."""
        from backend.service.conflicts import resolve_patrol_conflict

        liveness_result = MagicMock()
        liveness_result.fetchone.return_value = None  # A soft-deleted
        mock_session = AsyncMock()
        mock_session.execute.side_effect = [liveness_result]

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            with pytest.raises(ValueError, match="Surviving memory .* no longer exists"):
                await resolve_patrol_conflict(
                    "keep_existing", A_ID, B_ID, _patrol_deferred()
                )
        # B must not have been touched.
        assert mock_session.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_overwrite_refused_when_survivor_gone_at_write(self) -> None:
        """A deleted between the liveness check and the write → rowcount guard."""
        from backend.service.conflicts import resolve_patrol_conflict

        update_result = MagicMock()
        update_result.rowcount = 0  # A no longer live when the UPDATE runs
        mock_session = AsyncMock()
        mock_session.execute.side_effect = [MagicMock(), MagicMock(), update_result]

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            with pytest.raises(ValueError, match="Surviving memory .* no longer exists"):
                await resolve_patrol_conflict(
                    "overwrite", A_ID, B_ID, _patrol_deferred()
                )

    @pytest.mark.asyncio
    async def test_unknown_resolution_raises(self) -> None:
        from backend.service.conflicts import resolve_patrol_conflict

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [MagicMock()]  # liveness check passes

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            with pytest.raises(ValueError, match="Unknown resolution"):
                await resolve_patrol_conflict("nuke", A_ID, B_ID, _patrol_deferred())


class TestResolvePendingConflictDispatch:
    @pytest.mark.asyncio
    async def test_routes_patrol_to_patrol_pipeline(self) -> None:
        """conflict_type='patrol' dispatches to resolve_patrol_conflict."""
        from backend.service.conflicts import resolve_pending_conflict

        deferred = _patrol_deferred()
        conflict_row = MagicMock()
        conflict_row.existing_id = A_ID
        conflict_row.status = "pending"
        conflict_row.deferred = deferred
        conflict_row.conflict_type = "patrol"
        conflict_row.peer_id = B_ID

        select_result = MagicMock()
        select_result.fetchone.return_value = conflict_row
        update_result = MagicMock()
        mock_session = AsyncMock()

        mock_patrol = AsyncMock(
            return_value={"id": A_ID, "action": "conflict_resolved", "resolution": "keep_existing"}
        )
        mock_ingest = AsyncMock()

        with (
            patch(
                "backend.service.conflicts.get_session_factory",
                return_value=_make_session_factory(
                    mock_session, side_effect=[select_result, update_result]
                ),
            ),
            patch("backend.service.conflicts.resolve_patrol_conflict", mock_patrol),
            patch("backend.service.conflicts.resolve_conflict", mock_ingest),
        ):
            outcome = await resolve_pending_conflict("c1", "keep_existing")

        assert outcome["id"] == "c1"
        assert mock_patrol.await_args.args == ("keep_existing", A_ID, B_ID, deferred)
        mock_ingest.assert_not_awaited()
        # Row marked resolved by the shared UPDATE.
        sql, params = mock_session.execute.call_args_list[1][0]
        assert "status = 'resolved'" in str(sql)
        assert params["id"] == "c1"


class TestReopenPatrolConflict:
    @pytest.mark.asyncio
    async def test_reopens_resolved_patrol_conflict(self) -> None:
        """A resolved patrol conflict with live memories resets to pending."""
        from backend.service.conflicts import reopen_patrol_conflict

        conflict_row = MagicMock()
        conflict_row.existing_id = A_ID
        conflict_row.peer_id = B_ID
        conflict_row.status = "resolved"
        conflict_row.conflict_type = "patrol"
        select_result = MagicMock()
        select_result.fetchone.return_value = conflict_row
        survivor_result = MagicMock()
        survivor_result.fetchone.return_value = MagicMock()  # A is live
        peer_result = MagicMock()
        peer_result.fetchone.return_value = MagicMock()  # B is live
        update_result = MagicMock()
        mock_session = AsyncMock()

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(
                mock_session,
                side_effect=[select_result, survivor_result, peer_result, update_result],
            ),
        ):
            outcome = await reopen_patrol_conflict("c1")

        assert outcome == {"id": "c1", "status": "pending"}
        # UPDATE resets status/resolution/resolved_at.
        sql, params = mock_session.execute.call_args_list[3][0]
        assert "status = 'pending'" in str(sql)
        assert params["id"] == "c1"

    @pytest.mark.asyncio
    async def test_missing_conflict_raises_not_found(self) -> None:
        from backend.service.conflicts import (
            ConflictNotFoundError,
            reopen_patrol_conflict,
        )

        select_result = MagicMock()
        select_result.fetchone.return_value = None
        mock_session = AsyncMock()

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session, side_effect=[select_result]),
        ):
            with pytest.raises(ConflictNotFoundError, match="not found"):
                await reopen_patrol_conflict("missing")

    @pytest.mark.asyncio
    async def test_pending_conflict_not_reopenable(self) -> None:
        from backend.service.conflicts import reopen_patrol_conflict

        conflict_row = MagicMock()
        conflict_row.status = "pending"
        conflict_row.conflict_type = "patrol"
        conflict_row.peer_id = B_ID
        select_result = MagicMock()
        select_result.fetchone.return_value = conflict_row
        mock_session = AsyncMock()

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session, side_effect=[select_result]),
        ):
            with pytest.raises(ValueError, match="not resolved"):
                await reopen_patrol_conflict("c1")

    @pytest.mark.asyncio
    async def test_ingestion_conflict_not_reopenable(self) -> None:
        from backend.service.conflicts import reopen_patrol_conflict

        conflict_row = MagicMock()
        conflict_row.status = "resolved"
        conflict_row.conflict_type = "ingestion"
        conflict_row.peer_id = B_ID
        select_result = MagicMock()
        select_result.fetchone.return_value = conflict_row
        mock_session = AsyncMock()

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session, side_effect=[select_result]),
        ):
            with pytest.raises(ValueError, match="not a patrol conflict"):
                await reopen_patrol_conflict("c1")

    @pytest.mark.asyncio
    async def test_refuses_reopen_when_peer_deleted(self) -> None:
        from backend.service.conflicts import reopen_patrol_conflict

        conflict_row = MagicMock()
        conflict_row.status = "resolved"
        conflict_row.conflict_type = "patrol"
        conflict_row.existing_id = A_ID
        conflict_row.peer_id = B_ID
        select_result = MagicMock()
        select_result.fetchone.return_value = conflict_row
        survivor_result = MagicMock()
        survivor_result.fetchone.return_value = MagicMock()  # A is live
        peer_result = MagicMock()
        peer_result.fetchone.return_value = None  # B soft-deleted
        mock_session = AsyncMock()

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(
                mock_session, side_effect=[select_result, survivor_result, peer_result]
            ),
        ):
            with pytest.raises(ValueError, match="Losing memory .* no longer exists"):
                await reopen_patrol_conflict("c1")

    @pytest.mark.asyncio
    async def test_refuses_reopen_when_survivor_deleted(self) -> None:
        """A deleted separately while the conflict sat resolved → refuse."""
        from backend.service.conflicts import reopen_patrol_conflict

        conflict_row = MagicMock()
        conflict_row.status = "resolved"
        conflict_row.conflict_type = "patrol"
        conflict_row.existing_id = A_ID
        conflict_row.peer_id = B_ID
        select_result = MagicMock()
        select_result.fetchone.return_value = conflict_row
        survivor_result = MagicMock()
        survivor_result.fetchone.return_value = None  # A soft-deleted
        mock_session = AsyncMock()

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(
                mock_session, side_effect=[select_result, survivor_result]
            ),
        ):
            with pytest.raises(ValueError, match="Surviving memory .* no longer exists"):
                await reopen_patrol_conflict("c1")
