"""Unit tests for the pending-conflict queue service.

DB access is mocked at the session-factory boundary; the shared
``resolve_conflict`` pipeline is mocked so these stay hermetic.
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


def _conflict_result() -> dict:
    """A write_memory ``action="conflict"`` result as produced by _mark_conflict."""
    return {
        "action": "conflict",
        "summary": "New conflicting summary",
        "existing_id": "11111111-1111-1111-1111-111111111111",
        "existing_summary": "Existing summary",
        "entities": [],
        "relations": [],
        "_deferred": {
            "extracted": {
                "summary": "New conflicting summary",
                "entities": [],
                "relations": [],
            },
            "embedding": "[0.1, 0.2]",
            "source_type": "ci_build",
            "metadata": {"conflicts_with": "11111111-1111-1111-1111-111111111111"},
            "content_hash": "abc123",
        },
    }


class TestPersistPendingConflict:
    @pytest.mark.asyncio
    async def test_persists_deferred_payload(self) -> None:
        """A conflict result is stored with the deferred payload for later resolve."""
        from backend.service.conflicts import persist_pending_conflict

        new_id = "22222222-2222-2222-2222-222222222222"
        created = datetime(2026, 8, 7, 10, 0, 0)
        insert_result = MagicMock()
        insert_result.fetchone.return_value = [new_id, created]

        mock_session = AsyncMock()
        mock_session.execute.return_value = insert_result

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            result = await persist_pending_conflict("ci", _conflict_result())

        assert result["id"] == new_id
        assert result["status"] == "pending"

        sql, params = mock_session.execute.call_args[0]
        assert params["source"] == "ci"
        assert params["source_type"] == "ci_build"
        assert params["existing_id"] == "11111111-1111-1111-1111-111111111111"
        assert params["new_summary"] == "New conflicting summary"
        # Deferred payload round-trips as JSON
        assert json.loads(params["deferred"])["content_hash"] == "abc123"

    @pytest.mark.asyncio
    async def test_missing_deferred_raises(self) -> None:
        """A malformed conflict result (no deferred payload) is rejected loudly."""
        from backend.service.conflicts import persist_pending_conflict

        bad = {"action": "conflict", "existing_id": "x", "summary": "s"}
        with pytest.raises(ValueError, match="missing"):
            await persist_pending_conflict("ci", bad)

    @pytest.mark.asyncio
    async def test_duplicate_insert_returns_existing_row(self) -> None:
        """A redelivered identical conflict reuses the queued row instead of stacking one."""
        from backend.service.conflicts import persist_pending_conflict

        existing_id = "22222222-2222-2222-2222-222222222222"
        created = datetime(2026, 8, 7, 11, 0, 0)

        # INSERT ... ON CONFLICT DO NOTHING rejects the duplicate (no row);
        # the follow-up SELECT returns the already-queued row.
        insert_result = MagicMock()
        insert_result.fetchone.return_value = None
        select_result = MagicMock()
        select_result.fetchone.return_value = [existing_id, created]

        mock_session = AsyncMock()
        mock_session.execute.side_effect = [insert_result, select_result]

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            result = await persist_pending_conflict("ci", _conflict_result())

        assert result["id"] == existing_id
        # The dedup SELECT matches on (existing_id, content_hash) from the deferred payload
        sql, params = mock_session.execute.call_args_list[1][0]
        assert params["existing_id"] == "11111111-1111-1111-1111-111111111111"
        assert params["content_hash"] == "abc123"
        assert "status = 'pending'" in str(sql)


class TestListPendingConflicts:
    @pytest.mark.asyncio
    async def test_returns_pending_rows(self) -> None:
        from backend.service.conflicts import list_pending_conflicts

        def _row(**kwargs):
            r = MagicMock()
            for k, v in kwargs.items():
                setattr(r, k, v)
            return r

        rows = [
            _row(
                id="c1",
                source="ci",
                source_type="ci_build",
                existing_id="11111111-1111-1111-1111-111111111111",
                existing_summary="Existing",
                new_summary="New",
                status="pending",
                resolution=None,
                created_at=datetime(2026, 8, 7, 10, 0, 0),
            )
        ]
        select_result = MagicMock()
        select_result.fetchall.return_value = rows
        mock_session = AsyncMock()
        mock_session.execute.return_value = select_result

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session),
        ):
            items = await list_pending_conflicts()

        assert len(items) == 1
        assert items[0]["id"] == "c1"
        assert items[0]["source"] == "ci"
        assert items[0]["created_at"].startswith("2026-08-07")


class TestResolvePendingConflict:
    @pytest.mark.asyncio
    async def test_applies_resolution_via_shared_pipeline(self) -> None:
        """Resolving delegates to memory.resolve_conflict and marks the row done."""
        from backend.service.conflicts import resolve_pending_conflict

        conflict_row = MagicMock()
        conflict_row.existing_id = "11111111-1111-1111-1111-111111111111"
        conflict_row.status = "pending"
        conflict_row.deferred = json.dumps({"content_hash": "abc123"})
        select_result = MagicMock()
        select_result.fetchone.return_value = conflict_row
        update_result = MagicMock()
        mock_session = AsyncMock()

        mock_resolve = AsyncMock(
            return_value={
                "id": "11111111-1111-1111-1111-111111111111",
                "action": "conflict_resolved",
                "resolution": "merge",
            }
        )

        with (
            patch(
                "backend.service.conflicts.get_session_factory",
                return_value=_make_session_factory(
                    mock_session, side_effect=[select_result, update_result]
                ),
            ),
            patch("backend.service.conflicts.resolve_conflict", mock_resolve),
        ):
            outcome = await resolve_pending_conflict("c1", "merge")

        assert outcome["id"] == "c1"
        assert outcome["resolution"] == "merge"
        assert outcome["outcome"]["action"] == "conflict_resolved"

        # Delegated to the shared pipeline with the stored deferred payload
        assert mock_resolve.await_args.args == (
            "merge",
            "11111111-1111-1111-1111-111111111111",
            {"content_hash": "abc123"},
        )
        # Row marked resolved (status is hardcoded in the SQL, resolution is a param)
        sql, params = mock_session.execute.call_args_list[1][0]
        assert params["id"] == "c1"
        assert params["resolution"] == "merge"
        assert "status = 'resolved'" in str(sql)

    @pytest.mark.asyncio
    async def test_unknown_resolution_raises(self) -> None:
        from backend.service.conflicts import resolve_pending_conflict

        with pytest.raises(ValueError, match="Unknown resolution"):
            await resolve_pending_conflict("c1", "nuke")

    @pytest.mark.asyncio
    async def test_already_resolved_raises(self) -> None:
        from backend.service.conflicts import resolve_pending_conflict

        conflict_row = MagicMock()
        conflict_row.existing_id = "11111111-1111-1111-1111-111111111111"
        conflict_row.status = "resolved"
        conflict_row.deferred = "{}"
        select_result = MagicMock()
        select_result.fetchone.return_value = conflict_row
        mock_session = AsyncMock()

        with patch(
            "backend.service.conflicts.get_session_factory",
            return_value=_make_session_factory(mock_session, side_effect=[select_result]),
        ):
            with pytest.raises(ValueError, match="already resolved"):
                await resolve_pending_conflict("c1", "merge")
