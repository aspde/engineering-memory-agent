"""Tests for patrol API routes."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from backend.db import get_session_factory


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient with mocked session factory."""
    from backend.main import app

    return TestClient(app)


def _mock_session_row(
    id="00000000-0000-0000-0000-000000000001",
    patrol_type="daily",
    trigger="cron",
    status="completed",
    findings=None,
    dismissed_findings=None,
    started_at=None,
    completed_at=None,
):
    """Create a mock Row object."""
    from datetime import datetime, timezone

    row = MagicMock()
    row.id = id
    row.patrol_type = patrol_type
    row.trigger = trigger
    row.status = status
    row.findings = findings
    row.dismissed_findings = dismissed_findings
    row.started_at = started_at or datetime.now(timezone.utc)
    row.completed_at = completed_at
    return row


def _make_mock_session(rows=None):
    """Create a mock async session factory."""
    mock_session = AsyncMock()
    if rows is not None:
        mock_result = MagicMock()
        mock_result.fetchone.return_value = rows[0] if len(rows) == 1 else None
        mock_result.scalar.return_value = len(rows) if isinstance(rows, list) else 0
        mock_session.execute.return_value = mock_result
        # For list queries, make execute return iterable rows
        mock_session.execute.side_effect = None

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=ctx)
    return factory


class TestManualTrigger:
    """POST /api/patrol/trigger"""

    def test_manual_patrol_trigger_returns_accepted(self, client):
        """Trigger endpoint returns 202 with accepted status."""
        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=_make_mock_session(),
        ), patch(
            "backend.api.routes.patrol_routes.run_patrol",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = "test-patrol-id"

            resp = client.post(
                "/api/patrol/trigger",
                json={"patrol_type": "daily", "scope": "all"},
            )
            assert resp.status_code == 202
            data = resp.json()
            assert data["status"] == "accepted"

    def test_trigger_invalid_type_returns_422(self, client):
        """Invalid patrol_type should return 422."""
        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=_make_mock_session(),
        ):
            resp = client.post(
                "/api/patrol/trigger",
                json={"patrol_type": "invalid_type"},
            )
            assert resp.status_code == 422


class TestListLogs:
    """GET /api/patrol/logs"""

    def test_list_logs_returns_paginated(self, client):
        """List endpoint returns items and total count."""
        mock_session = AsyncMock()
        # Mock count query
        count_result = MagicMock()
        count_result.scalar.return_value = 3
        # Mock data query
        row1 = _mock_session_row(
            id="11111111-1111-1111-1111-111111111111",
            findings=json.dumps({"pattern_matches": [{"x": 1}], "knowledge_gaps": []}),
        )

        mock_session.execute = AsyncMock()
        # First call = count, second call = data
        mock_session.execute.side_effect = [
            count_result,
            MagicMock(__iter__=lambda s: iter([row1])),
        ]

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=factory,
        ):
            resp = client.get("/api/patrol/logs?limit=10&offset=0")
            assert resp.status_code == 200
            data = resp.json()
            assert "items" in data
            assert "total" in data
            assert data["total"] == 3

    def test_list_logs_filter_by_type(self, client):
        """Can filter by patrol_type."""
        mock_session = AsyncMock()
        count_result = MagicMock()
        count_result.scalar.return_value = 1

        row = _mock_session_row(patrol_type="weekly")

        mock_session.execute = AsyncMock()
        mock_session.execute.side_effect = [
            count_result,
            MagicMock(__iter__=lambda s: iter([row])),
        ]

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=factory,
        ):
            resp = client.get("/api/patrol/logs?patrol_type=weekly")
            assert resp.status_code == 200


class TestGetLog:
    """GET /api/patrol/logs/{id}"""

    def test_get_log_not_found_returns_404(self, client):
        """Missing log returns 404."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=factory,
        ):
            resp = client.get("/api/patrol/logs/nonexistent-id")
            assert resp.status_code == 404


class TestDismissFinding:
    """POST /api/patrol/findings/{id}/dismiss"""

    def test_dismiss_nonexistent_log_returns_404(self, client):
        """Dismissing a finding on a nonexistent log returns 404."""
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=factory,
        ):
            resp = client.post(
                "/api/patrol/findings/nonexistent-id/dismiss",
                json={"finding_id": "00000000-0000-0000-0000-000000000099"},
            )
            assert resp.status_code == 404

    def test_dismiss_finding_succeeds(self, client):
        """Dismissing a finding on an existing log returns ok."""
        mock_session = AsyncMock()
        # First execute returns the log exists check
        exists_result = MagicMock()
        exists_result.fetchone.return_value = MagicMock()
        mock_session.execute.return_value = exists_result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=factory,
        ):
            resp = client.post(
                "/api/patrol/findings/00000000-0000-0000-0000-000000000001/dismiss",
                json={"finding_id": "00000000-0000-0000-0000-000000000099"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is True


class TestQueuePatrolConflict:
    """POST /api/patrol/findings/{log_id}/conflict"""

    def test_queue_contradiction_returns_conflict_id(self, client):
        """A contradiction finding is queued and its conflict id returned."""
        mock_session = AsyncMock()
        exists_result = MagicMock()
        exists_result.fetchone.return_value = MagicMock()  # log exists
        mock_session.execute.return_value = exists_result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=factory,
        ), patch(
            "backend.api.routes.patrol_routes.persist_patrol_conflict",
            new_callable=AsyncMock,
        ) as mock_persist:
            mock_persist.return_value = {
                "id": "conflict-1",
                "status": "queued",
                "queued": True,
            }
            resp = client.post(
                "/api/patrol/findings/log-1/conflict",
                json={
                    "memory_a_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "memory_b_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "severity": "warning",
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["conflict_id"] == "conflict-1"
            assert data["status"] == "queued"
            mock_persist.assert_awaited_once()
            finding = mock_persist.await_args.args[1]
            assert finding["memory_a_id"] == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    def test_queue_missing_log_returns_404(self, client):
        """Queueing on a nonexistent log returns 404."""
        mock_session = AsyncMock()
        exists_result = MagicMock()
        exists_result.fetchone.return_value = None  # log missing
        mock_session.execute.return_value = exists_result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=factory,
        ):
            resp = client.post(
                "/api/patrol/findings/missing-log/conflict",
                json={"memory_a_id": "a", "memory_b_id": "b"},
            )
            assert resp.status_code == 404

    def test_queue_stale_finding_returns_409(self, client):
        """A stale finding (memory deleted) is a 409, not a crash."""
        mock_session = AsyncMock()
        exists_result = MagicMock()
        exists_result.fetchone.return_value = MagicMock()  # log exists
        mock_session.execute.return_value = exists_result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=factory,
        ), patch(
            "backend.api.routes.patrol_routes.persist_patrol_conflict",
            new_callable=AsyncMock,
            side_effect=ValueError("memory_a not found or already deleted"),
        ):
            resp = client.post(
                "/api/patrol/findings/log-1/conflict",
                json={"memory_a_id": "a", "memory_b_id": "b"},
            )
            assert resp.status_code == 409


class TestDismissFindingRealDB:
    """Real-DB regression: dismissal is keyed by *finding key*, not a UUID.

    The LLM finding JSON carries no per-finding ``id``, so the frontend sends
    keys like ``contradictions-0``.  The endpoint previously cast the key to
    ``::uuid``, which Postgres rejects for non-UUID input → 500 on every real
    dismiss.  The ``dismissed_findings`` column is TEXT[]; a non-UUID key must
    append and persist cleanly.
    """

    @pytest.fixture(autouse=True)
    async def _ensure_patrol_table(self) -> None:
        from backend.db.schema import init_db

        await init_db()

    @pytest.fixture(autouse=True)
    async def _clean_patrol_rows(self) -> None:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(text("DELETE FROM patrol_logs"))
            await session.commit()
        yield

    @pytest.fixture
    async def patrol_log_id(self) -> str:
        from uuid import uuid4

        log_id = str(uuid4())
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text(
                    """INSERT INTO patrol_logs
                       (id, patrol_type, trigger, status, findings)
                       VALUES (:id, 'weekly', 'cron', 'completed', :findings)"""
                ),
                {
                    "id": log_id,
                    "findings": json.dumps(
                        {
                            "contradictions": [
                                {
                                    "memory_a_id": "a",
                                    "memory_b_id": "b",
                                    "conflict_description": "opposite",
                                }
                            ]
                        }
                    ),
                },
            )
            await session.commit()
        return log_id

    async def test_dismiss_with_non_uuid_finding_key_persists(
        self, async_client: AsyncClient, patrol_log_id: str
    ) -> None:
        """A contradiction finding (no ``id`` → ``contradictions-0``) dismisses."""
        resp = await async_client.post(
            f"/api/patrol/findings/{patrol_log_id}/dismiss",
            json={"finding_id": "contradictions-0"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # The key is persisted and re-served by GET.
        detail = await async_client.get(f"/api/patrol/logs/{patrol_log_id}")
        assert detail.status_code == 200
        assert detail.json()["dismissed_findings"] == ["contradictions-0"]

    async def test_dismiss_is_idempotent_for_non_uuid_key(
        self, async_client: AsyncClient, patrol_log_id: str
    ) -> None:
        """Re-dismissing the same finding key is a no-op, not a duplicate."""
        for _ in range(2):
            resp = await async_client.post(
                f"/api/patrol/findings/{patrol_log_id}/dismiss",
                json={"finding_id": "pattern_matches-1"},
            )
            assert resp.status_code == 200

        detail = await async_client.get(f"/api/patrol/logs/{patrol_log_id}")
        assert detail.json()["dismissed_findings"] == ["pattern_matches-1"]
