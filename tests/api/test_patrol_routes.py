"""Tests for patrol API routes with mocked database."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


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
