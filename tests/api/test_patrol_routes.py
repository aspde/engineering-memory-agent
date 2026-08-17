"""Tests for patrol API routes."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
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
    from datetime import datetime

    row = MagicMock()
    row.id = id
    row.patrol_type = patrol_type
    row.trigger = trigger
    row.status = status
    row.findings = findings
    row.dismissed_findings = dismissed_findings
    row.started_at = started_at or datetime.now(UTC)
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
        import backend.api.routes.patrol_routes as mod

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

            # The patrol runs in a tracked background task (see the GC-guard
            # regression test below).  Wait for it to finish under the mock so
            # it never escapes to the real run_patrol after the patch exits.
            deadline = time.monotonic() + 5.0
            while mod._patrol_tasks and time.monotonic() < deadline:
                time.sleep(0.01)

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

    @pytest.mark.asyncio
    async def test_background_task_is_held_by_strong_reference(
        self, async_client: AsyncClient
    ) -> None:
        """Manual patrol runs in a tracked task, not an unreferenced one.

        ``asyncio.create_task`` keeps no strong reference, so an unreferenced
        patrol can be garbage-collected at its first await while the agent is
        still running — the run would silently vanish.  The task must be held
        in a module-level set for its whole lifetime and dropped only after it
        finishes.
        """
        import backend.api.routes.patrol_routes as mod

        release = asyncio.Event()

        async def _blocking_run(**kwargs):
            # Hold the run open so the test can inspect the task set while the
            # patrol is genuinely in flight.
            await release.wait()
            return "test-patrol-id"

        # Keep the mock installed until the task completes — the route fires
        # the run in the background, so it must not outlive the patch.
        with patch(
            "backend.api.routes.patrol_routes.run_patrol",
            new_callable=AsyncMock,
            side_effect=_blocking_run,
        ):
            resp = await async_client.post(
                "/api/patrol/trigger",
                json={"patrol_type": "daily", "scope": "all"},
            )
            assert resp.status_code == 202

            # While the patrol is in flight, it must be tracked.
            assert mod._patrol_tasks, (
                "background patrol task must be held by a strong reference"
            )

            # Let the patrol finish; the task then drops its own reference.
            release.set()
            deadline = time.monotonic() + 5.0
            while mod._patrol_tasks and time.monotonic() < deadline:
                await asyncio.sleep(0.01)

        assert not mod._patrol_tasks, (
            "completed patrol task must be removed from the task set"
        )


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


class TestMergePatrolFinding:
    """POST /api/patrol/findings/{log_id}/merge"""

    A_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    B_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    @staticmethod
    def _exists_factory(log_exists: bool):
        """A session factory whose first query reports whether the log exists."""
        mock_session = AsyncMock()
        exists_result = MagicMock()
        exists_result.fetchone.return_value = MagicMock() if log_exists else None
        mock_session.execute.return_value = exists_result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)
        return factory

    def test_merge_success_maps_daily_pattern_fields(self, client):
        """matched/new memory ids map onto the shared pair shape and the pair
        is merged via the shared resolution pipeline, B as the peer."""
        deferred = {"kind": "patrol", "metadata": {"patrol_log_id": "log-1"}}
        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=self._exists_factory(True),
        ), patch(
            "backend.api.routes.patrol_routes.load_patrol_pair",
            new_callable=AsyncMock,
            return_value=(
                self.A_ID,
                self.B_ID,
                {"id": self.A_ID, "summary": "历史记忆"},
                {"id": self.B_ID, "summary": "新记忆"},
            ),
        ) as mock_load, patch(
            "backend.api.routes.patrol_routes.build_patrol_deferred",
            return_value=deferred,
        ) as mock_build, patch(
            "backend.api.routes.patrol_routes.resolve_conflict",
            new_callable=AsyncMock,
            return_value={
                "id": self.A_ID,
                "action": "conflict_resolved",
                "resolution": "merge",
            },
        ) as mock_resolve:
            resp = client.post(
                "/api/patrol/findings/log-1/merge",
                json={
                    "matched_memory_id": self.A_ID,
                    "new_memory_id": self.B_ID,
                    "reason": "duplicate of historical memory",
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["ok"] is True
            assert data["kept_id"] == self.A_ID
            assert data["merged_id"] == self.B_ID
            assert data["action"] == "conflict_resolved"

            # The daily-pattern field names are mapped onto the shared shape:
            # matched (historical) → survivor A, new → losing B, reason →
            # conflict_description.
            mapped = mock_load.await_args.args[0]
            assert mapped["memory_a_id"] == self.A_ID
            assert mapped["memory_b_id"] == self.B_ID
            assert mapped["conflict_description"] == "duplicate of historical memory"
            assert mapped["severity"] == "warning"  # daily findings default
            # The merge is applied through the same pipeline as arbitration.
            assert mock_resolve.await_args.args == ("merge", self.A_ID, deferred)
            assert mock_resolve.await_args.kwargs == {"peer_id": self.B_ID}

    def test_merge_missing_log_returns_404(self, client):
        """Merging on a nonexistent log returns 404."""
        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=self._exists_factory(False),
        ):
            resp = client.post("/api/patrol/findings/missing-log/merge", json={})
            assert resp.status_code == 404

    def test_merge_stale_finding_returns_409(self, client):
        """A stale finding (memory deleted) is a 409, not a crash."""
        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=self._exists_factory(True),
        ), patch(
            "backend.api.routes.patrol_routes.load_patrol_pair",
            new_callable=AsyncMock,
            side_effect=ValueError(
                "memory_a or memory_b not found or already deleted — patrol finding is stale"
            ),
        ):
            resp = client.post(
                "/api/patrol/findings/log-1/merge",
                json={"matched_memory_id": "a", "new_memory_id": "b"},
            )
            assert resp.status_code == 409

    def test_merge_resolution_failure_returns_409(self, client):
        """A resolution refusal (e.g. survivor deleted at write time) is a 409."""
        with patch(
            "backend.api.routes.patrol_routes.get_session_factory",
            return_value=self._exists_factory(True),
        ), patch(
            "backend.api.routes.patrol_routes.load_patrol_pair",
            new_callable=AsyncMock,
            return_value=(self.A_ID, self.B_ID, {}, {}),
        ), patch(
            "backend.api.routes.patrol_routes.build_patrol_deferred",
            return_value={},
        ), patch(
            "backend.api.routes.patrol_routes.resolve_conflict",
            new_callable=AsyncMock,
            side_effect=ValueError("Surviving memory ... no longer exists"),
        ):
            resp = client.post(
                "/api/patrol/findings/log-1/merge",
                json={"matched_memory_id": "a", "new_memory_id": "b"},
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


class TestGetLogErrorRealDB:
    """Real-DB: GET /logs/{id} surfaces the error reason of a failed run.

    A patrol that fails outside the validation path (provider error /
    timeout / cancellation) persists ``status='failed'`` with a NULL
    findings and its reason in the ``error`` column.  The detail endpoint
    must return that reason so the UI shows *why* instead of a bare 失败.
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

    async def _insert(self, *, status: str, error: str | None) -> str:
        from uuid import uuid4

        log_id = str(uuid4())
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text(
                    """INSERT INTO patrol_logs
                       (id, patrol_type, trigger, status, error)
                       VALUES (:id, 'daily', 'cron', :status, :error)"""
                ),
                {"id": log_id, "status": status, "error": error},
            )
            await session.commit()
        return log_id

    async def test_failed_log_returns_error_reason(
        self, async_client: AsyncClient
    ) -> None:
        """A failed run's error column is served back on the detail endpoint."""
        log_id = await self._insert(status="failed", error="LLM provider unavailable")

        resp = await async_client.get(f"/api/patrol/logs/{log_id}")
        assert resp.status_code == 200
        assert resp.json()["error"] == "LLM provider unavailable"

    async def test_completed_log_has_null_error(
        self, async_client: AsyncClient
    ) -> None:
        """A completed run reports a null error — not the empty string."""
        log_id = await self._insert(status="completed", error=None)

        resp = await async_client.get(f"/api/patrol/logs/{log_id}")
        assert resp.status_code == 200
        assert resp.json()["error"] is None
