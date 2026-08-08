"""Tests for the pending-conflict HITL API — GET/POST /api/conflicts."""

import json
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from backend.db import get_session_factory
from backend.main import app


@pytest.fixture(autouse=True)
async def _ensure_tables() -> None:
    """Create tables (including pending_conflicts) before tests."""
    from backend.db.schema import init_db

    await init_db()


@pytest.fixture(autouse=True)
async def _clean_conflict_rows() -> None:
    """Isolate each test from prior runs (the test DB persists between runs)."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(text("DELETE FROM pending_conflicts"))
        await session.execute(text("DELETE FROM webhook_logs"))
        await session.commit()
    yield


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _deferred_payload(existing_id: str) -> dict:
    """A minimal _deferred payload for resolve_conflict(..., 'keep_existing')."""
    return {
        "extracted": {"summary": "New summary", "entities": [], "relations": []},
        "embedding": "[0.1]",
        "source_type": "ci_build",
        "metadata": {"conflicts_with": existing_id},
        "content_hash": "abc123",
    }


async def insert_pending_conflict(
    source: str = "ci",
    new_summary: str = "New summary",
    existing_summary: str = "Existing summary") -> str:
    """Insert a pending_conflicts row directly; return its UUID string."""
    existing_id = str(uuid4())
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                INSERT INTO pending_conflicts
                    (source, source_type, existing_id, existing_summary,
                     new_summary, deferred)
                VALUES (:source, :source_type, :existing_id, :existing_summary,
                        :new_summary, :deferred ::jsonb)
                RETURNING id
                """
            ),
            {
                "source": source,
                "source_type": "ci_build",
                "existing_id": existing_id,
                "existing_summary": existing_summary,
                "new_summary": new_summary,
                "deferred": json.dumps(_deferred_payload(existing_id)),
            })
        await session.commit()
        return str(result.fetchone()[0])


class TestListConflicts:
    @pytest.mark.asyncio
    async def test_empty_list(self, async_client: AsyncClient) -> None:
        resp = await async_client.get("/api/conflicts")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_lists_pending_conflicts(self, async_client: AsyncClient) -> None:
        conflict_id = await insert_pending_conflict()

        resp = await async_client.get("/api/conflicts")

        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["id"] == conflict_id
        assert items[0]["source"] == "ci"
        assert items[0]["status"] == "pending"
        assert items[0]["new_summary"] == "New summary"


class TestResolveConflict:
    @pytest.mark.asyncio
    async def test_resolve_keep_existing_marks_done(
        self, async_client: AsyncClient
    ) -> None:
        conflict_id = await insert_pending_conflict()

        resp = await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "keep_existing"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == conflict_id
        assert data["resolution"] == "keep_existing"

        # Row is marked resolved; no longer listed as pending
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("SELECT status FROM pending_conflicts WHERE id = :id"),
                {"id": conflict_id})
            assert result.fetchone()[0] == "resolved"

        list_resp = await async_client.get("/api/conflicts")
        assert list_resp.json() == []

    @pytest.mark.asyncio
    async def test_resolve_twice_returns_404(self, async_client: AsyncClient) -> None:
        conflict_id = await insert_pending_conflict()
        await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "keep_existing"})

        resp = await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "overwrite"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_resolution_returns_400(
        self, async_client: AsyncClient
    ) -> None:
        conflict_id = await insert_pending_conflict()
        resp = await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "nuke"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_conflict_returns_404(self, async_client: AsyncClient) -> None:
        resp = await async_client.post(
            "/api/conflicts/00000000-0000-0000-0000-000000000000/resolve",
            json={"resolution": "keep_existing"})
        assert resp.status_code == 404


class TestQueueDedup:
    """An identical conflict (webhook redelivery) is queued at most once."""

    @pytest.mark.asyncio
    async def test_identical_conflict_persisted_once(self) -> None:
        from backend.service.conflicts import persist_pending_conflict

        existing_id = str(uuid4())
        result = {
            "action": "conflict",
            "summary": "New summary",
            "existing_id": existing_id,
            "existing_summary": "Existing summary",
            "entities": [],
            "relations": [],
            "_deferred": {
                "extracted": {"summary": "New summary", "entities": [], "relations": []},
                "embedding": "[0.1]",
                "source_type": "ci_build",
                "metadata": {"conflicts_with": existing_id},
                "content_hash": "dedup-test-hash",
            },
        }

        first = await persist_pending_conflict("ci", result)
        second = await persist_pending_conflict("ci", result)

        # Same row returned, not a second queue entry.
        assert first["id"] == second["id"]

        session_factory = get_session_factory()
        async with session_factory() as session:
            row = await session.execute(
                text("SELECT count(*) FROM pending_conflicts WHERE status = 'pending'")
            )
            assert row.fetchone()[0] == 1


_TEST_EMBEDDING = "[{}]".format(",".join("0.1" for _ in range(1024)))


async def _insert_memory(
    memory_id: str,
    summary: str,
    content_hash: str | None = None,
    embedding: str = _TEST_EMBEDDING) -> None:
    """Insert a live memories row (patrol conflicts need real memory rows).

    Defaults to a per-row unique content_hash so parallel test rows never
    collide on ``uq_memories_content_hash_live`` (memories persist across
    tests; only pending_conflicts/webhook_logs are cleaned).
    """
    if content_hash is None:
        content_hash = f"hash-{memory_id}"
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                """\
                INSERT INTO memories (id, source_type, summary, entities, relations,
                                      embedding, content_hash)
                VALUES (:id, 'conversation', :summary, '[]', '[]', :embedding, :content_hash)
                """
            ),
            {
                "id": memory_id,
                "summary": summary,
                "embedding": embedding,
                "content_hash": content_hash,
            })
        await session.commit()


async def _insert_patrol_log(log_id: str) -> None:
    """Insert a patrol_logs row so the queue endpoint's existence check passes."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                """\
                INSERT INTO patrol_logs (id, patrol_type, trigger, status, started_at)
                VALUES (:id, 'weekly', 'cron', 'completed', now())
                """
            ),
            {"id": log_id})
        await session.commit()


class TestPatrolConflictRoutes:
    """Patrol contradictions — queue via patrol routes, resolve via conflicts."""

    @pytest.mark.asyncio
    async def test_queue_and_list_patrol_conflict(self, async_client: AsyncClient) -> None:
        """POST queue → row lands with conflict_type='patrol' + peer_id; GET lists it."""
        from backend.service.conflicts import persist_patrol_conflict

        a_id, b_id, log_id = str(uuid4()), str(uuid4()), str(uuid4())
        await _insert_memory(a_id, "Use PostgreSQL for storage")
        await _insert_memory(b_id, "Migrate away from PostgreSQL")
        await _insert_patrol_log(log_id)

        finding = {
            "memory_a_id": a_id,
            "memory_a_summary": "Use PostgreSQL for storage",
            "memory_b_id": b_id,
            "memory_b_summary": "Migrate away from PostgreSQL",
            "conflict_description": "opposite recommendations",
            "severity": "warning",
        }

        # Direct service call is the same path the route uses.
        queued = await persist_patrol_conflict(log_id, finding)
        assert queued["status"] == "queued"

        session_factory = get_session_factory()
        async with session_factory() as session:
            row = await session.execute(
                text(
                    """SELECT conflict_type, peer_id, existing_id FROM pending_conflicts
                       WHERE id = :id"""
                ),
                {"id": queued["id"]})
            stored = row.fetchone()
            assert stored.conflict_type == "patrol"
            assert str(stored.peer_id) == b_id
            assert str(stored.existing_id) == a_id

        resp = await async_client.get("/api/conflicts")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["conflict_type"] == "patrol"
        assert items[0]["peer_id"] == b_id

    @pytest.mark.asyncio
    async def test_queue_same_pair_once(self, async_client: AsyncClient) -> None:
        """Queuing the same pair twice yields one pending row (already_pending)."""
        from backend.service.conflicts import persist_patrol_conflict

        a_id, b_id, log_id = str(uuid4()), str(uuid4()), str(uuid4())
        await _insert_memory(a_id, "Use PostgreSQL for storage")
        await _insert_memory(b_id, "Migrate away from PostgreSQL")
        await _insert_patrol_log(log_id)

        finding = {
            "memory_a_id": a_id,
            "memory_a_summary": "Use PostgreSQL for storage",
            "memory_b_id": b_id,
            "memory_b_summary": "Migrate away from PostgreSQL",
        }

        first = await persist_patrol_conflict(log_id, finding)
        second = await persist_patrol_conflict(log_id, finding)
        assert first["id"] == second["id"]
        assert first["status"] == "queued"
        assert second["status"] == "already_pending"

        # Reverse the pair — still one pending row (unordered pair dedup).
        reversed_finding = {**finding, "memory_a_id": b_id, "memory_b_id": a_id}
        third = await persist_patrol_conflict(log_id, reversed_finding)
        assert third["id"] == first["id"]
        assert third["status"] == "already_pending"

        session_factory = get_session_factory()
        async with session_factory() as session:
            row = await session.execute(
                text("SELECT count(*) FROM pending_conflicts WHERE status = 'pending'")
            )
            assert row.fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_resolve_keep_existing_soft_deletes_peer(
        self, async_client: AsyncClient
    ) -> None:
        """Resolving a patrol conflict with keep_existing soft-deletes B, not A."""
        a_id, b_id, log_id = str(uuid4()), str(uuid4()), str(uuid4())
        await _insert_memory(a_id, "Use PostgreSQL for storage")
        await _insert_memory(b_id, "Migrate away from PostgreSQL")
        await _insert_patrol_log(log_id)

        from backend.service.conflicts import persist_patrol_conflict

        finding = {
            "memory_a_id": a_id,
            "memory_a_summary": "Use PostgreSQL for storage",
            "memory_b_id": b_id,
            "memory_b_summary": "Migrate away from PostgreSQL",
        }
        queued = await persist_patrol_conflict(log_id, finding)

        resp = await async_client.post(
            f"/api/conflicts/{queued['id']}/resolve",
            json={"resolution": "keep_existing"})
        assert resp.status_code == 200
        assert resp.json()["resolution"] == "keep_existing"

        session_factory = get_session_factory()
        async with session_factory() as session:
            rows = await session.execute(
                text(
                    """SELECT id, deleted_at FROM memories
                       WHERE id IN (:a, :b) ORDER BY id"""
                ),
                {"a": a_id, "b": b_id})
            by_id = {str(r.id): r.deleted_at for r in rows.fetchall()}
            assert by_id[a_id] is None  # A survives
            assert by_id[b_id] is not None  # B soft-deleted
            status = await session.execute(
                text("SELECT status FROM pending_conflicts WHERE id = :id"),
                {"id": queued["id"]})
            assert status.fetchone()[0] == "resolved"

        # Resolved → no longer pending, and the pair is not re-queueable.
        list_resp = await async_client.get("/api/conflicts")
        assert list_resp.json() == []

    @pytest.mark.asyncio
    async def test_resolve_keep_both_then_requeue_is_suppressed(self) -> None:
        """keep_both marks the pair arbitrated; re-queueing returns already_resolved."""
        from backend.service.conflicts import (
            persist_patrol_conflict,
            resolve_patrol_conflict)

        a_id, b_id, log_id = str(uuid4()), str(uuid4()), str(uuid4())
        await _insert_memory(a_id, "Use PostgreSQL for storage")
        await _insert_memory(b_id, "Migrate away from PostgreSQL")
        await _insert_patrol_log(log_id)

        finding = {
            "memory_a_id": a_id,
            "memory_a_summary": "Use PostgreSQL for storage",
            "memory_b_id": b_id,
            "memory_b_summary": "Migrate away from PostgreSQL",
        }
        queued = await persist_patrol_conflict(log_id, finding)

        session_factory = get_session_factory()
        async with session_factory() as session:
            row = await session.execute(
                text("SELECT existing_id, peer_id, deferred FROM pending_conflicts WHERE id = :id"),
                {"id": queued["id"]})
            conflict = row.fetchone()
            existing_id = str(conflict.existing_id)
            peer_id = str(conflict.peer_id)
            deferred = conflict.deferred
            if isinstance(deferred, str):
                import json

                deferred = json.loads(deferred)
            await session.execute(
                text(
                    """UPDATE pending_conflicts
                       SET status = 'resolved', resolution = 'keep_both', resolved_at = now()
                       WHERE id = :id"""
                ),
                {"id": queued["id"]})
            await session.commit()

        outcome = await resolve_patrol_conflict("keep_both", existing_id, peer_id, deferred)
        assert outcome["resolution"] == "keep_both"

        # Both memories still live → keep_both arbitration → re-queue suppressed.
        re_queued = await persist_patrol_conflict(log_id, finding)
        assert re_queued["status"] == "already_resolved"
        assert re_queued["queued"] is False


class TestPatrolConflictReopen:
    """POST /api/conflicts/{id}/reopen + resolved/type filtering."""

    @pytest.mark.asyncio
    async def test_reopen_resets_to_pending(self, async_client: AsyncClient) -> None:
        """A resolved patrol keep_both can be reopened for re-arbitration."""
        from backend.service.conflicts import persist_patrol_conflict

        a_id, b_id, log_id = str(uuid4()), str(uuid4()), str(uuid4())
        await _insert_memory(a_id, "Use PostgreSQL for storage")
        await _insert_memory(b_id, "Migrate away from PostgreSQL")
        await _insert_patrol_log(log_id)

        finding = {
            "memory_a_id": a_id,
            "memory_a_summary": "Use PostgreSQL for storage",
            "memory_b_id": b_id,
            "memory_b_summary": "Migrate away from PostgreSQL",
        }
        queued = await persist_patrol_conflict(log_id, finding)

        # Arbitrate with keep_both (both memories stay) → resolved row.
        resp = await async_client.post(
            f"/api/conflicts/{queued['id']}/resolve",
            json={"resolution": "keep_both"},
        )
        assert resp.status_code == 200

        # Reopen → back to pending.
        resp = await async_client.post(f"/api/conflicts/{queued['id']}/reopen")
        assert resp.status_code == 200
        assert resp.json() == {"id": queued["id"], "status": "pending"}

        session_factory = get_session_factory()
        async with session_factory() as session:
            row = await session.execute(
                text("SELECT status, resolution FROM pending_conflicts WHERE id = :id"),
                {"id": queued["id"]},
            )
            stored = row.fetchone()
            assert stored.status == "pending"
            assert stored.resolution is None

    @pytest.mark.asyncio
    async def test_reopen_rejects_ingestion_conflict(self, async_client: AsyncClient) -> None:
        """Reopening is refused for ingestion conflicts (store already mutated)."""
        conflict_id = await insert_pending_conflict()  # conflict_type defaults ingestion

        resp = await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "keep_existing"},
        )
        assert resp.status_code == 200

        resp = await async_client.post(f"/api/conflicts/{conflict_id}/reopen")
        assert resp.status_code == 409
        assert "not a patrol conflict" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_reopen_missing_conflict_returns_404(self, async_client: AsyncClient) -> None:
        resp = await async_client.post(
            "/api/conflicts/00000000-0000-0000-0000-000000000000/reopen"
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_filters_resolved_patrol(self, async_client: AsyncClient) -> None:
        """status=resolved + conflict_type=patrol lists only arbitrated patrol rows."""
        from backend.service.conflicts import persist_patrol_conflict

        a_id, b_id, log_id = str(uuid4()), str(uuid4()), str(uuid4())
        await _insert_memory(a_id, "Use PostgreSQL for storage")
        await _insert_memory(b_id, "Migrate away from PostgreSQL")
        await _insert_patrol_log(log_id)

        finding = {
            "memory_a_id": a_id,
            "memory_a_summary": "Use PostgreSQL for storage",
            "memory_b_id": b_id,
            "memory_b_summary": "Migrate away from PostgreSQL",
        }
        queued = await persist_patrol_conflict(log_id, finding)
        await async_client.post(
            f"/api/conflicts/{queued['id']}/resolve",
            json={"resolution": "keep_both"},
        )

        resp = await async_client.get(
            "/api/conflicts?status=resolved&conflict_type=patrol"
        )
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["id"] == queued["id"]
        assert items[0]["status"] == "resolved"
        assert items[0]["conflict_type"] == "patrol"

        # The default pending list no longer includes it.
        pending = await async_client.get("/api/conflicts")
        assert pending.json() == []
