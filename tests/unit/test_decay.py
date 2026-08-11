"""Tests for Ebbinghaus decay functions — pure math, no mocking needed."""

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.service.decay import compute_decay_factor


class TestComputeDecay:
    def test_fresh_memory_returns_one(self) -> None:
        # No recalled_at AND no created_at → no elapsed time → factor 1.0.
        assert compute_decay_factor(None, 0) == 1.0

    def test_never_recalled_but_old_decays(self) -> None:
        # recalled_at is NULL (never searched) but created_at is old — the
        # memory must decay by age, otherwise stale-but-unused entries would
        # keep full rank forever ("never recalled" ≠ "never forgets").
        old = datetime.now(timezone.utc) - timedelta(hours=240)
        factor = compute_decay_factor(None, 0, created_at=old)
        assert factor <= 0.1  # floored at _DECAY_FLOOR — decayed, not zero

    def test_recent_recall_near_one(self) -> None:
        just_now = datetime.now(timezone.utc) - timedelta(minutes=5)
        factor = compute_decay_factor(just_now, 3)
        # Recently recalled with moderate repetitions — should be > 0.95
        assert factor > 0.95

    def test_long_gap_decays(self) -> None:
        two_days_ago = datetime.now(timezone.utc) - timedelta(hours=240)
        factor = compute_decay_factor(two_days_ago, 0)
        # Long gap, first recall — should be heavily decayed (floored at
        # _DECAY_FLOOR, not zeroed, so the memory stays retrievable).
        assert factor <= 0.1

    def test_more_recalls_slows_decay(self) -> None:
        # Same elapsed time, different recall counts
        t = datetime.now(timezone.utc) - timedelta(hours=100)
        factor_few = compute_decay_factor(t, 1)
        factor_many = compute_decay_factor(t, 10)
        assert factor_many > factor_few

    def test_decay_factor_range(self) -> None:
        """Ensure all values are in [0, 1]."""
        for hours in [0, 1, 24, 168, 720]:
            for count in [0, 1, 5, 20]:
                t = datetime.now(timezone.utc) - timedelta(hours=hours)
                factor = compute_decay_factor(t, count)
                assert 0.0 <= factor <= 1.0

    def test_strength_never_zero(self) -> None:
        """Even never-recalled memories aren't instantly dead at t=0."""
        t = datetime.now(timezone.utc)
        factor = compute_decay_factor(t, 0)
        # e^(-0/S) = e^0 = 1.0 — but we return 1.0 early for recall_count == 0
        assert factor == 1.0


class TestSearchMemoriesLiveDecay:
    """search_memories computes decay at query time (real database).

    Fixes the snapshot-freeze: ranking must use the live decay factor
    (continuous in time), not the stored ``decay_factor`` that only moves
    when update_decay_batch runs.
    """

    _VEC = "[" + ",".join(["1.0"] + ["0.0"] * 1023) + "]"
    _QUERY = [1.0] + [0.0] * 1023

    @pytest.mark.asyncio
    async def test_live_decay_ranks_recently_recalled_first(self) -> None:
        """Two same-embedding memories — the recently recalled one wins.

        Identical unit vectors make cosine similarity tie, so ordering is
        decided by decay alone; the stale memory (30 days) must sort below
        the fresh one (1 minute) even though both are stored with the same
        ``decay_factor`` value.
        """
        from sqlalchemy import text

        from backend.db import get_session_factory
        from backend.db.schema import init_db
        from backend.service.decay import search_memories

        await init_db()
        async with get_session_factory()() as session:
            # Idempotent cleanup covering every hash this test-class inserts,
            # so the two tests pass in any execution order.
            await session.execute(text("DELETE FROM memories WHERE content_hash LIKE 'h-%'"))
            await session.execute(
                text(
                    "INSERT INTO memories (source_type, summary, embedding, "
                    "recalled_at, recall_count, content_hash) "
                    "VALUES ('test', 'stale', CAST(:vec AS vector), "
                    "now() - interval '30 days', 3, 'h-stale')"
                ),
                {"vec": self._VEC},
            )
            await session.execute(
                text(
                    "INSERT INTO memories (source_type, summary, embedding, "
                    "recalled_at, recall_count, content_hash) "
                    "VALUES ('test', 'fresh', CAST(:vec AS vector), "
                    "now() - interval '1 minute', 3, 'h-fresh')"
                ),
                {"vec": self._VEC},
            )
            await session.commit()

        results = await search_memories(self._QUERY, top_k=2, threshold=0.5)

        assert [r["summary"] for r in results] == ["fresh", "stale"]
        assert results[0]["decay_factor"] > results[1]["decay_factor"]
        assert 0.0 < results[0]["decay_factor"] <= 1.0

    @pytest.mark.asyncio
    async def test_never_recalled_fresh_memory_keeps_full_weight(self) -> None:
        """recalled_at IS NULL but created_at is fresh → decay_factor ≈ 1.0.

        A just-inserted memory that hasn't been searched yet has nothing to
        decay: the time base falls back to ``created_at`` (≈ now), so the
        factor stays near full retention.  (An *old* never-recalled memory
        decays by age — covered by ``test_never_recalled_old_memory_decays``.)
        """
        from sqlalchemy import text

        from backend.db import get_session_factory
        from backend.db.schema import init_db
        from backend.service.decay import search_memories

        await init_db()
        async with get_session_factory()() as session:
            await session.execute(text("DELETE FROM memories WHERE content_hash LIKE 'h-%'"))
            await session.execute(
                text(
                    "INSERT INTO memories (source_type, summary, embedding, "
                    "recalled_at, recall_count, content_hash) "
                    "VALUES ('test', 'null-recall', CAST(:vec AS vector), NULL, 0, 'h-null')"
                ),
                {"vec": self._VEC},
            )
            await session.commit()

        results = await search_memories(self._QUERY, top_k=1, threshold=0.5)

        assert results and results[0]["summary"] == "null-recall"
        assert results[0]["decay_factor"] > 0.99

    @pytest.mark.asyncio
    async def test_never_recalled_old_memory_decays(self) -> None:
        """recalled_at IS NULL and created_at is old → decay_factor < 1.0.

        The "natural sink" property: a memory written long ago that was never
        retrieved must not out-rank a fresh one just because it was never
        searched.  Time base falls back to ``created_at`` so age still decays
        it.
        """
        from sqlalchemy import text

        from backend.db import get_session_factory
        from backend.db.schema import init_db
        from backend.service.decay import search_memories

        await init_db()
        async with get_session_factory()() as session:
            await session.execute(text("DELETE FROM memories WHERE content_hash LIKE 'h-%'"))
            await session.execute(
                text(
                    "INSERT INTO memories (source_type, summary, embedding, "
                    "recalled_at, recall_count, content_hash, created_at) "
                    "VALUES ('test', 'old-null-recall', CAST(:vec AS vector), "
                    "NULL, 0, 'h-old-null', now() - interval '10 days')"
                ),
                {"vec": self._VEC},
            )
            await session.commit()

        results = await search_memories(self._QUERY, top_k=1, threshold=0.5)

        assert results and results[0]["summary"] == "old-null-recall"
        assert results[0]["decay_factor"] <= 0.1  # floored, not zeroed


    @pytest.mark.asyncio
    async def test_use_decay_false_ranks_by_similarity(self) -> None:
        """use_decay=False ranks by raw similarity: a stale-but-identical
        memory still ranks first, and no decay_factor/weighted_score keys are
        computed (the decay A/B control)."""
        from sqlalchemy import text

        from backend.db import get_session_factory
        from backend.db.schema import init_db
        from backend.service.decay import search_memories

        await init_db()
        async with get_session_factory()() as session:
            await session.execute(text("DELETE FROM memories WHERE content_hash LIKE 'h-%'"))
            # 30 days old, never recalled — decay-on would rank it ~0.
            await session.execute(
                text(
                    "INSERT INTO memories (source_type, summary, embedding, "
                    "recalled_at, recall_count, content_hash, created_at) "
                    "VALUES ('test', 'stale-similar', CAST(:vec AS vector), "
                    "NULL, 0, 'h-decay-off', now() - interval '30 days')"
                ),
                {"vec": self._VEC},
            )
            await session.commit()

        results = await search_memories(
            self._QUERY, top_k=1, threshold=0.5, use_decay=False
        )

        assert results and results[0]["summary"] == "stale-similar"
        assert results[0]["similarity"] == pytest.approx(1.0)
        # No decay artifacts on the no-decay path.
        assert "decay_factor" not in results[0]
        assert "weighted_score" not in results[0]


class TestUpdateDecayBatchLive:
    """update_decay_batch against the real DB — SQL formula matches the
    Python reference and ``id = ANY(:ids)`` accepts uuid objects."""

    @pytest.mark.asyncio
    async def test_atomic_batch_matches_python_formula(self) -> None:
        from sqlalchemy import text

        from backend.db import get_session_factory
        from backend.db.schema import init_db
        from backend.service.decay import compute_decay_factor, update_decay_batch

        await init_db()
        async with get_session_factory()() as session:
            await session.execute(
                text("DELETE FROM memories WHERE content_hash IN ('h-live-1', 'h-live-2')")
            )
            for h in ("h-live-1", "h-live-2"):
                await session.execute(
                    text(
                        "INSERT INTO memories (source_type, summary, recalled_at, "
                        "recall_count, content_hash) "
                        "VALUES ('test', :s, now() - interval '1 hour', 0, :h)"
                    ),
                    {"s": h, "h": h},
                )
            rows = await session.execute(
                text(
                    "SELECT id, recalled_at FROM memories "
                    "WHERE content_hash IN ('h-live-1', 'h-live-2') ORDER BY content_hash"
                )
            )
            pairs = rows.fetchall()  # [(uuid, datetime), ...]
            await session.commit()

        # uuid.UUID objects — exercises the ANY(:ids) parameter type.
        ids = [r[0] for r in pairs]
        # Reference the database clock so the Python mirror samples the same
        # NOW() the UPDATE will use — eliminates the Python-vs-DB wall-clock
        # skew that made the old tolerance-based comparison load-sensitive.
        async with get_session_factory()() as session:
            ref_now = (await session.execute(text("SELECT now()"))).scalar_one()
        # Post-update recall_count is 1 (0 + 1), matching the SQL formula.
        expected = {
            str(rid): round(compute_decay_factor(recalled_at, 1, now=ref_now), 4)
            for rid, recalled_at in pairs
        }

        out = await update_decay_batch(ids)

        # Only the time between the reference SELECT and the UPDATE's own
        # transaction timestamp separates the two samples now — sub-second
        # under test, comfortably inside the abs=1e-3 tolerance (each second
        # of gap shifts the factor by ~6.6e-5 at t≈1h, strength=3).  A real
        # formula regression (wrong strength / recall_count) shifts the
        # factor by ≥0.05 and still fails.
        assert out == pytest.approx(expected, abs=1e-3)
        # Both rows were bumped by the same single statement.
        async with get_session_factory()() as session:
            counts = await session.execute(
                text(
                    "SELECT recall_count FROM memories "
                    "WHERE content_hash IN ('h-live-1', 'h-live-2') ORDER BY content_hash"
                )
            )
            assert [c[0] for c in counts.fetchall()] == [1, 1]

    @pytest.mark.asyncio
    async def test_never_recalled_row_gets_first_recall(self) -> None:
        """A ``recalled_at IS NULL`` row is counted, not skipped.

        The old per-memory ``update_decay`` recorded a first recall for
        never-recalled rows (factor 1.0, elapsed 0); the batch must preserve
        that so a memory that exists but was never searched still starts
        accruing recall stats on its first hit.
        """
        from sqlalchemy import text

        from backend.db import get_session_factory
        from backend.db.schema import init_db
        from backend.service.decay import update_decay_batch

        await init_db()
        async with get_session_factory()() as session:
            await session.execute(
                text("DELETE FROM memories WHERE content_hash = 'h-first'")
            )
            await session.execute(
                text(
                    "INSERT INTO memories (source_type, summary, recalled_at, "
                    "recall_count, content_hash) "
                    "VALUES ('test', 'first-recall', NULL, 0, 'h-first')"
                )
            )
            rid = (
                await session.execute(
                    text(
                        "SELECT id FROM memories WHERE content_hash = 'h-first'"
                    )
                )
            ).scalar_one()
            await session.commit()

        out = await update_decay_batch([rid])

        # The row participates in the batch: count bumped to 1, factor 1.0.
        assert str(rid) in out
        assert out[str(rid)] == pytest.approx(1.0)
        async with get_session_factory()() as session:
            row = (
                await session.execute(
                    text(
                        "SELECT recall_count, recalled_at FROM memories "
                        "WHERE content_hash = 'h-first'"
                    )
                )
            ).fetchone()
            assert row[0] == 1
            assert row[1] is not None


class TestUpdateDecayBatch:
    """update_decay_batch — one atomic UPDATE for many memories (the N+1 fix)."""

    @pytest.mark.asyncio
    async def test_single_round_trip_for_all_ids(self) -> None:
        from backend.service import decay as mod

        rows = MagicMock()
        rows.fetchall.return_value = [("m1", 0.9210), ("m2", 0.9980)]
        mock_session = AsyncMock()
        mock_session.execute.return_value = rows
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        with patch.object(
            mod, "get_session_factory", return_value=MagicMock(return_value=ctx)
        ):
            out = await mod.update_decay_batch(["m1", "m2"])

        # Exactly one UPDATE ... RETURNING executes (previously N separate
        # sessions/commits — one per memory).
        mock_session.execute.assert_awaited_once()
        sql = str(mock_session.execute.call_args.args[0])
        assert "UPDATE memories" in sql and "ANY(:ids)" in sql
        assert out == {"m1": 0.9210, "m2": 0.9980}

    @pytest.mark.asyncio
    async def test_empty_input_is_noop(self) -> None:
        from backend.service import decay as mod

        assert await mod.update_decay_batch([]) == {}
