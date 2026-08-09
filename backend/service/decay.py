"""Ebbinghaus forgetting curve — decay weighting for memory retrieval.

On each recall, the decay factor is updated based on time elapsed since the
last recall.  More frequent recalls slow the decay; long gaps accelerate it.

Formula (simplified Ebbinghaus):
    R = e^(-t / S)
where:
    t = hours since last recall
    S = relative strength = 1 + (recall_count + 1) * 2
    recall_count = the stored count BEFORE this recall; the ``+ 1`` yields
                   the count AFTER the recall — all three call sites
                   (compute_decay_factor / update_decay_batch /
                   search_memories) share this post-recall convention.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory

logger = logging.getLogger(__name__)


def compute_decay_factor(
    recalled_at: datetime | None,
    recall_count: int,
    *,
    now: datetime | None = None,
) -> float:
    """Compute the current decay factor for a memory.

    A value of 1.0 means full retention; 0.0 means fully forgotten.

    Python mirror of the SQL formula inlined in :func:`update_decay_batch`
    and :func:`search_memories` — keep the three in sync.  ``recall_count``
    must be the count AFTER this recall (the post-recall convention: the SQL
    reads the stored pre-recall value and adds 1, so the mirror takes the
    incremented value directly).  ``now`` is an optional reference instant
    (default: the current wall clock); tests pass the database's ``NOW()``
    so the mirror is compared against the SQL formula at the *same* instant
    instead of a skewed local clock.

    Clock assumption: the application clock is assumed to be consistent with
    the database clock.  As a defensive measure, elapsed time is clamped to
    zero when it would come out negative (app clock lagging the DB), so a
    skewed wall clock can never push the factor above 1.0 or overflow
    ``math.exp``.  The decay formula itself is unchanged.
    """
    if recalled_at is None:
        return 1.0  # Just inserted, no time has passed yet

    if now is None:
        now = datetime.now(timezone.utc)
    hours_elapsed = (now - recalled_at).total_seconds() / 3600.0
    # Defensive: the application clock is assumed to be consistent with the
    # database clock (see search_memories).  If the app clock lags the DB's,
    # a just-written ``recalled_at`` looks like the future and elapsed goes
    # negative — the factor would exceed 1.0 (inflating rank) and could even
    # overflow math.exp.  Clamp to zero so a future-dated recall reads as
    # "just now" (factor 1.0), never above full retention.  This clamps the
    # elapsed *input* only; the decay formula itself is unchanged.
    hours_elapsed = max(0.0, hours_elapsed)
    strength = 1.0 + recall_count * 2.0
    decay = math.exp(-hours_elapsed / strength)
    return round(decay, 4)


async def update_decay_batch(memory_ids: list[Any]) -> dict[str, float]:
    """Atomically bump recall stats for many memories, in one round-trip.

    Replaces the old per-memory ``update_decay`` loop: N independent
    sessions/commits on every memory search (the N+1 write), and a non-atomic
    read-modify-write that lost increments under concurrent recalls.  The
    whole update is one ``UPDATE ... RETURNING`` where the new factor is
    computed in SQL from the stored ``recalled_at`` — same formula as
    :func:`compute_decay_factor` (strength = 1 + (recall_count + 1) * 2,
    rounded to 4 dp).  ``recall_count`` on the RHS is the *stored* pre-recall
    value, so ``+ 1`` yields the post-recall count that the statement is
    writing back — the three call sites share this post-recall convention.
    ``NOW()`` is the transaction timestamp, constant within the statement,
    so the factor is consistent across all updated rows.

    Returns ``{str(memory_id): new_decay_factor}``.  Ids whose row is missing
    are absent from the result — callers fall back to the factor they already
    hold.  A never-recalled row (``recalled_at IS NULL``) is still counted:
    ``COALESCE`` makes its elapsed time 0, so the first recall is recorded
    with factor 1.0 (matching the old per-memory ``update_decay``).
    """
    if not memory_ids:
        return {}

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                UPDATE memories
                SET recall_count = recall_count + 1,
                    recalled_at = NOW(),
                    decay_factor = round(
                        exp(-EXTRACT(EPOCH FROM (NOW() - COALESCE(recalled_at, NOW())))
                            / 3600.0
                            / (1.0 + (recall_count + 1) * 2.0)
                        )::numeric, 4
                    )::float8
                WHERE id = ANY(:ids)
                RETURNING id, decay_factor
                """
            ),
            {"ids": memory_ids},
        )
        await session.commit()
        return {str(row[0]): float(row[1]) for row in result.fetchall()}


async def search_memories(
    query_vector: list[float],
    top_k: int = 20,
    threshold: float = 0.0,
) -> list[dict]:
    """Vector search against memories table, weighted by live decay.

    The decay factor is computed live from ``recalled_at`` and
    ``recall_count`` at query time — not read from the stored snapshot.
    Ebbinghaus decay is a continuous curve in *time*; ranking by a stale
    stored factor would freeze a memory's rank at its last recall (a memory
    recalled a month ago and one recalled yesterday would both sort by their
    old snapshot values).  The stored ``decay_factor`` column remains —
    written by :func:`update_decay_batch` — for display / backwards
    compatibility, but ranking and the returned factor use the live value.

    Two-stage recall keeps the pgvector HNSW index (``hnsw (embedding
    vector_cosine_ops)``) in play: the index only serves scans that sort
    directly by ``embedding <=> :vec``, so stage 1 fetches a generous
    candidate window in index order (``candidate_n = max(top_k * 8, 50)``)
    and stage 2 re-ranks those candidates in Python by
    ``similarity * decay_factor``.  The rank order matches the old single
    SQL expression without a full-table scan + in-memory sort.

    The live factor uses the same post-recall strength convention as
    :func:`update_decay_batch` — ``1 + (recall_count + 1) * 2`` — so the
    factor computed right after this search agrees with the snapshot the
    next UPDATE writes.

    Clock assumption: the application clock is assumed to be consistent with
    the database clock (``recalled_at`` is written by the DB's ``NOW()``).
    As a defensive measure, :func:`compute_decay_factor` clamps elapsed time
    to zero when it would come out negative (app clock lagging the DB), so a
    skewed wall clock can never push a factor above 1.0 (full retention) or
    overflow ``math.exp``.
    """
    candidate_n = max(top_k * 8, 50)

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT id, source_type, summary, entities, relations,
                       recall_count, meta, created_at, recalled_at,
                       1 - (embedding <=> :vec ::vector) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL
                  AND deleted_at IS NULL
                  AND 1 - (embedding <=> :vec ::vector) > :threshold
                ORDER BY embedding <=> :vec ::vector
                LIMIT :candidate_n
                """
            ),
            {
                "vec": str(query_vector),
                "threshold": threshold,
                "candidate_n": candidate_n,
            },
        )
        rows = [dict(r._mapping) for r in result]

    now = datetime.now(timezone.utc)
    for r in rows:
        recalled_at = r.pop("recalled_at")
        # Post-recall convention: the count is about to be bumped by this
        # search's downstream update_decay_batch, so add 1 before mirroring.
        r["decay_factor"] = compute_decay_factor(
            recalled_at, r["recall_count"] + 1, now=now
        )
        r["weighted_score"] = r.pop("similarity") * r["decay_factor"]

    rows.sort(key=lambda r: r["weighted_score"], reverse=True)
    return rows[:top_k]
