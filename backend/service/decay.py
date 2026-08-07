"""Ebbinghaus forgetting curve — decay weighting for memory retrieval.

On each recall, the decay factor is updated based on time elapsed since the
last recall.  More frequent recalls slow the decay; long gaps accelerate it.

Formula (simplified Ebbinghaus):
    R = e^(-t / S)
where:
    t = hours since last recall
    S = relative strength = 1 + recall_count * 2
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
    and :func:`search_memories` — keep the three in sync.  ``now`` is an
    optional reference instant (default: the current wall clock); tests pass
    the database's ``NOW()`` so the mirror is compared against the SQL
    formula at the *same* instant instead of a skewed local clock.
    """
    if recalled_at is None:
        return 1.0  # Just inserted, no time has passed yet

    if now is None:
        now = datetime.now(timezone.utc)
    hours_elapsed = (now - recalled_at).total_seconds() / 3600.0
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
    :func:`compute_decay_factor` (strength = 1 + recall_count*2, rounded to
    4 dp).  ``NOW()`` is the transaction timestamp, constant within the
    statement, so the factor is consistent across all updated rows.

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

    The decay factor is computed in SQL at query time from ``recalled_at``
    and ``recall_count`` — not read from the stored snapshot.  Ebbinghaus
    decay is a continuous curve in *time*; ranking by a stale stored factor
    would freeze a memory's rank at its last recall (a memory recalled a
    month ago and one recalled yesterday would both sort by their old
    snapshot values).  The stored ``decay_factor`` column remains — written
    by :func:`update_decay_batch` — for display / backwards compatibility,
    but ranking and the returned factor use the live value.  Cosine
    similarity is multiplied by the live decay so frequently- and
    recently-recalled memories rank higher.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                WITH live AS (
                    SELECT id, source_type, summary, entities, relations,
                           recall_count, meta, created_at,
                           (1 - (embedding <=> :vec ::vector)) AS similarity,
                           CASE WHEN recalled_at IS NULL THEN 1.0::float8
                                ELSE exp(-(EXTRACT(EPOCH FROM (now() - recalled_at)) / 3600.0)
                                         / (1.0 + recall_count * 2.0))::float8
                           END AS decay_factor
                    FROM memories
                    WHERE embedding IS NOT NULL
                      AND deleted_at IS NULL
                      AND 1 - (embedding <=> :vec ::vector) > :threshold
                )
                SELECT id, source_type, summary, entities, relations,
                       recall_count, meta, created_at, decay_factor,
                       similarity * decay_factor AS weighted_score
                FROM live
                ORDER BY similarity * decay_factor DESC
                LIMIT :limit
                """
            ),
            {"vec": str(query_vector), "threshold": threshold, "limit": top_k},
        )
        return [dict(r._mapping) for r in result]
