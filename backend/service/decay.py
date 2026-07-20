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

from sqlalchemy import text

from backend.db import get_session_factory

logger = logging.getLogger(__name__)


def compute_decay_factor(
    recalled_at: datetime | None,
    recall_count: int,
) -> float:
    """Compute the current decay factor for a memory.

    A value of 1.0 means full retention; 0.0 means fully forgotten.
    """
    if recalled_at is None:
        return 1.0  # Just inserted, no time has passed yet

    now = datetime.now(timezone.utc)
    hours_elapsed = (now - recalled_at).total_seconds() / 3600.0
    strength = 1.0 + recall_count * 2.0
    decay = math.exp(-hours_elapsed / strength)
    return round(decay, 4)


async def update_decay(memory_id: str) -> float:
    """Bump recall_count, update recalled_at and decay_factor for a memory.

    Returns the new decay_factor.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        # Fetch current state
        result = await session.execute(
            text("SELECT recalled_at, recall_count FROM memories WHERE id = :id"),
            {"id": memory_id},
        )
        row = result.fetchone()
        if row is None:
            logger.warning("Memory %s not found for decay update", memory_id)
            return 1.0

        recalled_at = row.recalled_at
        recall_count = row.recall_count + 1

        new_decay = compute_decay_factor(recalled_at, recall_count)

        await session.execute(
            text(
                """\
                UPDATE memories
                SET recall_count = :count,
                    recalled_at = NOW(),
                    decay_factor = :decay
                WHERE id = :id
                """
            ),
            {"count": recall_count, "decay": new_decay, "id": memory_id},
        )
        await session.commit()

    logger.debug("Updated decay for %s: factor=%.4f, count=%d", memory_id, new_decay, recall_count)
    return new_decay


async def search_memories(
    query_vector: list[float],
    top_k: int = 20,
    threshold: float = 0.0,
) -> list[dict]:
    """Vector search against memories table, weighted by decay_factor.

    Cosine similarity is multiplied by decay_factor so that
    frequently-recalled, recently-recalled memories rank higher.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT id, source_type, summary, entities, relations,
                       decay_factor, recall_count, meta, created_at,
                       (1 - (embedding <=> :vec ::vector)) * decay_factor AS weighted_score
                FROM memories
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> :vec ::vector) > :threshold
                ORDER BY (1 - (embedding <=> :vec ::vector)) * decay_factor DESC
                LIMIT :limit
                """
            ),
            {"vec": str(query_vector), "threshold": threshold, "limit": top_k},
        )
        return [dict(r._mapping) for r in result]
