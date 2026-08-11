"""Ebbinghaus forgetting curve — decay weighting for memory retrieval.

On each recall, the decay factor is updated based on time elapsed since the
last recall.  More frequent recalls slow the decay; long gaps accelerate it.

Formula (simplified Ebbinghaus):
    R = max(e^(-t / S), FLOOR)
where:
    t = hours since the time base
        = now() - recalled_at if the memory was ever recalled
          else now() - created_at (never-recalled memories decay by age,
          so stale-but-unused entries sink instead of keeping 1.0 forever)
    S = relative strength = 1 + (recall_count + 1) * 12
        The 12x multiplier (was 2x) was tuned against the decay A/B
        (``tests/eval/decay_ab.py``): the old curve's half-life ≈ 0.69·S
        hours sank every cold memory to factor ≈ 0 (recall@5 fell 0.900 →
        0.367 on the synthetic aging profile), burying relevant old memories
        wholesale.  12x slows the curve so a cold memory decays over ~9+ hours
        instead of ~2, and FLOOR (0.10) keeps a fully-decayed memory
        retrievable when its similarity is high — the curve still biases
        ranking toward fresh+well-recalled memories without making
        old-but-relevant ones unrecoverable.  A/B after tuning: recall@5
        0.367 → 0.667, MRR 0.367 → 0.622 (S=8+floor 0.05 gave 0.633/0.559;
        the further step to 12/0.10 bought MRR at little recall cost).
    recall_count = the stored count BEFORE this recall; the ``+ 1`` yields
                   the count AFTER the recall — all three call sites
                   (compute_decay_factor / update_decay_batch /
                   search_memories) share this post-recall convention.

FLOOR is duplicated in the ``update_decay_batch`` SQL (GREATEST … 0.10) —
keep the two in sync.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory

# Retention floor — a fully-decayed memory keeps at least this much weight
# so a high-similarity match can still surface it.  Mirrored in the
# update_decay_batch SQL (GREATEST … 0.05); keep the two in sync.
_DECAY_FLOOR = 0.10

# Strength multiplier for the Ebbinghaus curve — see the module docstring for
# the A/B tuning rationale.
_STRENGTH_MULTIPLIER = 12.0

logger = logging.getLogger(__name__)


def compute_decay_factor(
    recalled_at: datetime | None,
    recall_count: int,
    *,
    now: datetime | None = None,
    created_at: datetime | None = None,
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

    Time base: ``recalled_at`` when the memory was ever recalled, else
    ``created_at``.  Falling back to ``created_at`` is deliberate — a memory
    written long ago that was never retrieved must decay like any other idle
    memory (otherwise "never recalled" would mean "never forgets", and stale
    entries would keep full rank forever).  When both are ``None`` (brand-new
    row with no time info in this call), there is no elapsed time and the
    factor is 1.0.

    Clock assumption: the application clock is assumed to be consistent with
    the database clock.  As a defensive measure, elapsed time is clamped to
    zero when it would come out negative (app clock lagging the DB), so a
    skewed wall clock can never push the factor above 1.0 or overflow
    ``math.exp``.  The decay formula itself is unchanged.
    """
    base = recalled_at if recalled_at is not None else created_at
    if base is None:
        return 1.0  # No time reference — nothing to decay against yet

    if now is None:
        now = datetime.now(timezone.utc)
    hours_elapsed = (now - base).total_seconds() / 3600.0
    # Defensive: the application clock is assumed to be consistent with the
    # database clock (see search_memories).  If the app clock lags the DB's,
    # a just-written ``recalled_at`` looks like the future and elapsed goes
    # negative — the factor would exceed 1.0 (inflating rank) and could even
    # overflow math.exp.  Clamp to zero so a future-dated recall reads as
    # "just now" (factor 1.0), never above full retention.  This clamps the
    # elapsed *input* only; the decay formula itself is unchanged.
    hours_elapsed = max(0.0, hours_elapsed)
    strength = 1.0 + recall_count * _STRENGTH_MULTIPLIER
    decay = math.exp(-hours_elapsed / strength)
    decay = max(decay, _DECAY_FLOOR)
    return round(decay, 4)


async def update_decay_batch(memory_ids: list[Any]) -> dict[str, float]:
    """Atomically bump recall stats for many memories, in one round-trip.

    Replaces the old per-memory ``update_decay`` loop: N independent
    sessions/commits on every memory search (the N+1 write), and a non-atomic
    read-modify-write that lost increments under concurrent recalls.  The
    whole update is one ``UPDATE ... RETURNING`` where the new factor is
    computed in SQL from the stored ``recalled_at`` — same formula as
    :func:`compute_decay_factor` (strength = 1 + (recall_count + 1) *
    _STRENGTH_MULTIPLIER, floored at _DECAY_FLOOR, rounded to 4 dp).  The
    multiplier and floor are passed as bound parameters so the SQL can never
    drift from the Python constants.  ``recall_count`` on the RHS is the
    *stored* pre-recall value, so ``+ 1`` yields the post-recall count that
    the statement is writing back — the three call sites share this
    post-recall convention.  ``NOW()`` is the transaction timestamp, constant
    within the statement, so the factor is consistent across all updated rows.

    Returns ``{str(memory_id): new_decay_factor}``.  Ids whose row is missing
    are absent from the result — callers fall back to the factor they already
    hold.  A never-recalled row (``recalled_at IS NULL``) is still counted:
    ``COALESCE`` falls back to ``created_at`` as the time base, so a memory
    written long ago that was never searched keeps its age-based decay on
    first recall (matching the live factor ``search_memories`` computes for
    it) rather than being "reset to fresh" at 1.0.
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
                        GREATEST(
                            exp(-EXTRACT(EPOCH FROM (NOW() - COALESCE(recalled_at, created_at)))
                                / 3600.0
                                / (1.0 + (recall_count + 1) * :strength_multiplier)
                            ),
                            :floor
                        )::numeric, 4
                    )::float8
                WHERE id = ANY(:ids)
                RETURNING id, decay_factor
                """
            ),
            {
                "ids": memory_ids,
                "strength_multiplier": _STRENGTH_MULTIPLIER,
                "floor": _DECAY_FLOOR,
            },
        )
        await session.commit()
        return {str(row[0]): float(row[1]) for row in result.fetchall()}


async def search_memories(
    query_vector: list[float],
    top_k: int = 20,
    threshold: float = 0.0,
    use_decay: bool = True,
) -> list[dict]:
    """Vector search against memories table, weighted by live decay.

    The decay factor is computed live from ``recalled_at`` and
    ``recall_count`` at query time — not read from the stored snapshot.
    Pass ``use_decay=False`` to rank by raw similarity instead — the decay
    A/B measures whether Ebbinghaus weighting changes retrieval at all
    (``tests/eval/decay_ab.py``).
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
    :func:`update_decay_batch` — ``1 + (recall_count + 1) *
    _STRENGTH_MULTIPLIER``, floored at ``_DECAY_FLOOR`` — so the factor
    computed right after this search agrees with the snapshot the next
    UPDATE writes.

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
    if use_decay:
        for r in rows:
            recalled_at = r.pop("recalled_at")
            created_at = r.pop("created_at")
            # Post-recall convention: the count is about to be bumped by this
            # search's downstream update_decay_batch, so add 1 before mirroring.
            # Time base falls back to created_at for never-recalled memories, so
            # an old-but-never-searched memory decays instead of keeping 1.0.
            r["decay_factor"] = compute_decay_factor(
                recalled_at, r["recall_count"] + 1, now=now, created_at=created_at
            )
            r["weighted_score"] = r.pop("similarity") * r["decay_factor"]
        rows.sort(key=lambda r: r["weighted_score"], reverse=True)
    else:
        # Raw-similarity ranking: no decay computation, no decay_factor key.
        # The A/B needs this as the "no decay" control — otherwise a memory
        # that happens to be old or cold would rank below an unrelated fresh
        # one, and the comparison couldn't isolate the decay effect.
        rows.sort(key=lambda r: r["similarity"], reverse=True)
    return rows[:top_k]
