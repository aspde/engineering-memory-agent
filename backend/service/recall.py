"""Recall tracking — how often and when each memory has been retrieved.

Ranking is pure similarity (see ``retrieval.search_memories``); this module
only records that a search surfaced a memory, so ``recall_count`` /
``recalled_at`` are metadata, not a ranking signal.

Why is there no Ebbinghaus decay weighting any more?  The decay A/B
(``tests/eval/reports/decay_ab_report.md``) measured recall@5 0.667 with
decay weighting vs 0.900 without it, on a *synthetic* aging profile — and
the profile's assumption that "recently-recalled = relevant" had no
real-corpus support.  Archival decisions are a human/LLM judgement over the
raw access history (surfaced to the patrol "stale memories" prompt via
``search_memories_tool``), not a machine-computed decay factor.

The ``decay_factor`` column was dropped in migration 0002 — nothing wrote
or read it since the decay weighting was removed.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import text

from backend.db import get_session_factory

logger = logging.getLogger(__name__)


async def record_recalls(memory_ids: list[UUID]) -> None:
    """Atomically bump ``recall_count`` / ``recalled_at`` for many memories.

    One ``UPDATE ... RETURNING``-free round-trip instead of N sequential
    commits (no N+1 writes), and the increment is server-side so concurrent
    recalls never lose counts.  A never-recalled row gets its first
    ``recalled_at`` here.  An empty list is a no-op — callers that survived
    an empty ranking still pass through cleanly.
    """
    if not memory_ids:
        return

    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                """\
                UPDATE memories
                SET recall_count = recall_count + 1,
                    recalled_at = NOW()
                WHERE id = ANY(:ids)
                """
            ),
            {"ids": memory_ids},
        )
        await session.commit()
