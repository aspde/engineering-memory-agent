"""Check for entity-normalization gaps — memories that extracted entities
but have no ``memory_entities`` link.

Entity normalisation (``backend.service.entity.normalize_entities``) runs
fire-and-forget after a memory write (``backend.service.memory``), so a
failure — LLM/embedding timeout, connection-pool exhaustion — is logged and
skipped silently: the memory is stored but its entities stay unlinked, and
the content-hash idempotency gate means the same source is never rewritten
to retry.  This script surfaces that gap with a read-only query:

    python -m scripts.check_normalization_gaps

Output: a one-line health summary plus every unlinked memory (id, source,
created_at, extracted entity names, summary).  Exits 0 when no gaps exist,
1 when gaps are found — usable as a CI guard or the trigger to run
``normalize_all_existing()`` (via a REPL) to backfill the links.

Nothing is written; safe to run at any time.
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from backend.db import get_session_factory

# A live memory "has entities to normalise" when its JSONB list is non-empty;
# an empty/absent list needs no link, so it is not a gap.
_GAP_WHERE = """
    m.deleted_at IS NULL
    AND jsonb_typeof(m.entities) = 'array'
    AND jsonb_array_length(m.entities) > 0
    AND NOT EXISTS (
        SELECT 1 FROM memory_entities me WHERE me.memory_id = m.id
    )
"""


async def check() -> int:
    """Report normalization gaps; return how many unlinked memories exist."""
    factory = get_session_factory()
    async with factory() as session:
        # ── Health summary (same shape as the hand-run diagnostic) ──
        r = await session.execute(
            text(
                """\
                SELECT
                  COUNT(*) FILTER (WHERE jsonb_typeof(entities)='array'
                                   AND jsonb_array_length(entities)>0) AS with_entities_jsonb,
                  COUNT(*) FILTER (WHERE jsonb_typeof(entities)='array'
                                   AND jsonb_array_length(entities)>0
                                   AND NOT EXISTS (
                                       SELECT 1 FROM memory_entities me
                                       WHERE me.memory_id = memories.id)) AS unlinked,
                  (SELECT COUNT(*) FROM memory_entities) AS links,
                  (SELECT COUNT(*) FROM entities) AS entities
                FROM memories WHERE deleted_at IS NULL
                """
            )
        )
        with_entities, unlinked, links, entities = r.one()
        print(
            f"Health: {with_entities} live memories with entities, "
            f"{unlinked} unlinked ({links} links, {entities} entities)"
        )

        if unlinked == 0:
            print("No entity-normalization gaps.")
            return 0

        # ── Details for the ones that need attention ──
        print(f"\n{unlinked} memory(ies) extracted entities but have no link:")
        rows = await session.execute(
            text(
                """\
                SELECT m.id, m.source_type, m.summary, m.created_at,
                       (SELECT string_agg(e->>'name', ', ')
                        FROM jsonb_array_elements(m.entities) e) AS entity_names
                FROM memories m
                WHERE """ + _GAP_WHERE + """
                ORDER BY m.created_at DESC
                """
            )
        )
        for row in rows:
            summary = (row.summary or "").strip().replace("\n", " ")
            if len(summary) > 100:
                summary = summary[:100] + "…"
            created = row.created_at.isoformat() if row.created_at else "?"
            print(f"  - {str(row.id)[:8]} [{row.source_type}] {created}")
            print(f"      entities: {row.entity_names or ''}")
            print(f"      summary : {summary}")
        return unlinked


def main() -> None:
    try:
        gaps = asyncio.run(check())
    except Exception as exc:
        print(f"check_normalization_gaps failed: {exc}", file=sys.stderr)
        sys.exit(2)
    # Exit 1 when gaps exist — a CI guard / backfill trigger.
    sys.exit(1 if gaps else 0)


if __name__ == "__main__":
    main()
