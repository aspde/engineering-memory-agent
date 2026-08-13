"""CLI entry: seed the eval corpus into the DB.

The eval corpus lives in ``seed_memories.jsonl``. This script writes it into
the chunks table (so ``retrieval.retrieve`` can be evaluated) and, optionally,
into the memories table (so ``retrieval.query_memories`` can be evaluated).

Examples:
    # Dry-run: print what would be written, write nothing
    python -m tests.eval.seed --dry-run

    # Seed chunks table only (default; no LLM cost)
    python -m tests.eval.seed

    # Seed chunks + memories tables (memories path needs LLM for embedding)
    python -m tests.eval.seed --memories

    # Re-seed from scratch: clear seed-tagged rows first
    python -m tests.eval.seed --clear --memories
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence

from tests.eval.dataset import SeedMemory, load_seed_memories

SEED_DOCUMENT_ID = "ema-eval-seed"
SEED_SOURCE_TYPE = "eval_seed"


async def _clear_chunks() -> int:
    """Delete chunks whose document_id matches the seed tag. Returns count."""
    from sqlalchemy import text

    from backend.db import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text("DELETE FROM chunks WHERE document_id = :did RETURNING id"),
            {"did": SEED_DOCUMENT_ID},
        )
        rows = result.fetchall()
        await session.commit()
        return len(rows)


async def _clear_memories() -> int:
    """Delete memories whose source_type matches the seed tag. Returns count."""
    from sqlalchemy import text

    from backend.db import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                "DELETE FROM memories WHERE source_type = :st RETURNING id"
            ),
            {"st": SEED_SOURCE_TYPE},
        )
        rows = result.fetchall()
        await session.commit()
        return len(rows)


async def _seed_chunks(seeds: Sequence[SeedMemory]) -> int:
    """Write each seed memory's content as one chunk. Returns count written."""
    from backend.service.retrieval import write_chunks

    total = 0
    # One document_id for all seeds — keeps the seed corpus identifiable and
    # makes --clear a single DELETE. chunk_index differentiates rows.
    contents = [s.content for s in seeds]
    total = await write_chunks(
        document_id=SEED_DOCUMENT_ID,
        chunks=contents,
        meta={"source": "eval_seed", "categories": [s.category for s in seeds]},
    )
    return total


async def _seed_memories(seeds: Sequence[SeedMemory]) -> int:
    """Write each seed as a structured memory (summary embed). Returns count.

    Bypasses ``extract_memory`` (no LLM call) — entities/relations are taken
    verbatim from the seed file. The embedding is generated for ``summary``.
    """
    import json

    from sqlalchemy import text

    from backend.db import get_session_factory
    from backend.service.embedding_service import get_embedding_provider

    provider = get_embedding_provider()
    summaries = [s.summary for s in seeds]
    vectors = await provider.embed(summaries)

    factory = get_session_factory()
    count = 0
    async with factory() as session:
        for s, vec in zip(seeds, vectors):
            await session.execute(
                text(
                    """\
                    INSERT INTO memories
                        (source_type, summary, entities, relations,
                         embedding, meta)
                    VALUES
                        (:st, :summary, :entities ::jsonb, :relations ::jsonb,
                         :vec ::vector, :meta ::jsonb)
                    """
                ),
                {
                    "st": SEED_SOURCE_TYPE,
                    "summary": s.summary,
                    "entities": json.dumps(s.entities, ensure_ascii=False),
                    "relations": json.dumps(s.relations, ensure_ascii=False),
                    "vec": str(vec),
                    "meta": json.dumps(
                        {"source": "eval_seed", "category": s.category, "seed_id": s.id},
                        ensure_ascii=False,
                    ),
                },
            )
            count += 1
        await session.commit()
    return count


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.seed",
        description="Seed the EMA eval corpus into the chunks/memories tables.",
    )
    p.add_argument(
        "--memories",
        action="store_true",
        help="Also seed the memories table (needs embedding provider). "
        "Default: only seed chunks.",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Delete previously-seeded rows before writing.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written; do not touch the DB.",
    )
    return p


async def _run(args: argparse.Namespace) -> int:
    seeds = load_seed_memories()
    print(f"Loaded {len(seeds)} seed memories", file=sys.stderr)

    if args.dry_run:
        for s in seeds:
            print(f"  [{s.category}] {s.id}: {s.summary[:80]}...")
        return 0

    if args.clear:
        n_chunks = await _clear_chunks()
        print(f"✓ Cleared {n_chunks} seed chunks", file=sys.stderr)
        if args.memories:
            n_mem = await _clear_memories()
            print(f"✓ Cleared {n_mem} seed memories", file=sys.stderr)

    n_chunks = await _seed_chunks(seeds)
    print(f"✓ Wrote {n_chunks} chunks (document_id={SEED_DOCUMENT_ID})", file=sys.stderr)

    if args.memories:
        n_mem = await _seed_memories(seeds)
        print(f"✓ Wrote {n_mem} memories (source_type={SEED_SOURCE_TYPE})", file=sys.stderr)

    return 0


def main() -> None:
    args = _build_parser().parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
