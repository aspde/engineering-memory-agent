"""CLI entry: seed the e2e eval corpus into the DB.

The e2e suite (``tests.eval.run_llm_eval --suite e2e``) drives real
retrieval, so each item's ``source_content`` must be searchable.  This
module writes those contents into the memories table (for memory-mode items,
via the ``query_memories`` path) and the chunks table (for chunk-mode items,
via the ``retrieve_hybrid`` path), tagged so the e2e corpus is identifiable
and clearable without touching the retrieval-eval seed corpus
(``tests.eval.seed``) or production memories.

Memory-mode rows are inserted directly — ``summary = source_content`` with a
fresh embedding — bypassing ``write_memory``'s LLM extraction so seeding is
fast and deterministic.  Chunk-mode rows reuse ``write_chunks``.  Both paths
are idempotent (content-hash / ON CONFLICT), so re-seeding is a no-op.

Examples:
    # Dry-run: print what would be seeded, write nothing
    python -m tests.eval.e2e_seed --dry-run

    # Seed the e2e corpus (memories + chunks)
    python -m tests.eval.e2e_seed

    # Re-seed from scratch: clear previously-seeded e2e rows first
    python -m tests.eval.e2e_seed --clear
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence

from tests.eval.llm_ground_truth import E2EItem, load_e2e_items

SEED_SOURCE_TYPE = "eval_e2e"
SEED_DOCUMENT_ID = "ema-e2e-seed"


def _content_hash(text: str) -> str:
    """SHA-256 of raw content — the exact-duplicate idempotency key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def clear_e2e_corpus() -> int:
    """Delete previously-seeded e2e rows. Returns total rows removed."""
    from sqlalchemy import text

    from backend.db import get_session_factory

    factory = get_session_factory()
    total = 0
    async with factory() as session:
        mem = await session.execute(
            text("DELETE FROM memories WHERE source_type = :st RETURNING id"),
            {"st": SEED_SOURCE_TYPE},
        )
        total += len(mem.fetchall())
        chunks = await session.execute(
            text("DELETE FROM chunks WHERE document_id = :did RETURNING id"),
            {"did": SEED_DOCUMENT_ID},
        )
        total += len(chunks.fetchall())
        await session.commit()
    return total


async def _seed_memory_item(item: E2EItem) -> int:
    """Insert one memory-mode item as a searchable memory row.

    ``summary = source_content`` (the e2e suite needs the facts retrievable
    via the memory path, which searches summaries).  Idempotent on the
    content-hash unique index; a re-seed of an unchanged item is a no-op.
    """
    from sqlalchemy import text

    from backend.db import get_session_factory
    from backend.service.embedding_service import get_embedding_provider

    provider = get_embedding_provider()
    vectors = await provider.embed([item.source_content])
    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            text(
                """\
                INSERT INTO memories
                    (source_type, summary, entities, relations, embedding, meta, content_hash)
                VALUES
                    (:st, :summary, '[]' ::jsonb, '[]' ::jsonb, :vec ::vector,
                     :meta ::jsonb, :hash)
                ON CONFLICT (content_hash) WHERE deleted_at IS NULL DO NOTHING
                """
            ),
            {
                "st": SEED_SOURCE_TYPE,
                "summary": item.source_content,
                "vec": str(vectors[0]),
                "meta": json.dumps(
                    {"source": "eval_e2e", "item_id": item.id}, ensure_ascii=False
                ),
                "hash": _content_hash(item.source_content),
            },
        )
        await session.commit()
    return 1


async def _seed_chunk_item(item: E2EItem) -> int:
    """Insert one chunk-mode item as a searchable chunk."""
    from backend.service.retrieval import write_chunks

    return await write_chunks(
        document_id=SEED_DOCUMENT_ID,
        chunks=[item.source_content],
        meta={"source": "eval_e2e", "item_id": item.id},
    )


async def seed_e2e_corpus(items: Sequence[E2EItem] | None = None) -> tuple[int, int]:
    """Seed every e2e item's source_content. Returns (memories, chunks) written.

    Idempotent: unchanged items are skipped (content-hash), so re-running is
    a no-op and ``--clear`` is only needed to force a fresh corpus.
    """
    items = list(items) if items is not None else load_e2e_items()
    n_mem = 0
    n_chunks = 0
    for it in items:
        if it.retrieval_mode == "memory":
            n_mem += await _seed_memory_item(it)
        else:
            n_chunks += await _seed_chunk_item(it)
    return n_mem, n_chunks


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.e2e_seed",
        description="Seed the EMA e2e eval corpus into the memories/chunks tables.",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Delete previously-seeded e2e rows before writing.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be seeded; do not touch the DB.",
    )
    return p


async def _run(args: argparse.Namespace) -> int:
    items = load_e2e_items()
    mem_items = [it for it in items if it.retrieval_mode == "memory"]
    chunk_items = [it for it in items if it.retrieval_mode == "chunk"]
    print(
        f"Loaded {len(items)} e2e items "
        f"({len(mem_items)} memory, {len(chunk_items)} chunk)",
        file=sys.stderr,
    )

    if args.dry_run:
        for it in items:
            print(f"  [{it.retrieval_mode}] {it.id}: {it.source_content[:80]}...")
        return 0

    # Ensure the schema exists before touching the tables: a fresh postgres
    # (e.g. the CI e2e-eval service container) has no memories/chunks tables
    # yet, and every INSERT/DELETE below would fail with "relation does not
    # exist".  init_db is CREATE TABLE IF NOT EXISTS, so it's a cheap no-op on
    # an already-initialized database.  Skipped in --dry-run (no DB access).
    from backend.db.schema import init_db

    await init_db()

    if args.clear:
        cleared = await clear_e2e_corpus()
        print(f"✓ Cleared {cleared} previously-seeded e2e row(s)", file=sys.stderr)

    n_mem, n_chunks = await seed_e2e_corpus(items)
    print(
        f"✓ Seeded {n_mem} memory row(s), {n_chunks} chunk row(s) "
        f"(tag={SEED_SOURCE_TYPE})",
        file=sys.stderr,
    )
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
