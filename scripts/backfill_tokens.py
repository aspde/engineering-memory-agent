"""Backfill the ``tokens`` column for pre-existing chunks.

After adding the ``tokens TEXT[]`` column + GIN index (see
``backend/db/schema.py``), rows written before the migration have empty
tokens.  This script re-segments each chunk's content with jieba and
populates the column so ``sparse_search`` can use the GIN index.

Usage:
    python -m scripts.backfill_tokens            # backfill all empty-token rows
    python -m scripts.backfill_tokens --dry-run  # preview without writing
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.retrieval import _tokenize


async def backfill(dry_run: bool = False) -> int:
    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text(
                "SELECT id, content FROM chunks "
                "WHERE tokens IS NULL OR cardinality(tokens) = 0"
            )
        )
        rows = [dict(r._mapping) for r in result]

        if not rows:
            print("No chunks need backfilling (all have tokens).")
            return 0

        print(f"Found {len(rows)} chunks to backfill.", file=sys.stderr)

        if dry_run:
            for r in rows[:5]:
                tokens = _tokenize(r["content"])
                print(f"  {r['id']}: {len(tokens)} tokens")
            if len(rows) > 5:
                print(f"  ... and {len(rows) - 5} more")
            return 0

        params = [
            {"tokens": list(_tokenize(r["content"])), "id": r["id"]}
            for r in rows
        ]
        # Single executemany — one round trip instead of N sequential UPDATEs.
        await session.execute(
            text("UPDATE chunks SET tokens = :tokens ::text[] WHERE id = :id"),
            params,
        )
        await session.commit()
        print(f"✓ Backfilled {len(rows)} chunks with jieba tokens.", file=sys.stderr)
        return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.backfill_tokens",
        description="Backfill the tokens column for pre-existing chunks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing.",
    )
    args = parser.parse_args()
    asyncio.run(backfill(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
