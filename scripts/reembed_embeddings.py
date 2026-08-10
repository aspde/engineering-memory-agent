"""Re-embed rows whose ``embedding`` column is NULL (after a dimension migration).

When the configured embedding model changes dimension, ``init_db`` empties the
pgvector columns (vectors of the old size can't fit the resized column) and
resizes them.  This script re-embeds the affected rows from their stored text
so retrieval works again:

    python -m scripts.reembed_embeddings                 # chunks + memories
    python -m scripts.reembed_embeddings --table chunks  # one table only
    python -m scripts.reembed_embeddings --dry-run       # report counts without writing

Text columns: ``chunks.content``, ``memories.summary``.  Only rows with a NULL
embedding are touched, so the script is idempotent and safe to re-run.  Rows
are processed in bounded batches so a large table never materialises all its
texts in memory at once.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.embedding_service import get_embedding_provider

# table → (text column whose embedding is derived from it, human label)
_TABLES: dict[str, tuple[str, str]] = {
    "chunks": ("content", "document chunks"),
    "memories": ("summary", "memories"),
}

_BATCH_SIZE = 500


async def _reembed_table(table: str, text_col: str) -> int:
    """Re-embed every row of *table* whose embedding is NULL, in batches."""
    provider = get_embedding_provider()
    factory = get_session_factory()
    total = 0

    while True:
        async with factory() as session:
            result = await session.execute(
                text(
                    f"SELECT id, {text_col} FROM {table} "
                    "WHERE embedding IS NULL ORDER BY id LIMIT :batch"
                ),
                {"batch": _BATCH_SIZE},
            )
            rows = [dict(r._mapping) for r in result]

        if not rows:
            break

        total += len(rows)
        vectors = await provider.embed([r[text_col] for r in rows])

        params = [
            {"id": r["id"], "vec": str(vec)}
            for r, vec in zip(rows, vectors)
        ]
        async with factory() as session:
            # Single executemany — one round trip instead of N sequential UPDATEs.
            await session.execute(
                text(f"UPDATE {table} SET embedding = :vec ::vector WHERE id = :id"),
                params,
            )
            await session.commit()

        print(f"  {table}: re-embedded {total} rows so far", file=sys.stderr)

    return total


async def backfill(tables: list[str], dry_run: bool = False) -> dict[str, int]:
    """Re-embed (or count, in dry-run) NULL-embedding rows for each *table*."""
    factory = get_session_factory()
    counts: dict[str, int] = {}

    for table in tables:
        text_col, label = _TABLES[table]
        if dry_run:
            # COUNT only — no model load, no writes.
            async with factory() as session:
                result = await session.execute(
                    text(f"SELECT COUNT(*) FROM {table} WHERE embedding IS NULL")
                )
                counts[table] = int(result.scalar() or 0)
            print(f"{label} ({table}): {counts[table]} rows need re-embedding")
            continue

        counts[table] = await _reembed_table(table, text_col)
        print(f"✓ Re-embedded {counts[table]} {label} rows.", file=sys.stderr)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.reembed_embeddings",
        description="Re-embed rows whose embedding column is NULL (after a dimension migration).",
    )
    parser.add_argument(
        "--table",
        choices=sorted(_TABLES),
        action="append",
        default=[],
        help="Table to re-embed (repeatable; default: all of chunks, memories).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many rows need re-embedding without writing.",
    )
    args = parser.parse_args()

    tables = args.table or sorted(_TABLES)
    asyncio.run(backfill(tables, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
