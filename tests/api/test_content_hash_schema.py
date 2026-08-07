"""Schema-level tests for the content-hash idempotency invariants.

These run against the real database (init_db), so they verify the partial
unique index semantics that the mock-based unit tests cannot: a soft-deleted
memory must not block hash reuse, and two live memories must never share a
hash.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from backend.db import get_session_factory
from backend.db.schema import init_db


@pytest.fixture(autouse=True)
async def _ensure_tables() -> None:
    """Create tables (including the content_hash indexes) before tests."""
    await init_db()


@pytest.fixture(autouse=True)
async def _clean_memories() -> None:
    """Isolate each test from prior runs (the test DB persists between runs)."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(text("DELETE FROM memories"))
        await session.commit()
    yield


async def _insert_memory(content_hash: str, deleted: bool = False) -> str:
    """Insert a memories row with the given content_hash; return its id."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                INSERT INTO memories (source_type, summary, entities, relations, content_hash, meta)
                VALUES ('test', 's', '[]', '[]', :hash, '{}')
                RETURNING id
                """
            ),
            {"hash": content_hash},
        )
        mid = result.fetchone()[0]
        if deleted:
            await session.execute(
                text("UPDATE memories SET deleted_at = now() WHERE id = :id"),
                {"id": mid},
            )
        await session.commit()
        return str(mid)


class TestMemoriesPartialUniqueIndex:
    @pytest.mark.asyncio
    async def test_soft_deleted_row_does_not_block_hash_reuse(self) -> None:
        """A tombstone keeps its content_hash but must not block a live row.

        Regression for the IntegrityError path: X holds H, gets soft-deleted,
        content C re-ingested → the idempotency gate (deleted_at IS NULL)
        misses X → merges into a live row → UPDATE sets content_hash=H.  The
        partial index (WHERE deleted_at IS NULL) lets that UPDATE succeed.
        """
        await _insert_memory("reused-hash", deleted=True)
        live_id = await _insert_memory("reused-hash")
        assert live_id

    @pytest.mark.asyncio
    async def test_two_live_rows_cannot_share_a_hash(self) -> None:
        """The unique constraint still holds among live rows."""
        await _insert_memory("live-hash")
        with pytest.raises(IntegrityError):
            await _insert_memory("live-hash")
