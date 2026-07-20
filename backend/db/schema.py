"""Database schema — tables created via raw SQL to keep things simple.

Tables:
  - chunks:  document fragments with pgvector embeddings
  - memories: structured long-term memories with entities, relations, decay
"""

from __future__ import annotations

from sqlalchemy import text

from backend.db import _engine

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id TEXT NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector(1024),
        meta        JSONB DEFAULT '{}',
        created_at  TIMESTAMPTZ DEFAULT now(),
        chunk_index INT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chunks_embedding
        ON chunks USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """,
    """
    CREATE TABLE IF NOT EXISTS memories (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source_type TEXT NOT NULL,
        summary     TEXT NOT NULL,
        entities    JSONB DEFAULT '[]',
        relations   JSONB DEFAULT '[]',
        embedding   vector(1024),
        decay_factor FLOAT DEFAULT 1.0,
        recalled_at TIMESTAMPTZ DEFAULT now(),
        recall_count INT DEFAULT 0,
        meta        JSONB DEFAULT '{}',
        created_at  TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memories_embedding
        ON memories USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """,
]


async def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    async with _engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))
    print("Database initialized (chunks + memories tables ready)")
