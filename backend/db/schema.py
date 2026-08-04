"""Database schema — tables created via raw SQL to keep things simple.

Tables:
  - chunks:  document fragments with pgvector embeddings
  - memories: structured long-term memories with entities, relations, decay
"""

from __future__ import annotations

from sqlalchemy import text

from backend.db import engine

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
        created_at  TIMESTAMPTZ DEFAULT now(),
        updated_at  TIMESTAMPTZ
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memories_embedding
        ON memories USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """,
    # Migration: add updated_at to existing memories tables that lack it.
    # CREATE TABLE IF NOT EXISTS won't mutate an existing table, so we
    # apply this ALTER for databases created before the column was added.
    """
    DO $$
    BEGIN
        ALTER TABLE memories ADD COLUMN updated_at TIMESTAMPTZ;
    EXCEPTION WHEN duplicate_column THEN NULL;
    END $$;
    """,
    """
    DO $$
    BEGIN
        ALTER TABLE memories ADD COLUMN deleted_at TIMESTAMPTZ;
    EXCEPTION WHEN duplicate_column THEN NULL;
    END $$;
    """,
    """
    CREATE TABLE IF NOT EXISTS entities (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name            TEXT NOT NULL,
        canonical_name  TEXT NOT NULL,
        type            TEXT NOT NULL,
        embedding       vector(1024),
        first_seen_at   TIMESTAMPTZ DEFAULT now(),
        UNIQUE(canonical_name, type)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_entities_embedding
        ON entities USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_entities (
        memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
        entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
        PRIMARY KEY (memory_id, entity_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        thread_id   TEXT NOT NULL UNIQUE,
        title       TEXT NOT NULL DEFAULT '',
        updated_at  TIMESTAMPTZ DEFAULT now(),
        created_at  TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_conversations_updated
        ON conversations (updated_at DESC)
    """,
    # ── Phase 2: connectors ─────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS webhook_logs (
        id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source          TEXT NOT NULL,
        event_type      TEXT,
        status          TEXT NOT NULL DEFAULT 'received',
        payload_summary TEXT,
        memory_id       UUID,
        error           TEXT,
        created_at      TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_webhook_logs_source
        ON webhook_logs (source, created_at DESC)
    """,
]


async def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))
    print("Database initialized (chunks + memories + conversations tables ready)")
