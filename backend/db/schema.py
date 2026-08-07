"""Database schema — tables created via raw SQL to keep things simple.

Tables:
  - chunks:  document fragments with pgvector embeddings
  - memories: structured long-term memories with entities, relations, decay
"""

from __future__ import annotations

from sqlalchemy import text

from backend.db import get_engine

_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id TEXT NOT NULL,
        content     TEXT NOT NULL,
        embedding   vector(1024),
        meta        JSONB DEFAULT '{}',
        created_at  TIMESTAMPTZ DEFAULT now(),
        chunk_index INT NOT NULL,
        content_hash TEXT
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
        content_hash TEXT,
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
        conflict_id     UUID,
        error           TEXT,
        created_at      TIMESTAMPTZ DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_webhook_logs_source
        ON webhook_logs (source, created_at DESC)
    """,
    # ── Phase 3: proactive agent ─────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS patrol_logs (
        id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        patrol_type       TEXT NOT NULL,
        trigger           TEXT NOT NULL,
        status            TEXT NOT NULL DEFAULT 'running',
        findings          JSONB,
        dismissed_findings UUID[],
        started_at        TIMESTAMPTZ DEFAULT now(),
        completed_at      TIMESTAMPTZ
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_patrol_logs_type_time
        ON patrol_logs (patrol_type, started_at DESC)
    """,
    # ── Hybrid search: jieba tokens column on chunks ────────────────
    # Stores Python-side jieba segmentation results as a text array so
    # sparse_search can use GIN-indexed ``tokens && :q_tokens`` instead of
    # loading every chunk into Python (O(N) → O(log N)).  The tokens are
    # produced by ``retrieval._tokenize`` (jieba + stopword filter) at
    # write time and backfilled for pre-existing rows.
    """
    DO $$
    BEGIN
        ALTER TABLE chunks ADD COLUMN tokens TEXT[] DEFAULT '{}';
    EXCEPTION WHEN duplicate_column THEN NULL;
    END $$;
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chunks_tokens ON chunks USING GIN(tokens)
    """,
    # ── Content-hash idempotency ────────────────────────────────────
    # content_hash is the exact-duplicate key for re-ingestion: re-ingesting
    # the same source (same commit, same doc, repeated webhook) must not
    # create duplicate memories/chunks.  CREATE TABLE IF NOT EXISTS won't
    # mutate existing tables, so apply ALTERs for databases created before
    # the columns existed (pre-existing rows get NULL → exempt from the
    # unique index).
    """
    DO $$
    BEGIN
        ALTER TABLE chunks ADD COLUMN content_hash TEXT;
    EXCEPTION WHEN duplicate_column THEN NULL;
    END $$;
    """,
    """
    DO $$
    BEGIN
        ALTER TABLE memories ADD COLUMN content_hash TEXT;
    EXCEPTION WHEN duplicate_column THEN NULL;
    END $$;
    """,
    # Scoped by document_id so the same chunk text in two different
    # documents is NOT treated as a duplicate.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_doc_content_hash
        ON chunks (document_id, content_hash)
    """,
    # Unique among *live* rows only: a soft-deleted memory keeps its
    # content_hash (tombstones are never rewritten), while the idempotency
    # gate filters deleted_at IS NULL — so a deleted row must not block a
    # later re-ingest from re-using the hash.  ``DROP INDEX`` clears the
    # pre-partial version created before this refinement.
    """
    DROP INDEX IF EXISTS uq_memories_content_hash
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_content_hash_live
        ON memories (content_hash) WHERE deleted_at IS NULL
    """,
    # ── Pending-conflict queue (webhook/connector HITL) ─────────────
    # Webhook deliveries have no interactive session, so a detected memory
    # conflict is persisted here and resolved later by a human through the
    # same resolve_conflict() path the agent interrupt uses.  ``deferred``
    # holds the _deferred payload (extracted, embedding, source_type,
    # metadata, content_hash) needed to apply the resolution.
    """
    CREATE TABLE IF NOT EXISTS pending_conflicts (
        id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        source           TEXT NOT NULL,
        source_type      TEXT,
        existing_id      UUID NOT NULL,
        existing_summary TEXT NOT NULL,
        new_summary      TEXT NOT NULL,
        deferred         JSONB NOT NULL,
        status           TEXT NOT NULL DEFAULT 'pending',
        resolution       TEXT,
        created_at       TIMESTAMPTZ DEFAULT now(),
        resolved_at      TIMESTAMPTZ
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pending_conflicts_open
        ON pending_conflicts (status, created_at DESC)
    """,
    # A given conflict (same existing memory + same new content) is queued at
    # most once while pending: webhooks are at-least-once, and each redelivery
    # of an identical conflicting payload must not stack another row (it would
    # multiply HITL work and, resolved in the wrong order, could collide on the
    # content_hash unique index).  Resolved rows leave the partial index, so a
    # *fresh* occurrence of the same content re-queues normally.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_conflicts_open_dedup
        ON pending_conflicts (existing_id, (deferred->>'content_hash'))
        WHERE status = 'pending'
    """,
    # Migration: add conflict_id to existing webhook_logs tables.
    """
    DO $$
    BEGIN
        ALTER TABLE webhook_logs ADD COLUMN conflict_id UUID;
    EXCEPTION WHEN duplicate_column THEN NULL;
    END $$;
    """,
]


async def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    async with get_engine().begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for stmt in _STATEMENTS:
            await conn.execute(text(stmt))
    print("Database initialized (chunks + memories + conversations tables ready)")
