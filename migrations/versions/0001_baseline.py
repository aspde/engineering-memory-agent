"""Baseline migration — the EMA schema in its current, complete form.

This captures the final structure that ``backend/db/schema.py``'s
``build_schema_statements()`` (all CREATE TABLEs plus every historical
``DO $$`` patch applied on top) produces today.  It is the version-1
snapshot: everything before this commit is considered "already migrated",
and every *future* schema change must be a new migration.

Two deliberate choices:

- **``IF NOT EXISTS`` semantics.**  Live databases were built by the
  old ``init_db()`` before this migration existed, so they already contain
  every table.  A fresh database gets them all from this migration.  Using
  ``CREATE TABLE IF NOT EXISTS`` / ``CREATE INDEX IF NOT EXISTS`` makes
  the first ``upgrade head`` safe for both: existing tables are skipped and
  the version table is stamped to head.
- **Embedding columns use the placeholder dimension 1024** (BGE-M3, the
  default in ``config.embedding.dimension``).  The real dimension is a
  *runtime* choice (the configured embedding model), so it cannot live in a
  static migration.  ``init_db()`` aligns the live column to the configured
  dimension afterwards (the same resize the old ``init_db`` did), so this
  placeholder is only ever the initial value.

revision: 0001_baseline
revises:
"""

from __future__ import annotations

from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None

VECTOR_DIM = 1024  # placeholder — aligned to the configured model by init_db


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── chunks (document fragments with pgvector embeddings) ──────────
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id TEXT NOT NULL,
            content     TEXT NOT NULL,
            embedding   vector({VECTOR_DIM}),
            meta        JSONB DEFAULT '{{}}',
            created_at  TIMESTAMPTZ DEFAULT now(),
            chunk_index INT NOT NULL,
            content_hash TEXT,
            tokens      TEXT[] DEFAULT '{{}}'
        )
        """
    )
    # Historical column additions — ``CREATE TABLE IF NOT EXISTS`` won't
    # mutate a table that predates a column, so live databases built before
    # these columns existed get them here, idempotently (a fresh database
    # already has them from the CREATE above, so these are no-ops).
    op.execute(
        "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tokens TEXT[] DEFAULT '{}'"
    )
    op.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT")

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_tokens ON chunks USING GIN(tokens)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_doc_content_hash "
        "ON chunks (document_id, content_hash)"
    )

    # ── memories (structured long-term memories with decay) ──────────
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS memories (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_type TEXT NOT NULL,
            summary     TEXT NOT NULL,
            entities    JSONB DEFAULT '[]',
            relations   JSONB DEFAULT '[]',
            content_hash TEXT,
            embedding   vector({VECTOR_DIM}),
            decay_factor FLOAT DEFAULT 1.0,
            recalled_at TIMESTAMPTZ DEFAULT now(),
            recall_count INT DEFAULT 0,
            meta        JSONB DEFAULT '{{}}',
            created_at  TIMESTAMPTZ DEFAULT now(),
            updated_at  TIMESTAMPTZ,
            deleted_at  TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ"
    )
    op.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS content_hash TEXT")

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memories_embedding "
        "ON memories USING hnsw (embedding vector_cosine_ops)"
    )
    # Replace the pre-partial unique index (created before the soft-delete
    # refinement): a tombstone must not block a later re-ingest re-using
    # the hash.
    op.execute("DROP INDEX IF EXISTS uq_memories_content_hash")
    # Partial unique index: only live rows participate, so a soft-deleted
    # tombstone never blocks a later re-ingest re-using the hash.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_content_hash_live "
        "ON memories (content_hash) WHERE deleted_at IS NULL"
    )

    # ── entities (normalized entity graph) ───────────────────────────
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS entities (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name            TEXT NOT NULL,
            canonical_name  TEXT NOT NULL,
            type            TEXT NOT NULL,
            embedding       vector({VECTOR_DIM}),
            first_seen_at   TIMESTAMPTZ DEFAULT now(),
            UNIQUE(canonical_name, type)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_entities_embedding "
        "ON entities USING hnsw (embedding vector_cosine_ops)"
    )

    # ── memory_entities (M:N join) ───────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_entities (
            memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
            entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
            PRIMARY KEY (memory_id, entity_id)
        )
        """
    )

    # ── conversations (chat history) ─────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            thread_id   TEXT NOT NULL UNIQUE,
            title       TEXT NOT NULL DEFAULT '',
            updated_at  TIMESTAMPTZ DEFAULT now(),
            created_at  TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversations_updated "
        "ON conversations (updated_at DESC)"
    )

    # ── webhook_logs (connector delivery log) ────────────────────────
    op.execute(
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
        """
    )
    op.execute(
        "ALTER TABLE webhook_logs ADD COLUMN IF NOT EXISTS conflict_id UUID"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_webhook_logs_source "
        "ON webhook_logs (source, created_at DESC)"
    )

    # ── patrol_logs (proactive-agent run history) ────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS patrol_logs (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            patrol_type       TEXT NOT NULL,
            trigger           TEXT NOT NULL,
            status            TEXT NOT NULL DEFAULT 'running',
            findings          JSONB,
            dismissed_findings TEXT[],
            started_at        TIMESTAMPTZ DEFAULT now(),
            completed_at      TIMESTAMPTZ
        )
        """
    )
    # Dismissed-finding keys are *finding identifiers*, not memory UUIDs —
    # the LLM's finding JSON carries no per-finding ``id``, so the frontend
    # keys dismissals by ``<group>-<index>`` (and contradiction findings by
    # their memory pair).  Migrate any pre-existing UUID[] column (which
    # every real dismiss failed to populate, so it is empty in practice).
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'patrol_logs'
                  AND column_name = 'dismissed_findings'
                  AND udt_name = '_uuid'
            ) THEN
                ALTER TABLE patrol_logs
                    ALTER COLUMN dismissed_findings TYPE TEXT[]
                    USING dismissed_findings::text[];
            END IF;
        END $$;
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_patrol_logs_type_time "
        "ON patrol_logs (patrol_type, started_at DESC)"
    )

    # ── pending_conflicts (webhook/connector + patrol HITL queue) ────
    op.execute(
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
            conflict_type    TEXT NOT NULL DEFAULT 'ingestion',
            peer_id          UUID,
            created_at       TIMESTAMPTZ DEFAULT now(),
            resolved_at      TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "ALTER TABLE pending_conflicts ADD COLUMN IF NOT EXISTS "
        "conflict_type TEXT NOT NULL DEFAULT 'ingestion'"
    )
    op.execute(
        "ALTER TABLE pending_conflicts ADD COLUMN IF NOT EXISTS peer_id UUID"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_conflicts_open "
        "ON pending_conflicts (status, created_at DESC)"
    )
    # One open conflict per (existing memory, new content) pair.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_conflicts_open_dedup "
        "ON pending_conflicts (existing_id, (deferred->>'content_hash')) "
        "WHERE status = 'pending'"
    )
    # Patrol contradictions are unordered memory pairs — queued once while open.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_conflicts_patrol_pair "
        "ON pending_conflicts "
        "(LEAST(existing_id, peer_id), GREATEST(existing_id, peer_id)) "
        "WHERE status = 'pending' AND conflict_type = 'patrol'"
    )

    # ── llm_usage (persisted LLM cost/observability rows) ────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_usage (
            id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            seq                  BIGSERIAL,
            trace_id             TEXT,
            thread_id            TEXT,
            scenario             TEXT NOT NULL,
            provider             TEXT NOT NULL,
            model                TEXT NOT NULL,
            input_tokens         INT,
            output_tokens        INT,
            total_tokens         INT,
            cache_read_tokens     INT,
            cache_creation_tokens INT,
            latency_ms           INT,
            status               TEXT NOT NULL DEFAULT 'success',
            error                TEXT,
            prompt_chars         INT,
            response_chars       INT,
            prompt_sample        TEXT,
            response_sample      TEXT,
            attempts             INT,
            created_at           TIMESTAMPTZ DEFAULT now()
        )
        """
    )
    op.execute("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS prompt_sample TEXT")
    op.execute("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS response_sample TEXT")
    op.execute(
        "ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS cache_read_tokens INT"
    )
    op.execute(
        "ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS cache_creation_tokens INT"
    )
    op.execute("ALTER TABLE llm_usage ADD COLUMN IF NOT EXISTS attempts INT")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_created "
        "ON llm_usage (created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_trace ON llm_usage (trace_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_llm_usage_thread "
        "ON llm_usage (thread_id, created_at DESC)"
    )


def downgrade() -> None:
    """Drop every table the baseline created, newest-first (CASCADE so
    cross-table references resolve regardless of order)."""
    for table in (
        "llm_usage",
        "pending_conflicts",
        "patrol_logs",
        "webhook_logs",
        "conversations",
        "memory_entities",
        "entities",
        "memories",
        "chunks",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
