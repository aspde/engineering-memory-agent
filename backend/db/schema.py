"""Database schema — tables created via raw SQL to keep things simple.

Tables:
  - chunks:  document fragments with pgvector embeddings
  - memories: structured long-term memories with entities, relations, decay

Embedding columns are ``vector(<dimension>)`` where *dimension* comes from
``config.embedding.dimension`` (:func:`build_schema_statements`).  When the
configured embedding model changes dimension, :func:`init_db` migrates the
existing columns instead of failing every write: embeddings are derived data,
so the column is emptied, resized, and re-embedded from the stored text via
``python -m scripts.reembed_embeddings``.
"""

from __future__ import annotations

from sqlalchemy import text

from backend.db import get_engine
from backend.shared.config import config


def _resize_statement(table: str, index: str, dimension: int) -> str:
    """ALTER *table*'s embedding column to ``vector(dimension)`` when it differs.

    pgvector vectors only fit a same-or-smaller column type; resizing either
    direction requires the column to be empty.  Embeddings are derived data
    (recomputable from stored text via ``python -m scripts.reembed_embeddings``), so the
    migration drops the index, empties the column, resizes it, and rebuilds
    the index — a dimension change is never a silent truncation/padding.
    """
    return f"""
    DO $$
    DECLARE
        cur TEXT;
    BEGIN
        SELECT format_type(a.atttypid, a.atttypmod) INTO cur
        FROM pg_attribute a
        WHERE a.attrelid = '{table}'::regclass AND a.attname = 'embedding';
        IF cur IS DISTINCT FROM 'vector({dimension})' AND cur LIKE 'vector%' THEN
            EXECUTE 'DROP INDEX IF EXISTS {index}';
            EXECUTE 'UPDATE {table} SET embedding = NULL';
            EXECUTE 'ALTER TABLE {table} ALTER COLUMN embedding TYPE vector({dimension})';
            EXECUTE 'CREATE INDEX IF NOT EXISTS {index} ON {table} USING hnsw (embedding vector_cosine_ops)';
        END IF;
    END $$;
    """


def build_schema_statements(dimension: int) -> list[str]:
    """DDL for the full schema, embedding columns of *dimension* dims.

    Exported separately from ``init_db`` so tests can assert on the generated
    SQL for any dimension without touching a database.
    """
    v = f"vector({dimension})"
    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id TEXT NOT NULL,
            content     TEXT NOT NULL,
            embedding   {v},
            meta        JSONB DEFAULT '{{}}',
            created_at  TIMESTAMPTZ DEFAULT now(),
            chunk_index INT NOT NULL,
            content_hash TEXT
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """,
        _resize_statement("chunks", "idx_chunks_embedding", dimension),
        f"""
        CREATE TABLE IF NOT EXISTS memories (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_type TEXT NOT NULL,
            summary     TEXT NOT NULL,
            entities    JSONB DEFAULT '[]',
            relations   JSONB DEFAULT '[]',
            content_hash TEXT,
            embedding   {v},
            decay_factor FLOAT DEFAULT 1.0,
            recalled_at TIMESTAMPTZ DEFAULT now(),
            recall_count INT DEFAULT 0,
            meta        JSONB DEFAULT '{{}}',
            created_at  TIMESTAMPTZ DEFAULT now(),
            updated_at  TIMESTAMPTZ
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memories_embedding
            ON memories USING hnsw (embedding vector_cosine_ops)
        """,
        _resize_statement("memories", "idx_memories_embedding", dimension),
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
        f"""
        CREATE TABLE IF NOT EXISTS entities (
            id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name            TEXT NOT NULL,
            canonical_name  TEXT NOT NULL,
            type            TEXT NOT NULL,
            embedding       {v},
            first_seen_at   TIMESTAMPTZ DEFAULT now(),
            UNIQUE(canonical_name, type)
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_entities_embedding
            ON entities USING hnsw (embedding vector_cosine_ops)
        """,
        _resize_statement("entities", "idx_entities_embedding", dimension),
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
            dismissed_findings TEXT[],
            started_at        TIMESTAMPTZ DEFAULT now(),
            completed_at      TIMESTAMPTZ
        )
        """,
        # Dismissed-finding keys are *finding identifiers*, not memory UUIDs —
        # the LLM's finding JSON carries no per-finding ``id``, so the frontend
        # keys dismissals by ``<group>-<index>`` (and contradiction findings by
        # their memory pair).  Migrate any pre-existing UUID[] column (which
        # every real dismiss failed to populate, so it is empty in practice).
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
        # ── Embedding index migration: ivfflat → HNSW ─────────────────────
        # HNSW (pgvector >= 0.5) has no cluster-centroid dependency, so it
        # stays accurate on small corpora where ivfflat's lists=100 spread
        # probes over mostly-empty clusters (the seed-010 lesson).  Fresh
        # databases get HNSW via the CREATE INDEX statements above; this
        # block converts databases created while the schema still used
        # ivfflat — replacing each embedding index once, then a no-op.
        """
        DO $$
        DECLARE
            t text;
            i text;
        BEGIN
            FOREACH t IN ARRAY ARRAY['chunks', 'memories', 'entities'] LOOP
                i := CASE t WHEN 'chunks' THEN 'idx_chunks_embedding'
                            WHEN 'memories' THEN 'idx_memories_embedding'
                            ELSE 'idx_entities_embedding' END;
                IF EXISTS (
                    SELECT 1 FROM pg_indexes
                    WHERE indexname = i AND indexdef LIKE '%ivfflat%'
                ) THEN
                    EXECUTE format('DROP INDEX %I', i);
                    EXECUTE format('CREATE INDEX %I ON %I USING hnsw (embedding vector_cosine_ops)', i, t);
                END IF;
            END LOOP;
        END $$;
        """,
        # ── LLM usage tracing (observability) ────────────────────────────
        # One row per LLM call, written by the provider layer into an
        # in-memory buffer and flushed to this table in batches (see
        # ``backend/service/usage.py``).  ``trace_id`` links the calls of one
        # agent run end-to-end; ``thread_id`` links across requests in one
        # conversation.  ``scenario`` is the cost-observability tag from the
        # metrics module.  Rows are append-only; retention is an operational
        # concern (delete via SQL / a later job), not a schema concern.
        """
        CREATE TABLE IF NOT EXISTS llm_usage (
            id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            seq            BIGSERIAL,
            trace_id       TEXT,
            thread_id      TEXT,
            scenario       TEXT NOT NULL,
            provider       TEXT NOT NULL,
            model          TEXT NOT NULL,
            input_tokens   INT,
            output_tokens  INT,
            total_tokens   INT,
            cache_read_tokens     INT,
            cache_creation_tokens INT,
            latency_ms     INT,
            status         TEXT NOT NULL DEFAULT 'success',
            error          TEXT,
            prompt_chars   INT,
            response_chars INT,
            created_at     TIMESTAMPTZ DEFAULT now()
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_llm_usage_created
            ON llm_usage (created_at DESC)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_llm_usage_trace
            ON llm_usage (trace_id)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_llm_usage_thread
            ON llm_usage (thread_id, created_at DESC)
        """,
        # Sampled prompt/response text for post-hoc quality analysis (see
        # ``backend/service/usage.py``).  NULL = not sampled; error calls are
        # always sampled, success calls at ``USAGE_SAMPLE_RATE``.  Added via
        # ALTER so databases created before the columns existed stay valid.
        """
        DO $$
        BEGIN
            ALTER TABLE llm_usage ADD COLUMN prompt_sample TEXT;
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
        """,
        """
        DO $$
        BEGIN
            ALTER TABLE llm_usage ADD COLUMN response_sample TEXT;
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
        """,
        # Prompt-cache accounting columns (see ``metrics.extract_tokens``):
        # cached input tokens read from / created into a provider cache, so
        # cost summaries can apply the discounted cache-read rate.  Added via
        # ALTER so databases created before the columns existed stay valid.
        """
        DO $$
        BEGIN
            ALTER TABLE llm_usage ADD COLUMN cache_read_tokens INT;
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
        """,
        """
        DO $$
        BEGIN
            ALTER TABLE llm_usage ADD COLUMN cache_creation_tokens INT;
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
        """,
        # Transport-retry accounting (see ``resilience.py``): how many times
        # the provider call was attempted before this row's outcome — 1 = a
        # clean first try, 3 = two tenacity retries were swallowed before
        # success (or the final failure).  NULL for rows written before this
        # column existed; ``record_call`` defaults to None when a caller
        # doesn't report attempts.  Added via ALTER so databases created
        # before the column existed stay valid.
        """
        DO $$
        BEGIN
            ALTER TABLE llm_usage ADD COLUMN attempts INT;
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
        """,
        # ── Pending-conflict queue: patrol (inspection) conflicts ─────────
        # Patrol contradictions are two *already-stored* memories (A, B), unlike
        # ingestion conflicts (existing in store + new not yet written).  The
        # ``conflict_type`` column routes resolution to the patrol pipeline; the
        # ``peer_id`` column names the memory that loses the arbitration (B) so
        # ``resolve_patrol_conflict`` can soft-delete it.
        """
        DO $$
        BEGIN
            ALTER TABLE pending_conflicts ADD COLUMN conflict_type TEXT NOT NULL DEFAULT 'ingestion';
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
        """,
        """
        DO $$
        BEGIN
            ALTER TABLE pending_conflicts ADD COLUMN peer_id UUID;
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
        """,
        # An unordered pair (A,B) or (B,A) is queued at most once while pending.
        # Resolved rows leave this partial index, so a *fresh* occurrence of the
        # same pair can re-queue — but keep_both's "arbitrated" record is a
        # resolved row, and re-queueing after it is suppressed in the service
        # layer (persist_patrol_conflict checks resolved pairs explicitly).
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_pending_conflicts_patrol_pair
            ON pending_conflicts (LEAST(existing_id, peer_id), GREATEST(existing_id, peer_id))
            WHERE status = 'pending' AND conflict_type = 'patrol'
        """,
    ]
    return statements


async def init_db(dimension: int | None = None) -> None:
    """Create tables and indexes if they don't exist.

    *dimension* defaults to ``config.embedding.dimension`` — tests may pass an
    explicit value to exercise other dimensions without touching config.
    """
    statements = build_schema_statements(
        dimension if dimension is not None else config.embedding.dimension
    )
    async with get_engine().begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        for stmt in statements:
            await conn.execute(text(stmt))
    print("Database initialized (chunks + memories + conversations tables ready)")
