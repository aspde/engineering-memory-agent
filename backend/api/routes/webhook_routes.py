"""Webhook receiver — validates signatures, dispatches to connectors.

Receive path is split so senders never wait on the (slow) extraction
pipeline:

  Request (fast, ~ms):  verify signature → validate → normalize →
                        write a ``received`` delivery log → return 202.
  Background (slow):    connector.process() (LLM extraction + embedding)
                        → update the delivery log to processed /
                        conflict_pending / failed.

The background dispatch is in-process (``asyncio.create_task``), matching
the same availability tradeoff as the InMemorySaver checkpointer fallback:
a process restart drops in-flight deliveries, and their delivery-log rows
stay ``received`` so the gap is visible in
``GET /api/connectors/{source}/logs``.

A concurrency cap backpressures extraction storms (each in-flight delivery
holds an LLM slot for 10-40s).  Beyond it the request is answered 503; the
sender's retry is safe because the content-hash idempotency gate skips
already-ingested content.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.connectors.registry import get_connector, list_connectors
from backend.db import get_session_factory
from backend.service.conflicts import persist_pending_conflict
from backend.shared.config import config, current_trace_id
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])

# Max simultaneous extraction pipelines.  Unbounded, a CI failure storm
# would open dozens of concurrent provider calls (each 10-40s of LLM work).
# Backed by a plain counter (not an ``asyncio.Semaphore``) so it stays
# event-loop-agnostic like the embed cache in retrieval.py.
_WEBHOOK_MAX_CONCURRENCY = 4
_webhook_active = 0
_webhook_slots_lock = threading.Lock()

# Strong references to in-flight background delivery tasks.
# ``asyncio.create_task`` keeps no strong reference of its own: the loop's
# bookkeeping is weak, so a fire-and-forget task can be garbage-collected at
# its first await — while the delivery is waiting on the LLM (10-40s) — and
# the ``received`` delivery-log row would silently never advance (no retry,
# only a log line).  Holding the task here extends its lifetime until it
# finishes; ``_process_delivery`` drops it in its ``finally``.  (Same pattern
# as ``_normalization_tasks`` in ``backend/service/memory.py``.)
_webhook_tasks: set[asyncio.Task] = set()


class WebhookResponse(BaseModel):
    source: str
    status: str  # "accepted" — processing continues in the background
    delivery_id: str | None = None
    memory_id: str | None = None
    conflict_id: str | None = None
    error: str | None = None


# ── Signature verification ────────────────────────────────────────────


def _verify_signature(source: str, body: bytes, request: Request) -> bool:
    """Verify HMAC-SHA256 signature for *source*.

    Reads the secret from ``WEBHOOK_{SOURCE}_SECRET`` env var.  Accepts
    signatures in these headers (checked in order):

    * ``X-Webhook-Signature``  (generic, format ``sha256=<hex>``)
    * ``X-Hub-Signature-256``  (GitHub-style, format ``sha256=<hex>``)

    When no secret is configured for *source*, the request is accepted
    unauthenticated in non-production environments (convenient for dev /
    internal networks) but **rejected in production** — an unauthenticated
    payload must never be trusted on a deployed system (fail closed).
    """
    secret = os.getenv(f"WEBHOOK_{source.upper()}_SECRET", "")
    if not secret:
        if config.app_env == "production":
            logger.error(
                "Webhook source=%s has no %s secret configured and "
                "APP_ENV=production — rejecting unauthenticated payload",
                source,
                f"WEBHOOK_{source.upper()}_SECRET",
            )
            return False
        # Dev / test: accept unauthenticated (dev-friendly).
        logger.warning(
            "No webhook secret configured for source=%s — accepting "
            "unauthenticated payload (dev/test only; production rejects)",
            source,
        )
        return True

    sig_header: str | None = None
    for header_name in ("X-Webhook-Signature", "X-Hub-Signature-256"):
        candidate = request.headers.get(header_name)
        if candidate:
            sig_header = candidate
            break

    if not sig_header:
        logger.warning("Webhook for source=%s missing signature header", source)
        return False

    # Expect format: sha256=<hexdigest>
    if not sig_header.startswith("sha256="):
        logger.warning("Webhook for source=%s has malformed signature header", source)
        return False

    expected_hex = sig_header[len("sha256="):]
    computed = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(computed, expected_hex)


# ── Delivery log ──────────────────────────────────────────────────────


async def _log_delivery(
    source: str,
    event_type: str | None,
    status: str,
    payload_summary: str,
    memory_id: str | None = None,
    conflict_id: str | None = None,
    error: str | None = None,
) -> str:
    """Insert a delivery-log row, returning its id (the ``delivery_id``)."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                INSERT INTO webhook_logs
                    (source, event_type, status, payload_summary,
                     memory_id, conflict_id, error)
                VALUES
                    (:source, :event_type, :status, :payload_summary,
                     :memory_id, :conflict_id, :error)
                RETURNING id
                """
            ),
            {
                "source": source,
                "event_type": event_type,
                "status": status,
                "payload_summary": payload_summary[:200],
                "memory_id": memory_id,
                "conflict_id": conflict_id,
                "error": error,
            },
        )
        row = result.fetchone()
        delivery_id = str(row[0]) if row else ""
        await session.commit()
    return delivery_id


async def _update_delivery(
    delivery_id: str,
    *,
    status: str,
    memory_id: str | None = None,
    conflict_id: str | None = None,
    error: str | None = None,
) -> None:
    """Record a terminal outcome on the delivery row created at accept time."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                """\
                UPDATE webhook_logs
                SET status = :status,
                    memory_id = :memory_id,
                    conflict_id = :conflict_id,
                    error = :error
                WHERE id = :id
                """
            ),
            {
                "id": delivery_id,
                "status": status,
                "memory_id": memory_id,
                "conflict_id": conflict_id,
                "error": error,
            },
        )
        await session.commit()


# ── Concurrency cap ───────────────────────────────────────────────────


def _try_acquire_slot() -> bool:
    """Reserve one in-flight extraction slot; False when the cap is hit."""
    global _webhook_active
    with _webhook_slots_lock:
        if _webhook_active >= _WEBHOOK_MAX_CONCURRENCY:
            return False
        _webhook_active += 1
        return True


def _release_slot() -> None:
    global _webhook_active
    with _webhook_slots_lock:
        _webhook_active -= 1


# ── Background processing ─────────────────────────────────────────────


async def _process_delivery(
    delivery_id: str,
    source: str,
    connector: Any,
    content: str,
    metadata: dict[str, Any],
) -> None:
    """Run the connector's (slow) extraction pipeline, then record the outcome.

    Runs as an ``asyncio.create_task`` spawned by ``receive_webhook``; the
    delivery-log row was written with ``status='received'`` at accept time.
    Every exit path releases the concurrency slot it consumed.
    """
    # Link every LLM call this ingestion makes to one trace (the agent chat
    # and patrol paths already set a trace id) so ``GET /api/usage/trace/{id}``
    # can replay a whole webhook intake — extraction → grading → merge —
    # end-to-end instead of leaving its calls trace-less.
    trace_token = current_trace_id.set(f"webhook:{delivery_id}")
    try:
        result = await connector.process(content, metadata)
        memory_id: str | None = result.get("id") if isinstance(result, dict) else None
        conflict_id: str | None = None
        if isinstance(result, dict) and result.get("action") == "conflict":
            # No interactive session on a webhook — persist for later HITL
            # instead of silently dropping the conflicting content.
            pending = await persist_pending_conflict(source, result)
            conflict_id = pending["id"]
        await _update_delivery(
            delivery_id,
            status="conflict_pending" if conflict_id else "processed",
            memory_id=memory_id,
            conflict_id=conflict_id,
        )
        logger.info(
            "Webhook delivery %s processed (source=%s status=%s)",
            delivery_id, source, "conflict_pending" if conflict_id else "processed",
        )
    except Exception as exc:
        logger.exception("Webhook background processing failed for delivery %s", delivery_id)
        try:
            await _update_delivery(delivery_id, status="failed", error=str(exc))
        except Exception:
            logger.exception("Failed to record failure outcome for delivery %s", delivery_id)
    finally:
        current_trace_id.reset(trace_token)
        _release_slot()
        _webhook_tasks.discard(asyncio.current_task())


# ── Route ─────────────────────────────────────────────────────────────


@router.post("/{source}", status_code=202, response_model=WebhookResponse)
async def receive_webhook(source: str, request: Request) -> WebhookResponse:
    """Receive a webhook from an external system.

    Synchronous part (signature → validate → normalize) stays on the
    request path and answers fast; the slow connector processing runs in
    the background and its outcome lands in ``webhook_logs`` (queryable
    via ``GET /api/connectors/{source}/logs``).
    """
    # 1. Read raw body for signature verification
    try:
        body = await request.body()
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read request body")

    # 2. Verify signature
    if not _verify_signature(source, body, request):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # 3. Lookup connector
    connector = get_connector(source)
    if connector is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown source '{source}'. Registered sources: "
            + ", ".join(c["source_type"] for c in list_connectors()),
        )

    # 4. Parse JSON body
    try:
        payload: dict[str, Any] = await request.json()
    except Exception:
        await _log_delivery(source, None, "failed", "", error="Invalid JSON body")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # 5. Determine event_type (connector-specific — look for common keys)
    event_type: str | None = None
    if isinstance(payload, dict):
        event_type = payload.get("event_type") or payload.get("event") or payload.get("webhookEvent")

    payload_summary = str(payload)[:200]

    # 6. Validate (pure sync check — stays on the request path)
    try:
        if not connector.validate(payload):
            await _log_delivery(
                source, event_type, "failed", payload_summary,
                error="Payload validation failed — missing required fields",
            )
            raise HTTPException(status_code=400, detail="Payload validation failed")
    except HTTPException:
        raise
    except Exception as exc:
        await _log_delivery(source, event_type, "failed", payload_summary, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Validation error: {exc}") from exc

    # 7. Normalize → build metadata (pure sync transforms — cheap)
    try:
        metadata: dict[str, Any] = connector.build_metadata(payload)
        content: str = connector.normalize(payload)
    except Exception as exc:
        await _log_delivery(source, event_type, "failed", payload_summary, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    # 8. Backpressure — refuse when too many extractions are in flight.
    if not _try_acquire_slot():
        logger.warning(
            "Webhook source=%s rejected: %d extraction(s) already in flight",
            source, _WEBHOOK_MAX_CONCURRENCY,
        )
        raise HTTPException(
            status_code=503,
            detail="Webhook processing queue is full — retry shortly",
        )

    # 9. Record the delivery and dispatch processing to the background.
    try:
        delivery_id = await _log_delivery(source, event_type, "received", payload_summary)
        task = asyncio.create_task(
            _process_delivery(delivery_id, source, connector, content, metadata)
        )
        _webhook_tasks.add(task)
    except Exception as exc:
        _release_slot()
        logger.exception("Failed to dispatch webhook delivery (source=%s)", source)
        raise HTTPException(status_code=500, detail=f"Failed to dispatch: {exc}") from exc

    logger.info("Webhook accepted (source=%s delivery=%s)", source, delivery_id)
    return WebhookResponse(source=source, status="accepted", delivery_id=delivery_id)
