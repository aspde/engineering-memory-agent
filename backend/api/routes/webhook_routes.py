"""Webhook receiver — validates signatures, dispatches to connectors."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.connectors.registry import get_connector, list_connectors
from backend.db import get_session_factory
from backend.service.conflicts import persist_pending_conflict
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhook"])


class WebhookResponse(BaseModel):
    source: str
    status: str  # "processed" | "failed" | "conflict_pending"
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

    Returns True if no secret is configured for *source* (unauthenticated
    mode, convenient for dev / internal networks).
    """
    secret = os.getenv(f"WEBHOOK_{source.upper()}_SECRET", "")
    if not secret:
        # No secret configured — accept unauthenticated (dev-friendly).
        logger.debug("No webhook secret configured for source=%s — skipping verification", source)
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
) -> None:
    """Write a row to ``webhook_logs``."""
    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(
                text(
                    """\
                    INSERT INTO webhook_logs
                        (source, event_type, status, payload_summary,
                         memory_id, conflict_id, error)
                    VALUES
                        (:source, :event_type, :status, :payload_summary,
                         :memory_id, :conflict_id, :error)
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
            await session.commit()
    except Exception:
        logger.exception("Failed to write webhook_log for source=%s", source)


# ── Route ─────────────────────────────────────────────────────────────


@router.post("/{source}", response_model=WebhookResponse)
async def receive_webhook(source: str, request: Request) -> WebhookResponse:
    """Receive a webhook from an external system.

    Pipeline: verify signature → lookup connector → validate →
    normalize → process → log.
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

    # 6. Validate
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

    # 7. Build metadata (connector-specific hook)
    metadata: dict[str, Any] = connector.build_metadata(payload)

    # 8. Normalize → process
    try:
        content = connector.normalize(payload)
        result = await connector.process(content, metadata)
        memory_id: str | None = result.get("id") if isinstance(result, dict) else None
        conflict_id: str | None = None
        if isinstance(result, dict) and result.get("action") == "conflict":
            # No interactive session on a webhook — persist for later HITL
            # instead of silently dropping the conflicting content.
            pending = await persist_pending_conflict(source, result)
            conflict_id = pending["id"]
    except HTTPException:
        raise
    except Exception as exc:
        await _log_delivery(source, event_type, "failed", payload_summary, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    # 9. Log success
    if conflict_id:
        await _log_delivery(
            source, event_type, "conflict_pending", payload_summary,
            conflict_id=conflict_id,
        )
        return WebhookResponse(
            source=source, status="conflict_pending", conflict_id=conflict_id
        )

    await _log_delivery(source, event_type, "processed", payload_summary, memory_id=memory_id)

    return WebhookResponse(source=source, status="processed", memory_id=memory_id)
