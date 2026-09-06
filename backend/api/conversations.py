"""Conversation-row persistence for sidebar history.

Both the chat route (interactive threads) and the scenario route (draft
threads) upsert a ``conversations`` row so a thread appears in the history
sidebar before/while its content streams in.  One helper, one patch point —
tests neutralise it via ``tests/conftest.py`` to keep test traffic out of
the conversations table.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from backend.db import get_session_factory

logger = logging.getLogger(__name__)


async def upsert_conversation(thread_id: str, title: str = "") -> None:
    """Insert or update a conversation row with *title*."""
    try:
        async with get_session_factory()() as session:
            await session.execute(
                text(
                    "INSERT INTO conversations (thread_id, title, updated_at) "
                    "VALUES (:tid, :title, now()) "
                    "ON CONFLICT (thread_id) DO UPDATE SET "
                    "title = COALESCE(NULLIF(:title, ''), conversations.title), "
                    "updated_at = now()"
                ),
                {"tid": thread_id, "title": title},
            )
            await session.commit()
    except Exception:
        logger.warning("Failed to upsert conversation", exc_info=True)
