"""EMA — Engineering Memory Agent — FastAPI Application."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager

# ── Windows: psycopg 3 (used by AsyncPostgresSaver) requires
#    SelectorEventLoop — ProActorEventLoop is incompatible.
if sys.platform == "win32":
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)

from fastapi import FastAPI

from backend.api.router import api_router
from backend.db import close_db
from backend.db.schema import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB tables and PostgresSaver checkpoint table."""
    await init_db()

    # Create checkpoint table for conversation persistence
    try:
        from backend.service.agent_service import _setup_checkpointer

        await _setup_checkpointer()
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to setup checkpointer — conversations will be ephemeral"
        )

    yield
    await close_db()

    # Close checkpointer pool on shutdown
    try:
        from backend.service.agent_service import _close_checkpointer

        await _close_checkpointer()
    except Exception:
        import logging

        logging.getLogger(__name__).warning("Failed to close checkpointer pool")


app = FastAPI(
    title="EMA — Engineering Memory Agent",
    version="0.1.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api")
