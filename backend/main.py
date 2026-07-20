"""EMA — Engineering Memory Agent — FastAPI Application."""

from __future__ import annotations

from contextlib import asynccontextmanager

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


app = FastAPI(
    title="EMA — Engineering Memory Agent",
    version="0.1.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api")
