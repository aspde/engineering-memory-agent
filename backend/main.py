"""EMA — Engineering Memory Agent — FastAPI Application."""

from __future__ import annotations

import os

# ── Offline mode & BLAS env vars — MUST be set BEFORE any import that
# may transitively pull in transformers / huggingface_hub / numpy / torch.
# langchain-core → transformers → huggingface_hub reads HF_HUB_OFFLINE
# at import time; setting it later has no effect.
for _k, _v in {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}.items():
    os.environ[_k] = _v

import asyncio
import sys
from contextlib import asynccontextmanager

# ── Windows: psycopg 3 (used by AsyncPostgresSaver) requires
#    SelectorEventLoop — ProActorEventLoop is incompatible.
#    uvicorn creates its own event loop after importing this module,
#    so set_event_loop here has no lasting effect.  On Windows the
#    checkpointer falls back to InMemorySaver — acceptable for dev.
#    Production should run in a Linux container where ProactorEventLoop
#    doesn't exist and psycopg async works natively.

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.router import api_router
from backend.db import close_db
from backend.db.schema import init_db

# Paths relative to backend/main.py
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
_FRONTEND_STATIC = Path(__file__).resolve().parent.parent / "frontend" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB tables and PostgresSaver checkpoint table."""
    await init_db()

    # Register connectors at startup.  Connectors missing required
    # configuration (API keys, etc.) are still registered but flagged
    # as "pending" so the frontend can show their status.
    try:
        from backend.connectors.registry import register_connector
        from backend.connectors.ci import CIConnector
        from backend.connectors.feishu import FeishuConnector
        from backend.connectors.pingcode import PingCodeConnector

        _pingcode_secret = os.getenv("WEBHOOK_PINGCODE_SECRET", "")
        register_connector(
            "pingcode",
            PingCodeConnector(),
            status="active" if _pingcode_secret else "pending",
        )
        _ci_secret = os.getenv("WEBHOOK_CI_SECRET", "")
        register_connector(
            "ci",
            CIConnector(),
            status="active" if _ci_secret else "pending",
        )
        _feishu_secret = os.getenv("WEBHOOK_FEISHU_SECRET", "")
        register_connector(
            "feishu",
            FeishuConnector(),
            status="active" if _feishu_secret else "pending",
        )
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to register connectors — webhook endpoints will be unavailable."
        )

    # Create checkpoint table for conversation persistence
    try:
        from backend.service.agent_service import _setup_checkpointer

        await _setup_checkpointer()
    except Exception as _exc:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to setup checkpointer — conversations will be ephemeral. "
            "Error: %s",
            _exc,
        )

    # Warm the embedding model in a background thread so the first
    # request doesn't block for 30+ seconds loading BGE-M3.
    try:
        from backend.service.embedding_service import get_embedding_provider

        async def _warm_embedding() -> None:
            try:
                await asyncio.to_thread(get_embedding_provider)
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "Background embedding warmup failed — model will "
                    "load on first use (may delay the first request)."
                )

        asyncio.create_task(_warm_embedding())
    except Exception:
        pass

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

# ── SPA static files & fallback ──
# Mount asset directories so the browser can load JS/CSS/images.
if _FRONTEND_DIST.joinpath("assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIST / "assets")), name="assets")

if _FRONTEND_STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_STATIC)), name="static")


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    """Catch-all: serve index.html for any non-API route (SPA client-side routing)."""
    index_path = _FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
    return {"detail": "Frontend not built — run: cd frontend && npm run build"}
