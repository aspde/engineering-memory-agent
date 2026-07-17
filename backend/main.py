"""EMA — Engineering Memory Agent — FastAPI Application."""

from __future__ import annotations

from fastapi import FastAPI

from backend.api.router import api_router

app = FastAPI(
    title="EMA — Engineering Memory Agent",
    version="0.1.0",
    docs_url="/docs",
)

app.include_router(api_router, prefix="/api")
