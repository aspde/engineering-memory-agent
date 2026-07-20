"""Pytest fixtures for EMA project."""

import os
from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

# Set test environment before any imports
os.environ["APP_ENV"] = "test"

# Force offline before SentenceTransformer imports are triggered
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient]:
    """Create an async HTTP client for testing FastAPI endpoints.

    Import the FastAPI app lazily so APP_ENV is set first.
    """
    from backend.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
