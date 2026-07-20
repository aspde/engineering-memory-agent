"""Tests for the agent chat API endpoint."""

import pytest
from httpx import AsyncClient


class TestAgentChat:
    @pytest.mark.asyncio
    async def test_chat_returns_200(self, async_client: AsyncClient, monkeypatch) -> None:
        """A valid request returns ChatResponse with status 200."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "Hello! How can I help?",
            "messages": [],
            "retrieved_chunks": [],
            "retrieved_memories": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "Hi there", "thread_id": "test-thread-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hello! How can I help?"
        assert data["thread_id"] == "test-thread-1"

    @pytest.mark.asyncio
    async def test_empty_message_returns_422(self, async_client: AsyncClient) -> None:
        """Empty message should fail Pydantic validation."""
        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "", "thread_id": "test-thread-2"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_message_returns_422(self, async_client: AsyncClient) -> None:
        """Missing required field should fail validation."""
        response = await async_client.post(
            "/api/agent/chat",
            json={"thread_id": "test-thread-3"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_auto_generates_thread_id(self, async_client: AsyncClient, monkeypatch) -> None:
        """When thread_id is omitted, one is auto-generated."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
            "retrieved_chunks": [],
            "retrieved_memories": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "hello"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"]  # should be auto-generated UUID

    @pytest.mark.asyncio
    async def test_includes_sources(self, async_client: AsyncClient, monkeypatch) -> None:
        """Response includes sources from retrieval."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "Based on what I found...",
            "messages": [],
            "retrieved_chunks": [
                {"content": "def foo(): pass", "score": 0.92}
            ],
            "retrieved_memories": [
                {"summary": "EMA uses PostgreSQL", "weighted_score": 0.88}
            ],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "what is EMA?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert data["sources"][0]["type"] == "chunk"
        assert data["sources"][1]["type"] == "memory"
