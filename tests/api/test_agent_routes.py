"""Tests for the agent chat API endpoint — includes HITL interrupt/resume."""

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
    async def test_empty_message_accepted(self, async_client: AsyncClient, monkeypatch) -> None:
        """Empty message is allowed (needed for resume-only requests)."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "What would you like me to do?",
            "messages": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "", "thread_id": "test-thread-2"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_message_uses_default(self, async_client: AsyncClient, monkeypatch) -> None:
        """Missing message defaults to empty string (accepted)."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"thread_id": "test-thread-3"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_auto_generates_thread_id(self, async_client: AsyncClient, monkeypatch) -> None:
        """When thread_id is omitted, one is auto-generated."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
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
    async def test_includes_tool_call_traces(self, async_client: AsyncClient, monkeypatch) -> None:
        """Response includes tool call traces from ToolMessages."""
        from unittest.mock import AsyncMock
        from langchain_core.messages import ToolMessage

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "Based on what I found...",
            "messages": [
                ToolMessage(content="Found 1 memory: EMA uses PostgreSQL", tool_call_id="call_1", name="search_memories_tool"),
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
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool"] == "search_memories_tool"

    @pytest.mark.asyncio
    async def test_returns_status_field(self, async_client: AsyncClient, monkeypatch) -> None:
        """Response includes status field with 'completed' value."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "Done.",
            "messages": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"


class TestAgentChatHITL:
    """Tests for Human-in-the-Loop interrupt/resume flow."""

    @pytest.mark.asyncio
    async def test_interrupt_returns_interrupted_status(self, async_client: AsyncClient, monkeypatch) -> None:
        """When agent hits interrupt(), return status='interrupted' with payload."""
        from unittest.mock import AsyncMock

        from langgraph.types import Interrupt

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "__interrupt__": (
                Interrupt(
                    value={
                        "tool_name": "write_memory_tool",
                        "tool_args": {"content": "test"},
                        "summary": "test",
                    }
                ),
            ),
            "messages": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "remember this"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "interrupted"
        assert data["interrupt"] is not None
        assert data["interrupt"]["tool_name"] == "write_memory_tool"

    @pytest.mark.asyncio
    async def test_resume_with_approval_true(self, async_client: AsyncClient, monkeypatch) -> None:
        """Resuming with approved=true calls agent with Command(resume=...)."""
        from unittest.mock import AsyncMock

        from langgraph.types import Command

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "Memory written successfully.",
            "messages": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={
                "message": "",
                "thread_id": "test-hitl-1",
                "resume_data": {"approved": True},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["response"] == "Memory written successfully."

        # Verify Command(resume=...) was passed to ainvoke
        call_args = mock_agent.ainvoke.call_args
        arg = call_args[0][0]
        assert isinstance(arg, Command)
        assert arg.resume == {"approved": True}

    @pytest.mark.asyncio
    async def test_resume_with_approval_false(self, async_client: AsyncClient, monkeypatch) -> None:
        """Resuming with approved=false calls agent with Command(resume=...)."""
        from unittest.mock import AsyncMock

        from langgraph.types import Command

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "I understand, I won't write that memory.",
            "messages": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={
                "thread_id": "test-hitl-2",
                "resume_data": {"approved": False, "reason": "Not needed"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

        call_args = mock_agent.ainvoke.call_args
        arg = call_args[0][0]
        assert isinstance(arg, Command)
        assert arg.resume == {"approved": False, "reason": "Not needed"}

    @pytest.mark.asyncio
    async def test_agent_error_returns_error_status(self, async_client: AsyncClient, monkeypatch) -> None:
        """When ainvoke raises, return status='error' instead of 500."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = RuntimeError("Something went wrong")

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "Something went wrong" in data["response"]
