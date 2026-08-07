"""Tests for the agent chat API endpoint — includes HITL interrupt/resume."""

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from backend.db import get_session_factory


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
    async def test_silent_tools_excluded_from_tool_calls(self, async_client: AsyncClient, monkeypatch) -> None:
        """write_memory_tool results are excluded from tool_calls (silent tools)."""
        from unittest.mock import AsyncMock
        from langchain_core.messages import ToolMessage

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "Memory has been saved.",
            "messages": [
                ToolMessage(
                    content='{"action":"inserted","summary":"EMA uses PostgreSQL"}',
                    tool_call_id="call_1",
                    name="write_memory_tool",
                ),
            ],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "remember EMA uses PostgreSQL"},
        )
        assert response.status_code == 200
        data = response.json()
        # write_memory_tool is silent — no tool_call trace in response
        assert len(data["tool_calls"]) == 0

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

    @pytest.mark.asyncio
    async def test_agent_chat_times_out(self, async_client: AsyncClient, monkeypatch) -> None:
        """When the agent exceeds AGENT_TIMEOUT, return status='error' with a timeout message."""
        import asyncio
        from unittest.mock import AsyncMock

        from backend.api.routes.agent_routes import config as agent_config

        async def _hang(*args, **kwargs) -> dict:
            await asyncio.sleep(5)
            return {"final_response": "late", "messages": []}

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = _hang

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )
        monkeypatch.setattr(agent_config, "agent_timeout", 0.1)

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "hi", "thread_id": "timeout-thread-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "超时" in data["response"]

    @pytest.mark.asyncio
    async def test_agent_chat_error_does_not_leak_internal_exception(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A failing agent run returns a generic error, never the exception text.

        Internal details (provider keys, DB URLs, stack traces) must stay
        server-side — the response should be a user-facing message only.
        """
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = RuntimeError(
            "secret: postgresql://user:pass@db:5432/ema_prod"
        )

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "hi", "thread_id": "leak-thread-1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert "secret" not in data["response"]
        assert "postgresql://" not in data["response"]
        assert "RuntimeError" not in data["response"]


class TestTokenUsageEndpoint:
    """Tests for the /api/agent/usage cost-monitoring endpoint."""

    @pytest.mark.asyncio
    async def test_usage_returns_zero_initially(self, async_client: AsyncClient) -> None:
        from backend.shared.metrics import reset_structured_failures, reset_token_usage

        reset_token_usage()
        reset_structured_failures()
        response = await async_client.get("/api/agent/usage")
        assert response.status_code == 200
        data = response.json()
        assert data["total_tokens"] == 0
        assert data["by_scenario"] == {}
        assert data["scenarios"] == 0
        assert data["structured_failures"] == {}

    @pytest.mark.asyncio
    async def test_usage_reflects_recorded_tokens(self, async_client: AsyncClient) -> None:
        from types import SimpleNamespace

        from backend.shared.metrics import (
            record_usage,
            reset_token_usage,
        )

        reset_token_usage()
        record_usage("agent_chat", SimpleNamespace(total_tokens=500))
        record_usage("conflict_detection", {"total_tokens": 120})

        response = await async_client.get("/api/agent/usage")
        assert response.status_code == 200
        data = response.json()
        assert data["total_tokens"] == 620
        assert data["by_scenario"]["agent_chat"] == 500
        assert data["by_scenario"]["conflict_detection"] == 120
        assert data["scenarios"] == 2

        reset_token_usage()

    @pytest.mark.asyncio
    async def test_usage_reset_clears_counters(self, async_client: AsyncClient) -> None:
        from types import SimpleNamespace

        from backend.shared.metrics import record_structured_failure, record_usage

        record_usage("x", SimpleNamespace(total_tokens=999))
        record_structured_failure("extraction_entities")
        response = await async_client.post("/api/agent/usage/reset")
        assert response.status_code == 200
        assert response.json() == {"reset": True}

        get_resp = await async_client.get("/api/agent/usage")
        assert get_resp.json()["total_tokens"] == 0
        assert get_resp.json()["structured_failures"] == {}


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
        """When ainvoke raises, return status='error' (not 500) without leaking the
        internal exception text to the client."""
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
        assert "Something went wrong" not in data["response"]
        assert "RuntimeError" not in data["response"]


class TestDeleteThread:
    """Tests for the DELETE /api/agent/thread/{thread_id} endpoint."""

    @pytest.fixture(autouse=True)
    async def _ensure_tables(self) -> None:
        """Ensure conversations table exists before tests (idempotent).

        Matches the pattern in test_connector_routes / test_webhook_routes.
        Without this the direct INSERT below fails when agent_routes runs
        before the connector/webhook suites have called init_db().
        """
        from backend.db.schema import init_db

        await init_db()

    @pytest.mark.asyncio
    async def test_delete_existing_thread(self, async_client: AsyncClient) -> None:
        """Deleting a thread that exists returns deleted=true and removes the record."""
        thread_id = "delete-test-001"
        session_factory = get_session_factory()

        # Insert a conversation row directly (bypassing the noop fixture).
        async with session_factory() as session:
            await session.execute(
                text(
                    "INSERT INTO conversations (thread_id, title) "
                    "VALUES (:tid, :title) "
                    "ON CONFLICT (thread_id) DO UPDATE SET title = :title"
                ),
                {"tid": thread_id, "title": "test"},
            )
            await session.commit()

        # Delete it via the API.
        response = await async_client.delete(f"/api/agent/thread/{thread_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["thread_id"] == thread_id
        assert data["deleted"] is True

        # Verify the record is gone.
        async with session_factory() as session:
            row = await session.execute(
                text("SELECT 1 FROM conversations WHERE thread_id = :tid"),
                {"tid": thread_id},
            )
            assert row.fetchone() is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_thread_returns_404(
        self, async_client: AsyncClient
    ) -> None:
        """Deleting a thread that doesn't exist returns 404.

        Previously skipped on Windows: the HTTPException raised inside the
        verify session's ``__aexit__`` triggered a spurious "Event loop is
        closed".  That path was fixed by 4e2a708 (rollback of the aborted
        checkpoint-table sub-transaction) plus the delete_thread refactor
        that checks existence in a separate read-only session, so the test
        runs everywhere now.
        """
        response = await async_client.delete(
            "/api/agent/thread/00000000-0000-0000-0000-000000000000"
        )
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestStreamFinalAnswer:
    """Tests for the SSE final-answer replay helper."""

    @pytest.mark.asyncio
    async def test_stops_when_client_disconnected(self) -> None:
        """A disconnected SSE client aborts token replay immediately."""
        from backend.api.routes.agent_routes import _stream_final_answer

        class _Disconnected:
            async def is_disconnected(self) -> bool:
                return True

        chunks = [c async for c in _stream_final_answer(_Disconnected(), "hello world")]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_yields_all_chunks_when_connected(self) -> None:
        """While connected, every 4-char chunk is streamed."""
        import json

        from backend.api.routes.agent_routes import _stream_final_answer

        class _Connected:
            async def is_disconnected(self) -> bool:
                return False

        chunks = [c async for c in _stream_final_answer(_Connected(), "hello world")]
        assert len(chunks) == 3  # "hell", "o wo", "rld"
        assert json.loads(chunks[0].split("data: ", 1)[1])["content"] == "hell"

    @pytest.mark.asyncio
    async def test_llm_error_token_reaches_sse_client(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A custom-stream token carrying an LLM error is forwarded as SSE.

        The nodes emit a *generic* error message through the same custom
        stream as real tokens (see ``test_streams_error_text_on_llm_failure``);
        this guards the route half of the chain — the regression was that
        errors were persisted to state but never streamed, leaving the
        assistant message empty on provider failure.
        """
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()

        error_text = "抱歉，当前回答生成失败，请稍后重试。"
        async def _astream(*args, **kwargs):
            yield (), "custom", {"type": "token", "content": error_text}
            yield (), "updates", {"generate_final": {"final_response": error_text}}

        mock_agent.astream = _astream
        mock_agent.aget_state.return_value = None

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat/stream",
            json={"message": "hi"},
        )
        assert response.status_code == 200
        assert '"type": "token"' in response.text
        assert error_text in response.text

    @pytest.mark.asyncio
    async def test_stream_outer_error_does_not_leak_internal_exception(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """An exception thrown while iterating the graph stream yields a generic
        SSE error event — never the internal exception text."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()

        async def _boom(*args, **kwargs):
            raise RuntimeError("secret-db-url")
            yield  # pragma: no cover

        mock_agent.astream = _boom

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat/stream",
            json={"message": "hi"},
        )
        assert response.status_code == 200
        assert '"type": "error"' in response.text
        assert "secret-db-url" not in response.text
