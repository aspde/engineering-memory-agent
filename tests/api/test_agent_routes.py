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
            lambda *a, **k: mock_agent,
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
            lambda *a, **k: mock_agent,
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
            lambda *a, **k: mock_agent,
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
            lambda *a, **k: mock_agent,
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
            lambda *a, **k: mock_agent,
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
    async def test_force_write_passes_flag_and_returns_memory_write(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """force_write=True is forwarded into the graph input, and the write's
        outcome is surfaced as ``memory_write`` on the response."""
        from unittest.mock import AsyncMock

        from langchain_core.messages import ToolMessage

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "已记录",
            "messages": [
                ToolMessage(
                    content='{"action":"inserted","summary":"端口改为 8080"}',
                    tool_call_id="call_1",
                    name="write_memory_tool",
                ),
            ],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={
                "message": "记住：端口改为 8080",
                "thread_id": "fw-1",
                "force_write": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["memory_write"] == {
            "action": "inserted",
            "summary": "端口改为 8080",
        }
        # The flag reached the graph input.
        input_args = mock_agent.ainvoke.call_args.args[0]
        assert input_args.get("force_write") is True

    @pytest.mark.asyncio
    async def test_no_force_write_no_memory_write(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """Without force_write, the response carries memory_write=None."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "你好",
            "messages": [],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "你好", "thread_id": "fw-2"},
        )
        assert response.status_code == 200
        assert response.json()["memory_write"] is None

    @pytest.mark.asyncio
    async def test_previous_turn_write_not_reported(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A write_memory_tool result from an earlier turn must not be reported
        as this turn's memory_write."""
        from unittest.mock import AsyncMock

        from langchain_core.messages import HumanMessage, ToolMessage

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [
                HumanMessage(content="old turn"),
                ToolMessage(
                    content='{"action":"inserted","summary":"old"}',
                    tool_call_id="call_1",
                    name="write_memory_tool",
                ),
                HumanMessage(content="this turn"),
            ],
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "this turn", "thread_id": "fw-3", "force_write": True},
        )
        assert response.status_code == 200
        assert response.json()["memory_write"] is None

    @pytest.mark.asyncio
    async def test_force_write_conflict_status_interrupted(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A force-write that hits a conflict surfaces as an interrupt — the
        same HITL path as any other write."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "__interrupt__": [
                type(
                    "_I",
                    (),
                    {
                        "value": {
                            "type": "conflict",
                            "new_summary": "EMA uses MySQL",
                            "existing_id": "mem-1",
                            "existing_summary": "EMA uses PostgreSQL",
                            "options": ["keep_existing", "overwrite", "merge", "keep_both"],
                            "deferred": {},
                        }
                    },
                )()
            ]
        }

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={
                "message": "记住：EMA uses MySQL",
                "thread_id": "fw-4",
                "force_write": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "interrupted"
        assert data["interrupt"]["type"] == "conflict"

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
            lambda *a, **k: mock_agent,
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
        """When the agent exceeds AGENT_TIMEOUT, return 504 with a timeout message."""
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
            lambda *a, **k: mock_agent,
        )
        monkeypatch.setattr(agent_config, "agent_timeout", 0.1)

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "hi", "thread_id": "timeout-thread-1"},
        )
        assert response.status_code == 504
        data = response.json()
        assert "超时" in data["detail"]

    @pytest.mark.asyncio
    async def test_agent_chat_error_does_not_leak_internal_exception(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A failing agent run returns 502 with a generic error, never the exception text.

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
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "hi", "thread_id": "leak-thread-1"},
        )
        assert response.status_code == 502
        data = response.json()
        assert "secret" not in data["detail"]
        assert "postgresql://" not in data["detail"]
        assert "RuntimeError" not in data["detail"]


class TestAgentConcurrencyCap:
    """Beyond MAX_AGENT_CONCURRENCY simultaneous runs the request is refused
    with 503 — never queued behind long ReAct loops."""

    async def _request(self, async_client: AsyncClient, path: str = "/api/agent/chat") -> object:
        return await async_client.post(
            path,
            json={"message": "hi", "thread_id": "cap-thread-1"},
        )

    @pytest.mark.asyncio
    async def test_chat_refused_when_cap_reached(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """Zero slots → the non-streaming chat endpoint answers 503."""
        from backend.shared import config as config_mod

        monkeypatch.setattr(config_mod.config, "max_agent_concurrency", 0)

        response = await self._request(async_client)
        assert response.status_code == 503
        assert "繁忙" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_stream_refused_when_cap_reached(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """Zero slots → the streaming endpoint answers a plain HTTP 503, not
        an SSE error event."""
        from backend.shared import config as config_mod

        monkeypatch.setattr(config_mod.config, "max_agent_concurrency", 0)

        response = await self._request(async_client, "/api/agent/chat/stream")
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_refused_request_does_not_upsert_conversation(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A refused request returns 503 before any DB write — no conversation
        upsert runs for a request that will never execute."""
        from unittest.mock import AsyncMock

        from backend.shared import config as config_mod

        monkeypatch.setattr(config_mod.config, "max_agent_concurrency", 0)
        mock_upsert = AsyncMock()
        monkeypatch.setattr(
            "backend.api.routes.agent_routes._upsert_conversation", mock_upsert
        )

        response = await self._request(async_client)
        assert response.status_code == 503
        mock_upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slot_released_after_run(self, async_client: AsyncClient, monkeypatch) -> None:
        """A completed run releases its slot — the next request is admitted."""
        from unittest.mock import AsyncMock

        from backend.shared import config as config_mod

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
        }
        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )
        monkeypatch.setattr(config_mod.config, "max_agent_concurrency", 1)

        first = await self._request(async_client)
        assert first.status_code == 200
        # The first run's finally released the slot — a second is admitted.
        second = await self._request(async_client)
        assert second.status_code == 200


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
            lambda *a, **k: mock_agent,
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
            lambda *a, **k: mock_agent,
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
            lambda *a, **k: mock_agent,
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
        """When ainvoke raises, return 502 (not 200) without leaking the internal
        exception text to the client."""
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = RuntimeError("Something went wrong")

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat",
            json={"message": "test"},
        )
        assert response.status_code == 502
        data = response.json()
        assert "Something went wrong" not in data["detail"]
        assert "RuntimeError" not in data["detail"]


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


class TestAgentStream:
    """Tests for the /api/agent/chat/stream SSE route (first run and resume)."""

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
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat/stream",
            json={"message": "hi"},
        )
        assert response.status_code == 200
        assert '"type": "token"' in response.text
        assert error_text in response.text

    @pytest.mark.asyncio
    async def test_custom_error_event_reaches_sse_client_as_own_event(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A custom-stream error event (emitted after partial tokens) is
        forwarded as its own SSE error event, not glued onto a token line.

        Regression guard for the #4 fix: nodes emit ``{"type": "error"}``
        through the custom stream instead of appending the apology to the
        partial answer; the route must forward it as a distinct ``error``
        event so the client renders it separately.
        """
        import json
        from unittest.mock import AsyncMock

        mock_agent = AsyncMock()

        async def _astream(*args, **kwargs):
            yield (), "custom", {"type": "token", "content": "半句答案"}
            yield (), "custom", {
                "type": "error",
                "message": "抱歉，当前回答生成失败，请稍后重试。",
            }
            yield (), "updates", {"generate_final": {"final_response": ""}}

        mock_agent.astream = _astream
        mock_agent.aget_state.return_value = None

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat/stream",
            json={"message": "hi"},
        )
        assert response.status_code == 200

        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.strip().startswith("data: ")
        ]
        types = [e.get("type") for e in events]
        # The partial answer and the apology travel as separate events —
        # the error is never appended to a token.
        assert "token" in types
        assert "error" in types
        token_event = next(e for e in events if e.get("type") == "token")
        error_event = next(e for e in events if e.get("type") == "error")
        assert token_event["content"] == "半句答案"
        assert "抱歉" not in token_event["content"]
        assert "抱歉，当前回答生成失败，请稍后重试。" in error_event["message"]

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
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat/stream",
            json={"message": "hi"},
        )
        assert response.status_code == 200
        assert '"type": "error"' in response.text
        assert "secret-db-url" not in response.text

    @pytest.mark.asyncio
    async def test_stream_resume_yields_live_tokens(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """Resume streams real token deltas through the same custom-stream
        pipeline as a new message (not a replayed final answer)."""
        from unittest.mock import AsyncMock

        from langgraph.types import Command

        mock_agent = AsyncMock()

        astream_calls: list = []

        async def _astream(*args, **kwargs):
            astream_calls.append((args, kwargs))
            yield (), "updates", {"check_approval": None}
            yield (), "custom", {"type": "token", "content": "记忆"}
            yield (), "custom", {"type": "token", "content": "已写入。"}
            yield (), "updates", {"generate_final": {"final_response": "记忆已写入。"}}

        mock_agent.astream = _astream
        mock_agent.aget_state.return_value = None

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat/stream",
            json={
                "message": "",
                "thread_id": "test-hitl-1",
                "resume_data": {"approved": True},
            },
        )
        assert response.status_code == 200
        assert '"type": "token"' in response.text
        assert "记忆" in response.text
        assert "已写入。" in response.text

        # Resume must drive the graph via Command(resume=...), not ainvoke.
        call_args, call_kwargs = astream_calls[0]
        arg = call_args[0]
        assert isinstance(arg, Command)
        assert arg.resume == {"approved": True}
        assert call_kwargs["stream_mode"] == ["updates", "custom"]

    @pytest.mark.asyncio
    async def test_stream_resume_second_interrupt(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A resume that hits another interrupt surfaces it as an SSE event and
        stops the stream (no stale tokens after the interrupt)."""
        from unittest.mock import AsyncMock

        from langgraph.types import Interrupt

        mock_agent = AsyncMock()

        async def _astream(*args, **kwargs):
            yield (), "updates", {"check_conflict": None}
            yield (), "updates", {
                "__interrupt__": (
                    Interrupt(
                        value={"type": "conflict", "new_summary": "new"},
                    ),
                ),
            }
            yield (), "custom", {"type": "token", "content": "IGNORED"}

        mock_agent.astream = _astream
        mock_agent.aget_state.return_value = None

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat/stream",
            json={
                "message": "",
                "thread_id": "test-hitl-1",
                "resume_data": {"resolution": "merge"},
            },
        )
        assert response.status_code == 200
        assert '"type": "interrupt"' in response.text
        assert '"type": "conflict"' in response.text
        # Stream returns at the interrupt — the trailing token never reaches the client.
        assert "IGNORED" not in response.text

    @pytest.mark.asyncio
    async def test_stream_disconnect_closes_stream_and_marks_interrupted(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        """A disconnected SSE client must close the underlying agent stream and
        mark the thread interrupted.

        Regression: the disconnect path only ``return``ed — the ``agent.astream``
        generator was abandoned (LangGraph "Task was destroyed" warnings) and
        ``_mark_interrupted_thread`` was never called.  The checkpoint kept the
        user's message, so retrying the same question appended a second
        identical user turn the model read as an independent ask.
        """
        from unittest.mock import AsyncMock

        from langchain_core.messages import SystemMessage

        import backend.api.routes.agent_routes as routes_mod

        # The SSE client goes away as soon as streaming begins.
        monkeypatch.setattr(
            routes_mod, "_is_disconnected", AsyncMock(return_value=True)
        )

        closed: list[bool] = []

        class _FakeStream:
            """Minimal async iterator standing in for the graph's astream.

            Records ``aclose()`` so the test can assert the route explicitly
            closes the underlying stream on disconnect.
            """

            def __init__(self, events):
                self._events = iter(events)
                self._closed = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._events)
                except StopIteration:
                    raise StopAsyncIteration from None

            async def aclose(self):
                if not self._closed:
                    self._closed = True
                    closed.append(True)

        mock_agent = AsyncMock()
        mock_agent.astream = lambda *a, **k: _FakeStream([
            ((), "updates", {"call_llm": None}),
        ])

        monkeypatch.setattr(
            "backend.api.routes.agent_routes.get_agent_for_thread",
            lambda *a, **k: mock_agent,
        )

        response = await async_client.post(
            "/api/agent/chat/stream",
            json={"message": "hi", "thread_id": "disconnect-thread-1"},
        )
        assert response.status_code == 200

        # The graph stream generator was explicitly closed — no dangling task.
        assert closed == [True]

        # The thread was marked interrupted (a SystemMessage appended to state)
        # so a retry reads as a continuation, not a duplicate user turn.
        mock_agent.aupdate_state.assert_awaited_once()
        update = mock_agent.aupdate_state.call_args.args[1]
        msgs = update.get("messages", [])
        assert any(
            isinstance(m, SystemMessage) and "interrupted" in str(m.content)
            for m in msgs
        )
