"""Unit tests for tests/eval/task_executors.py — the real agent graph.

Drives ``make_task_runner`` (which builds the production ``build_agent_graph``)
with fake ``@tool`` functions and a patched ``backend.agent.nodes.get_llm_provider``,
so the tests exercise the *real graph machinery* — ReAct loop routing,
HITL interrupts and auto-resume, max_steps force-termination, timeout — with
zero LLM / DB access.  Tools are real LangChain tools with fake bodies; the
provider streams scripted ``tool_calls`` / content events via the shared
``tests._fake_llm`` helpers.

The graph names a gated write tool ``write_memory_tool`` to trigger the
production approval gate (``CHAT_APPROVAL_TOOLS``); the runner auto-approves.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.tools import tool

from backend.agent.tool_envelope import build_tool_envelope
from tests._fake_llm import (
    content_stream,
    sequential_stream,
    text_stream,
    tool_call_stream,
)
from tests.eval.task_executors import auto_approve_resume, make_task_runner


@pytest.fixture(autouse=True)
def _disable_auto_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-memory is off for these runs (the factory also disables it)."""
    from backend.shared import config as config_mod

    monkeypatch.setattr(config_mod.config, "auto_memory_enabled", False)


@tool
async def fake_search(query: str) -> str:
    """Search memories."""
    return build_tool_envelope(
        f"Found: {query}",
        [{"id": "c4a11b2e", "type": "memory", "summary": query[:50]}],
    )


@tool
async def write_memory_tool(content: str) -> str:
    """Write a memory (gated name → exercises the approval interrupt)."""
    return json.dumps(
        {"id": "m-eval-1", "action": "inserted", "summary": content[:30]},
        ensure_ascii=False,
    )


@tool
async def conflict_write_tool(content: str) -> str:
    """Write tool whose result carries a conflict (→ check_conflict interrupt)."""
    return json.dumps(
        {
            "action": "conflict",
            "summary": content[:30],
            "existing_id": "m-existing",
            "existing_summary": "旧记忆",
            "entities": [],
            "relations": [],
            "_deferred": {"summary": content[:30]},
        },
        ensure_ascii=False,
    )


def _patch_provider(monkeypatch: pytest.MonkeyPatch, provider: Mock) -> None:
    import backend.agent.nodes as nodes_mod

    monkeypatch.setattr(nodes_mod, "get_llm_provider", lambda: provider)


class TestAutoApproveResume:
    def test_single_call_approves(self) -> None:
        assert auto_approve_resume({"tool_name": "write_memory_tool"}) == {
            "approved": True
        }

    def test_batch_approves_every_call(self) -> None:
        payload = {
            "type": "batch",
            "calls": [{"id": "c1"}, {"id": "c2"}],
        }
        assert auto_approve_resume(payload) == {
            "calls": [{"id": "c1", "approved": True}, {"id": "c2", "approved": True}]
        }

    def test_conflict_keeps_existing(self) -> None:
        payload = {"type": "conflict", "existing_id": "m1"}
        assert auto_approve_resume(payload) == {"resolution": "keep_existing"}


class TestTaskRunnerGraph:
    @pytest.mark.asyncio
    async def test_multi_step_trajectory(self, monkeypatch) -> None:
        """search → write (gated → auto-approved) → final answer.

        Exercises the full production graph: ReAct loop, safe-tool
        pass-through, approval interrupt + auto-resume, conflict node
        pass-through, and the tool-turn final synthesis.
        """
        provider = AsyncMock()
        provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {"id": "call_s", "name": "fake_search", "args": {"query": "选型"}},
            ]),
            tool_call_stream([
                {
                    "id": "call_w",
                    "name": "write_memory_tool",
                    "args": {"content": "嵌入模型用 BGE-M3"},
                },
            ]),
            content_stream("已完成搜索与写入。"),
        )
        provider.chat_stream = text_stream("向量检索后端选用了 pgvector，同时已记住该结论。")
        _patch_provider(monkeypatch, provider)

        runner = make_task_runner(tools=[fake_search, write_memory_tool])
        outcome = await runner("查选型并记住")

        assert [c["name"] for c in outcome.tool_calls] == [
            "fake_search",
            "write_memory_tool",
        ]
        assert outcome.n_steps == 3
        assert outcome.within_budget is True
        assert outcome.had_error is False
        assert outcome.answer == "向量检索后端选用了 pgvector，同时已记住该结论。"
        # The search envelope's source id the model saw is available for citation.
        assert "c4a11b2e" in outcome.source_ids
        assert "Found: 选型" in outcome.context_text
        # The write envelope is not a retrieval envelope → no false source ids.
        assert len(outcome.source_ids) == 1

    @pytest.mark.asyncio
    async def test_conflict_interrupt_auto_resolved(self, monkeypatch) -> None:
        """A conflict write pauses at check_conflict; the runner keeps existing."""
        from backend.agent import nodes as nodes_mod

        async def _fake_resolve(resolution: str, existing_id: str, deferred: dict):
            return {
                "id": existing_id,
                "action": "conflict_resolved",
                "resolution": resolution,
            }

        monkeypatch.setattr(nodes_mod, "resolve_conflict", _fake_resolve)

        provider = AsyncMock()
        provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {
                    "id": "call_c",
                    "name": "write_memory_tool",
                    "args": {"content": "新记忆"},
                },
            ]),
            content_stream("记忆已写入。"),
        )
        provider.chat_stream = text_stream("冲突已按保留旧记忆处理。")
        _patch_provider(monkeypatch, provider)

        runner = make_task_runner(tools=[conflict_write_tool])
        outcome = await runner("写入一条可能冲突的记忆")

        assert outcome.had_error is False
        assert "write_memory_tool" in [c["name"] for c in outcome.tool_calls]
        assert outcome.answer == "冲突已按保留旧记忆处理。"

    @pytest.mark.asyncio
    async def test_max_steps_force_terminates(self, monkeypatch) -> None:
        """The agent never stops calling tools → max_steps routes to final."""
        provider = AsyncMock()
        provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {"id": "c1", "name": "fake_search", "args": {"query": "q1"}},
            ]),
            tool_call_stream([
                {"id": "c2", "name": "fake_search", "args": {"query": "q2"}},
            ]),
        )
        provider.chat_stream = text_stream("循环被强制终止，给出回答。")
        _patch_provider(monkeypatch, provider)

        runner = make_task_runner(tools=[fake_search], max_steps=2)
        outcome = await runner("反复搜索")

        assert outcome.n_steps == 2
        assert outcome.within_budget is False  # hit the budget → force-terminated
        assert outcome.answer == "循环被强制终止，给出回答。"
        assert outcome.had_error is False

    @pytest.mark.asyncio
    async def test_no_tool_refrain(self, monkeypatch) -> None:
        """Plain chat: the graph reuses call_llm's output, no tool call."""
        provider = AsyncMock()
        provider.chat_raw_stream = sequential_stream(
            content_stream("好的，明白了，谢谢！"),
        )
        _patch_provider(monkeypatch, provider)

        runner = make_task_runner(tools=[fake_search])
        outcome = await runner("好的，谢谢！")

        assert outcome.tool_calls == []
        assert outcome.answer == "好的，明白了，谢谢！"
        assert outcome.had_error is False
        assert outcome.n_steps == 1

    @pytest.mark.asyncio
    async def test_provider_hang_times_out(self, monkeypatch) -> None:
        """A hung provider is interrupted by the task timeout → had_error."""

        async def _hang(*args, **kwargs):
            await asyncio.sleep(10)
            yield {"type": "content", "text": "never"}  # pragma: no cover

        provider = Mock()
        provider.chat_raw_stream = Mock(side_effect=_hang)
        _patch_provider(monkeypatch, provider)

        runner = make_task_runner(tools=[fake_search], timeout=0.2)
        outcome = await runner("挂起的任务")

        assert outcome.had_error is True
        assert outcome.tool_calls == []
        # A timeout is a loop-discipline failure, not "within budget".
        assert outcome.within_budget is False

    @pytest.mark.asyncio
    async def test_rejected_approval_routes_back_to_llm(self, monkeypatch) -> None:
        """A resume policy that rejects sends the LLM back to re-decide."""
        from tests.eval.task_executors import make_task_runner as _make

        def _reject(payload):
            return {"approved": False, "reason": "eval simulated rejection"}

        provider = AsyncMock()
        provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {
                    "id": "call_r",
                    "name": "write_memory_tool",
                    "args": {"content": "不该写入"},
                },
            ]),
            content_stream("抱歉，我不能执行这个写入。"),
        )
        provider.chat_stream = text_stream("已拒绝写入请求。")
        _patch_provider(monkeypatch, provider)

        runner = _make(
            tools=[write_memory_tool],
            resume=_reject,
        )
        outcome = await runner("执行写操作")

        # The write was rejected → the LLM re-decided and answered instead.
        assert outcome.had_error is False
        assert outcome.answer == "已拒绝写入请求。"
