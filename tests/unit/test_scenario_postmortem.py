"""Tests for the postmortem scenario compose logic and prompt template."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage


class TestComposePostmortemPrompt:
    """Verify the compose function builds the correct prompt structure."""

    @pytest.mark.asyncio
    async def test_compose_includes_timeline_section_in_system_prompt(self):
        """System prompt must instruct the agent to produce a timeline."""
        from backend.service.scenarios.postmortem import POSTMORTEM_SYSTEM_PROMPT

        assert "时间线" in POSTMORTEM_SYSTEM_PROMPT
        assert "| 时间 | 事件 | 来源 |" in POSTMORTEM_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_compose_includes_similar_incidents_section(self):
        """System prompt must instruct the agent to search for similar incidents."""
        from backend.service.scenarios.postmortem import POSTMORTEM_SYSTEM_PROMPT

        assert "相似故障" in POSTMORTEM_SYSTEM_PROMPT
        assert "共享根因" in POSTMORTEM_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_compose_includes_root_cause_section(self):
        """System prompt must instruct root cause analysis with entity history."""
        from backend.service.scenarios.postmortem import POSTMORTEM_SYSTEM_PROMPT

        assert "根因分析" in POSTMORTEM_SYSTEM_PROMPT
        assert "diff" in POSTMORTEM_SYSTEM_PROMPT.lower()

    @pytest.mark.asyncio
    async def test_compose_includes_improvement_section(self):
        """System prompt must instruct actionable recommendations."""
        from backend.service.scenarios.postmortem import POSTMORTEM_SYSTEM_PROMPT

        assert "改进建议" in POSTMORTEM_SYSTEM_PROMPT
        assert "🔴高" in POSTMORTEM_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_compose_includes_entity_section(self):
        """System prompt must instruct listing related entities."""
        from backend.service.scenarios.postmortem import POSTMORTEM_SYSTEM_PROMPT

        assert "关联实体" in POSTMORTEM_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_compose_uses_chinese(self):
        """System prompt must instruct Chinese output."""
        from backend.service.scenarios.postmortem import POSTMORTEM_SYSTEM_PROMPT

        assert "简体中文" in POSTMORTEM_SYSTEM_PROMPT


class TestComposePostmortemInvocation:
    """Verify the compose function calls the agent correctly."""

    @pytest.mark.asyncio
    async def test_compose_passes_system_and_user_messages(self):
        """Agent is invoked with a SystemMessage and a HumanMessage."""
        from backend.service.scenarios.postmortem import compose_postmortem

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "测试复盘报告",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_postmortem(
                incident_memory_id="00000000-0000-0000-0000-000000000001"
            )

        assert result == "测试复盘报告"
        # Verify agent was called with correct message types
        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        assert any(isinstance(m, SystemMessage) for m in messages)
        assert any(isinstance(m, HumanMessage) for m in messages)

    @pytest.mark.asyncio
    async def test_compose_includes_incident_id_in_user_message(self):
        """When incident_memory_id is provided, it appears in the user message."""
        from backend.service.scenarios.postmortem import compose_postmortem

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "包含故障 ID 的报告",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_postmortem(
                incident_memory_id="abc-123-def"
            )

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(user_msgs) == 1
        assert "abc-123-def" in str(user_msgs[0].content)

    @pytest.mark.asyncio
    async def test_compose_without_incident_id_uses_generic_message(self):
        """When no incident_memory_id, user message asks for recent incidents."""
        from backend.service.scenarios.postmortem import compose_postmortem

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "通用复盘报告",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_postmortem()

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert any("搜索" in str(m.content).lower() for m in user_msgs)

    @pytest.mark.asyncio
    async def test_compose_handles_agent_error(self):
        """Agent failures return an error message, not an exception."""
        from backend.service.scenarios.postmortem import compose_postmortem

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = RuntimeError("LLM timeout")

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_postmortem(incident_memory_id="test-id")

        assert "失败" in result
        assert "LLM timeout" in result

    @pytest.mark.asyncio
    async def test_compose_falls_back_to_last_message(self):
        """When final_response is empty, falls back to last AIMessage content."""
        from backend.service.scenarios.postmortem import compose_postmortem

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "",
            "messages": [
                type("FakeMsg", (), {
                    "content": "",
                    "tool_calls": [{"name": "search", "args": {}}],
                })(),
                type("FakeMsg", (), {
                    "content": "通过 tool 生成的内容",
                    "tool_calls": None,
                })(),
            ],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_postmortem(incident_memory_id="test")

        assert "通过 tool 生成的内容" in result
