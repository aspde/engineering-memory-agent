"""Tests for the onboarding scenario compose logic and prompt template."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage


class TestComposeOnboardingPrompt:
    """Verify the onboarding system prompt contains required sections."""

    def test_compose_ranks_modules_by_memory_count(self):
        """System prompt must instruct ranking modules by memory count."""
        from backend.service.scenarios.onboarding import ONBOARDING_SYSTEM_PROMPT

        assert "核心模块" in ONBOARDING_SYSTEM_PROMPT
        assert "记忆数量" in ONBOARDING_SYSTEM_PROMPT
        assert "历史故障次数" in ONBOARDING_SYSTEM_PROMPT

    def test_compose_includes_reading_order(self):
        """System prompt must instruct a recommended reading order."""
        from backend.service.scenarios.onboarding import ONBOARDING_SYSTEM_PROMPT

        assert "推荐阅读顺序" in ONBOARDING_SYSTEM_PROMPT
        assert "memory ID" in ONBOARDING_SYSTEM_PROMPT

    def test_compose_includes_key_decisions(self):
        """System prompt must instruct listing key architectural decisions."""
        from backend.service.scenarios.onboarding import ONBOARDING_SYSTEM_PROMPT

        assert "关键决策" in ONBOARDING_SYSTEM_PROMPT
        assert "ADR" in ONBOARDING_SYSTEM_PROMPT

    def test_compose_includes_incident_patterns(self):
        """System prompt must instruct summarising recent incident patterns."""
        from backend.service.scenarios.onboarding import ONBOARDING_SYSTEM_PROMPT

        assert "近期故障模式" in ONBOARDING_SYSTEM_PROMPT

    def test_compose_uses_chinese(self):
        """System prompt must instruct Chinese output."""
        from backend.service.scenarios.onboarding import ONBOARDING_SYSTEM_PROMPT

        assert "简体中文" in ONBOARDING_SYSTEM_PROMPT


class TestComposeOnboardingInvocation:
    """Verify the compose function calls the agent correctly."""

    @pytest.mark.asyncio
    async def test_compose_full_scope_uses_generic_message(self):
        """Default scope 'full' searches the entire project."""
        from backend.service.scenarios.onboarding import compose_onboarding_guide

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "Onboarding 全览",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_onboarding_guide(scope="full")

        assert result == "Onboarding 全览"
        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert any("整个项目" in str(m.content) for m in user_msgs)

    @pytest.mark.asyncio
    async def test_compose_scoped_message_includes_entity(self):
        """When scope is an entity name, it appears in the user message."""
        from backend.service.scenarios.onboarding import compose_onboarding_guide

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "模块 Onboarding",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_onboarding_guide(scope="PostgreSQL")

        assert result == "模块 Onboarding"
        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert any("PostgreSQL" in str(m.content) for m in user_msgs)

    @pytest.mark.asyncio
    async def test_compose_passes_system_message(self):
        """Agent is invoked with onboarding-specific system prompt."""
        from backend.service.scenarios.onboarding import compose_onboarding_guide

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_onboarding_guide()

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        assert "Onboarding" in str(sys_msgs[0].content)

    @pytest.mark.asyncio
    async def test_compose_default_scope_is_full(self):
        """When scope is not provided, defaults to full project overview."""
        from backend.service.scenarios.onboarding import compose_onboarding_guide

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "全项目",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_onboarding_guide()

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        assert any("整个项目" in str(m.content) for m in user_msgs)
