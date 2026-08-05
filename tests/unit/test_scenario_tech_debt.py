"""Tests for the tech debt radar scenario compose logic and prompt template."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage


class TestComposeTechDebtPrompt:
    """Verify the tech debt system prompt contains required sections."""

    def test_compose_flags_workarounds_older_than_3_months(self):
        """System prompt must instruct scanning for workarounds > 3 months old."""
        from backend.service.scenarios.tech_debt import TECH_DEBT_SYSTEM_PROMPT

        assert "3" in TECH_DEBT_SYSTEM_PROMPT  # 3 months threshold
        assert "workaround" in TECH_DEBT_SYSTEM_PROMPT.lower()
        assert "临时方案" in TECH_DEBT_SYSTEM_PROMPT

    def test_compose_includes_documentation_gaps(self):
        """System prompt must instruct identifying documentation gaps."""
        from backend.service.scenarios.tech_debt import TECH_DEBT_SYSTEM_PROMPT

        assert "文档缺口" in TECH_DEBT_SYSTEM_PROMPT
        assert "documentation" in TECH_DEBT_SYSTEM_PROMPT.lower()

    def test_compose_includes_auto_resolve_detection(self):
        """System prompt must instruct detecting auto-resolved workarounds."""
        from backend.service.scenarios.tech_debt import TECH_DEBT_SYSTEM_PROMPT

        assert "可能已解决" in TECH_DEBT_SYSTEM_PROMPT
        assert "commit" in TECH_DEBT_SYSTEM_PROMPT.lower()

    def test_compose_includes_priority_section(self):
        """System prompt must instruct top-priority recommendations."""
        from backend.service.scenarios.tech_debt import TECH_DEBT_SYSTEM_PROMPT

        assert "建议优先级" in TECH_DEBT_SYSTEM_PROMPT

    def test_compose_includes_overview_section(self):
        """System prompt must instruct an overview/count summary."""
        from backend.service.scenarios.tech_debt import TECH_DEBT_SYSTEM_PROMPT

        assert "总览" in TECH_DEBT_SYSTEM_PROMPT

    def test_compose_searches_multiple_keywords(self):
        """System prompt must search for multiple workaround-related keywords."""
        from backend.service.scenarios.tech_debt import TECH_DEBT_SYSTEM_PROMPT

        assert "TODO" in TECH_DEBT_SYSTEM_PROMPT
        assert "hotfix" in TECH_DEBT_SYSTEM_PROMPT.lower()
        assert "临时" in TECH_DEBT_SYSTEM_PROMPT

    def test_compose_uses_chinese(self):
        """System prompt must instruct Chinese output."""
        from backend.service.scenarios.tech_debt import TECH_DEBT_SYSTEM_PROMPT

        assert "简体中文" in TECH_DEBT_SYSTEM_PROMPT


class TestComposeTechDebtInvocation:
    """Verify the compose function calls the agent correctly."""

    @pytest.mark.asyncio
    async def test_compose_returns_report(self):
        """Compose function returns agent's response on success."""
        from backend.service.scenarios.tech_debt import compose_tech_debt_report

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "技术债报告内容",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_tech_debt_report()

        assert result == "技术债报告内容"

    @pytest.mark.asyncio
    async def test_compose_passes_no_params(self):
        """Compose function takes no parameters — scans everything."""
        from backend.service.scenarios.tech_debt import compose_tech_debt_report

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "报告",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_tech_debt_report()

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        # Should have system + user message
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    @pytest.mark.asyncio
    async def test_compose_user_message_includes_scan_instructions(self):
        """User message instructs the agent to scan specific categories."""
        from backend.service.scenarios.tech_debt import compose_tech_debt_report

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_tech_debt_report()

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msg = [m for m in messages if isinstance(m, HumanMessage)][0]
        user_text = str(user_msg.content)
        assert "workaround" in user_text.lower()
        assert "3" in user_text  # 3-month threshold

    @pytest.mark.asyncio
    async def test_compose_system_prompt_is_tech_debt_specific(self):
        """System prompt references tech debt radar mode."""
        from backend.service.scenarios.tech_debt import compose_tech_debt_report

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_tech_debt_report()

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        assert "技术债" in str(sys_msgs[0].content)

    @pytest.mark.asyncio
    async def test_compose_handles_agent_error(self):
        """Agent failures return an error message."""
        from backend.service.scenarios.tech_debt import compose_tech_debt_report

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = RuntimeError("Agent crash")

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_tech_debt_report()

        assert "失败" in result
        assert "Agent crash" in result
