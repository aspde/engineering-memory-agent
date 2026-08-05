"""Tests for the code review scenario compose logic and prompt template."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage


class TestComposeCodeReviewPrompt:
    """Verify the code review system prompt contains required sections."""

    def test_compose_flags_high_risk_files(self):
        """System prompt must instruct risk classification."""
        from backend.service.scenarios.code_review import CODE_REVIEW_SYSTEM_PROMPT

        assert "高风险文件" in CODE_REVIEW_SYSTEM_PROMPT
        assert "🔴" in CODE_REVIEW_SYSTEM_PROMPT
        assert "🟡" in CODE_REVIEW_SYSTEM_PROMPT
        assert "🟢" in CODE_REVIEW_SYSTEM_PROMPT

    def test_compose_includes_decision_consistency_check(self):
        """System prompt must instruct checking PR against ADRs."""
        from backend.service.scenarios.code_review import CODE_REVIEW_SYSTEM_PROMPT

        assert "决策一致性" in CODE_REVIEW_SYSTEM_PROMPT
        assert "ADR" in CODE_REVIEW_SYSTEM_PROMPT

    def test_compose_includes_review_suggestions(self):
        """System prompt must instruct actionable review suggestions."""
        from backend.service.scenarios.code_review import CODE_REVIEW_SYSTEM_PROMPT

        assert "审查建议" in CODE_REVIEW_SYSTEM_PROMPT

    def test_compose_uses_chinese(self):
        """System prompt must instruct Chinese output."""
        from backend.service.scenarios.code_review import CODE_REVIEW_SYSTEM_PROMPT

        assert "简体中文" in CODE_REVIEW_SYSTEM_PROMPT


class TestComposeCodeReviewInvocation:
    """Verify the compose function calls the agent correctly."""

    @pytest.mark.asyncio
    async def test_compose_passes_diff_and_description(self):
        """Both PR diff and description are included in the user message."""
        from backend.service.scenarios.code_review import compose_review_context

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "审查结果",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_review_context(
                pr_diff="--- a/foo.py\n+++ b/foo.py\n+new line",
                pr_description="修复连接池泄漏",
            )

        assert result == "审查结果"
        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        user_text = str(user_msgs[0].content)
        assert "修复连接池泄漏" in user_text
        assert "foo.py" in user_text

    @pytest.mark.asyncio
    async def test_compose_truncates_long_diff(self):
        """Very long diffs are truncated to fit context."""
        from backend.service.scenarios.code_review import compose_review_context

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
        }

        long_diff = "x" * 12000

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_review_context(pr_diff=long_diff)

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        user_text = str(user_msgs[0].content)
        # Should be truncated — original 12000 chars reduced to ~8000
        assert len(user_text) < len(long_diff) + 500
        assert "已截断" in user_text

    @pytest.mark.asyncio
    async def test_compose_truncates_long_description(self):
        """Very long PR descriptions are truncated."""
        from backend.service.scenarios.code_review import compose_review_context

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
        }

        long_desc = "y" * 3000

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_review_context(pr_description=long_desc)

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        user_text = str(user_msgs[0].content)
        assert "已截断" in user_text

    @pytest.mark.asyncio
    async def test_compose_empty_input_returns_help(self):
        """When both diff and description are empty, returns guidance."""
        from backend.service.scenarios.code_review import compose_review_context

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "请提供 PR diff 或 PR 描述以生成审查上下文。",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            result = await compose_review_context()
            assert "PR diff" in result or "PR 描述" in result
            # Agent must be invoked so the checkpoint is populated
            mock_agent.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_compose_passes_system_message(self):
        """Agent is invoked with a SystemMessage (scenario-specific prompt)."""
        from backend.service.scenarios.code_review import compose_review_context

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "ok",
            "messages": [],
        }

        with patch(
            "backend.service.agent_service.get_agent",
            return_value=mock_agent,
        ):
            await compose_review_context(pr_diff="diff content")

        call_args = mock_agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        sys_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        assert "代码审查" in str(sys_msgs[0].content)
