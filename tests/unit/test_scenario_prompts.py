"""Tests for the four vertical scenarios' prompts and compose functions.

Structure mirrors the production code: the system prompts live in the
central registry (``backend.service.prompts``, key ``scenario.*``) and every
compose function is a thin wrapper over the shared ``invoke_scenario_agent``
(whose contract — approval bypass, interrupt surfacing, error handling,
last-message fallback — is covered in ``test_scenario_invoke.py``).  So this
module only asserts what is scenario-specific:

- each system prompt contains its required sections (parametrised)
- each compose function forwards its distinctive parameters into the user
  message (incident id, onboarding scope, PR diff truncation, …)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

# ── System-prompt content (parametrised over the central registry) ────


def _prompt_text(key: str) -> str:
    from backend.service.prompts import get_prompt

    return get_prompt(key)[1]


@pytest.mark.parametrize(
    ("key", "required_markers"),
    [
        # postmortem: timeline, similar incidents, root cause, improvements, entities.
        (
            "scenario.postmortem",
            ["时间线", "| 时间 | 事件 | 来源 |", "相似故障", "共享根因",
             "根因分析", "diff", "改进建议", "🔴高", "关联实体", "简体中文"],
        ),
        # onboarding: module ranking, reading order, key decisions, incident patterns.
        (
            "scenario.onboarding",
            ["核心模块", "记忆数量", "历史故障次数", "推荐阅读顺序", "memory ID",
             "关键决策", "ADR", "近期故障模式", "简体中文"],
        ),
        # code review: risk flags, decision consistency, suggestions.
        (
            "scenario.code_review",
            ["高风险文件", "🔴", "🟡", "🟢", "决策一致性", "ADR", "审查建议", "简体中文"],
        ),
        # tech debt: workaround scan (>3 months), doc gaps, auto-resolve, priorities.
        (
            "scenario.tech_debt",
            ["3", "workaround", "临时方案", "文档缺口", "documentation",
             "可能已解决", "commit", "建议优先级", "总览", "TODO", "hotfix",
             "临时", "简体中文"],
        ),
    ],
)
def test_scenario_prompt_contains_required_sections(key, required_markers):
    text = _prompt_text(key)
    missing = [m for m in required_markers if m not in text]
    assert not missing, f"{key} prompt missing markers: {missing}"


# ── Compose helpers ──────────────────────────────────────────────────


def _mock_agent(final_response: str = "ok") -> AsyncMock:
    agent = AsyncMock()
    agent.ainvoke.return_value = {"final_response": final_response, "messages": []}
    return agent


def _user_message(agent: AsyncMock) -> str:
    """The HumanMessage content the compose call sent to the agent."""
    from langchain_core.messages import HumanMessage

    messages = agent.ainvoke.call_args[0][0]["messages"]
    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    assert len(user_msgs) == 1
    return str(user_msgs[0].content)


async def _invoke(compose, mock_agent, **kwargs):
    with patch(
        "backend.service.agent_service.get_agent", return_value=mock_agent
    ):
        return await compose(**kwargs)


# ── Postmortem ───────────────────────────────────────────────────────


class TestComposePostmortem:
    @pytest.mark.asyncio
    async def test_incident_id_flows_into_user_message(self):
        from backend.service.scenarios.postmortem import compose_postmortem

        agent = _mock_agent("复盘报告")
        result = await _invoke(compose_postmortem, agent, incident_memory_id="abc-123-def")
        assert result == "复盘报告"
        assert "abc-123-def" in _user_message(agent)

    @pytest.mark.asyncio
    async def test_no_incident_id_asks_for_recent_search(self):
        from backend.service.scenarios.postmortem import compose_postmortem

        agent = _mock_agent()
        await _invoke(compose_postmortem, agent)
        assert "搜索" in _user_message(agent)


# ── Onboarding ───────────────────────────────────────────────────────


class TestComposeOnboarding:
    @pytest.mark.asyncio
    async def test_full_scope_mentions_whole_project(self):
        from backend.service.scenarios.onboarding import compose_onboarding_guide

        agent = _mock_agent("Onboarding 全览")
        result = await _invoke(compose_onboarding_guide, agent)
        assert result == "Onboarding 全览"
        assert "整个项目" in _user_message(agent)

    @pytest.mark.asyncio
    async def test_entity_scope_names_the_entity(self):
        from backend.service.scenarios.onboarding import compose_onboarding_guide

        agent = _mock_agent("模块指南")
        await _invoke(compose_onboarding_guide, agent, scope="PostgreSQL")
        assert "PostgreSQL" in _user_message(agent)

    @pytest.mark.asyncio
    async def test_system_prompt_is_onboarding_specific(self):
        from langchain_core.messages import SystemMessage

        from backend.service.scenarios.onboarding import compose_onboarding_guide

        agent = _mock_agent()
        await _invoke(compose_onboarding_guide, agent)
        sys_msgs = [
            m for m in agent.ainvoke.call_args[0][0]["messages"]
            if isinstance(m, SystemMessage)
        ]
        assert len(sys_msgs) == 1
        assert "Onboarding" in str(sys_msgs[0].content)


# ── Code review ──────────────────────────────────────────────────────


class TestComposeCodeReview:
    @pytest.mark.asyncio
    async def test_diff_and_description_both_flow_into_user_message(self):
        from backend.service.scenarios.code_review import compose_review_context

        agent = _mock_agent("审查结果")
        result = await _invoke(
            compose_review_context,
            agent,
            pr_diff="--- a/foo.py\n+++ b/foo.py\n+new line",
            pr_description="修复连接池泄漏",
        )
        assert result == "审查结果"
        user_text = _user_message(agent)
        assert "修复连接池泄漏" in user_text
        assert "foo.py" in user_text

    @pytest.mark.asyncio
    async def test_long_diff_is_truncated(self):
        from backend.service.scenarios.code_review import compose_review_context

        agent = _mock_agent()
        long_diff = "x" * 12000
        await _invoke(compose_review_context, agent, pr_diff=long_diff)
        user_text = _user_message(agent)
        assert len(user_text) < len(long_diff) + 500
        assert "已截断" in user_text

    @pytest.mark.asyncio
    async def test_long_description_is_truncated(self):
        from backend.service.scenarios.code_review import compose_review_context

        agent = _mock_agent()
        await _invoke(compose_review_context, agent, pr_description="y" * 3000)
        assert "已截断" in _user_message(agent)

    @pytest.mark.asyncio
    async def test_empty_input_still_invokes_the_agent(self):
        """No diff/description → guidance message, but the agent still runs so
        the checkpoint is populated."""
        from backend.service.scenarios.code_review import compose_review_context

        agent = _mock_agent("请提供 PR diff 或描述")
        result = await _invoke(compose_review_context, agent)
        assert "PR diff" in result or "PR 描述" in result
        agent.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_prompt_is_code_review_specific(self):
        from langchain_core.messages import SystemMessage

        from backend.service.scenarios.code_review import compose_review_context

        agent = _mock_agent()
        await _invoke(compose_review_context, agent, pr_diff="diff content")
        sys_msgs = [
            m for m in agent.ainvoke.call_args[0][0]["messages"]
            if isinstance(m, SystemMessage)
        ]
        assert len(sys_msgs) == 1
        assert "代码审查" in str(sys_msgs[0].content)


# ── Tech debt ────────────────────────────────────────────────────────


class TestComposeTechDebt:
    @pytest.mark.asyncio
    async def test_returns_report_and_sends_exactly_two_messages(self):
        """The compose takes no parameters — a fixed scan instruction pair."""
        from langchain_core.messages import HumanMessage, SystemMessage

        from backend.service.scenarios.tech_debt import compose_tech_debt_report

        agent = _mock_agent("技术债报告内容")
        result = await _invoke(compose_tech_debt_report, agent)
        assert result == "技术债报告内容"

        messages = agent.ainvoke.call_args[0][0]["messages"]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    @pytest.mark.asyncio
    async def test_user_message_carries_scan_instructions(self):
        from backend.service.scenarios.tech_debt import compose_tech_debt_report

        agent = _mock_agent()
        await _invoke(compose_tech_debt_report, agent)
        user_text = _user_message(agent)
        assert "workaround" in user_text.lower()
        assert "3" in user_text  # 3-month threshold
