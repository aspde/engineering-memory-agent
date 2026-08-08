"""Onboarding scenario — generate structured project overviews for new members."""

from __future__ import annotations

from backend.service.prompts import get_prompt
from backend.service.scenarios import invoke_scenario_agent

# Prompt text lives in the central registry; re-exported for compatibility.
ONBOARDING_SYSTEM_PROMPT = get_prompt("scenario.onboarding")[1]


async def compose_onboarding_guide(scope: str = "full") -> str:
    """Compose an onboarding guide for new team members.

    Args:
        scope: ``"full"`` for the complete project overview, or an entity
            name to focus the guide on a specific area.

    Returns:
        Formatted onboarding guide in Markdown.
    """
    if scope and scope != "full":
        user_message = (
            f"请为项目中的 '{scope}' 模块/实体生成一份新人 Onboarding 指南。\n"
            f"先搜索与该实体相关的所有记忆、关联实体和决策记录，再按模板输出。"
        )
    else:
        user_message = (
            "请为整个项目生成一份新人 Onboarding 指南（项目全览）。\n"
            "先全面搜索项目的核心模块、架构决策、历史故障和文档记忆，再按模板输出。"
        )

    return await invoke_scenario_agent(ONBOARDING_SYSTEM_PROMPT, user_message)
