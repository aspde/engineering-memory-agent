"""Postmortem scenario — compose postmortem drafts from incident memories."""

from __future__ import annotations

from backend.service.prompts import get_prompt
from backend.service.scenarios import invoke_scenario_agent

# Prompt text lives in the central registry; re-exported for compatibility.
POSTMORTEM_SYSTEM_PROMPT = get_prompt("scenario.postmortem")[1]


async def compose_postmortem(incident_memory_id: str = "") -> str:
    """Compose a postmortem draft for the given incident memory.

    Args:
        incident_memory_id: UUID of the incident memory.  If empty, the
            agent searches for the most recent incident-like memory.

    Returns:
        Formatted postmortem draft in Markdown.
    """
    if incident_memory_id:
        user_message = (
            f"请根据以下信息生成故障复盘报告草稿：\n\n"
            f"故障记录 ID: {incident_memory_id}\n\n"
            f"请先搜索相关记忆和实体，然后按模板输出复盘报告。"
        )
    else:
        user_message = "请搜索最近的故障相关记忆，生成一份故障复盘报告草稿。"

    return await invoke_scenario_agent(POSTMORTEM_SYSTEM_PROMPT, user_message)
