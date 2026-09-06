"""Postmortem scenario — compose postmortem drafts from incident memories."""

from __future__ import annotations

from backend.runner.scenarios import invoke_scenario_agent
from backend.service.prompts import get_prompt

# Prompt text lives in the central registry; re-exported for compatibility.
POSTMORTEM_SYSTEM_PROMPT = get_prompt("scenario.postmortem")[1]


async def compose_postmortem(
    incident_memory_id: str = "",
    trigger_event: dict | None = None,
) -> str:
    """Compose a postmortem draft for the given incident memory.

    Args:
        incident_memory_id: UUID of the incident memory.  If empty, the
            agent searches for the most recent incident-like memory.
        trigger_event: Optional context about the event that triggered this
            run (event path only): ``{"source", "content", "memory_id"}``.
            Injected into the user message so the agent anchors its search
            on the concrete incident instead of guessing.

    Returns:
        Formatted postmortem draft in Markdown (with the trailing JSON
        contract block per prompt v4).
    """
    parts: list[str] = ["请生成故障复盘报告草稿。\n"]

    if trigger_event:
        source = trigger_event.get("source", "")
        content = str(trigger_event.get("content", ""))[:2000]
        memory_id = trigger_event.get("memory_id", "")
        parts.append("## 本次触发事件（自动触发，请以此为中心检索相关记忆）")
        if source:
            parts.append(f"事件来源: {source}")
        if memory_id:
            parts.append(f"事件记忆 ID: {memory_id}")
        if content:
            parts.append(f"事件内容:\n{content}")
        parts.append("")
    elif incident_memory_id:
        parts.append(f"## 故障记录\nID: {incident_memory_id}\n")

    if not trigger_event and not incident_memory_id:
        parts.append(
            "未提供具体故障记录。请先搜索最近的故障相关记忆，选定一条作为复盘对象。"
        )
    else:
        parts.append(
            "请先搜索相关记忆和实体，然后按模板输出复盘报告，"
            "并在报告末尾附上 JSON 契约块。"
        )

    user_message = "\n".join(parts)
    return await invoke_scenario_agent(POSTMORTEM_SYSTEM_PROMPT, user_message)
