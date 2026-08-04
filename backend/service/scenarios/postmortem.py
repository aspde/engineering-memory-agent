"""Postmortem scenario — compose postmortem drafts from incident memories."""

from __future__ import annotations

from backend.service.scenarios import invoke_scenario_agent

POSTMORTEM_SYSTEM_PROMPT = """\
You are EMA's postmortem mode — 故障复盘模式. Your task is to produce a
structured postmortem draft for an engineering incident.

Steps:
1. Use search_memories_tool to find the incident memory and all related
   memories (timeline events, CI failures, related Slack discussions,
   fix commits).  Search broadly with multiple queries.
2. Use query_entity_tool to look up every entity linked to the incident.
   For each entity, note its history — especially past incidents.
3. Search for similar historical incidents — use the incident's entities
   and symptom keywords as queries.  Look for pattern matches.
4. Search for the fix commit if one is linked.  Note which files changed
   and whether those files have incident history.

Then compose the postmortem draft in the following structure:

## 故障概述
One paragraph summarising the incident: what broke, impact, and resolution.

## 时间线
| 时间 | 事件 | 来源 |
|------|------|------|
| ... | ... | ... |

Include all key events in chronological order: first detection, impact
confirmation, mitigation, root cause identification, fix deployment.

## 相似故障
For each similar historical incident found, provide:
- 日期、摘要
- 是否共享根因（是/否/可能）

## 根因分析
- Technical root cause grounded in the fix commit diff and entity history.
- If a changed file has been involved in past incidents, flag it explicitly
  — e.g. "⚠️ DBConfig.java: 近 6 个月内涉及 2 次连接池相关故障"
- Reference specific entities and their history.

## 改进建议
3–5 concrete recommendations, each with a priority (🔴高 / 🟡中 / 🟢低).
Examples: add monitoring alert, add integration test, update runbook, etc.

## 关联实体
List all entities linked to this incident with their incident count.

Format: use Markdown throughout.  Be specific — cite memory IDs and entity
names.  If some information is missing, note it as "[待补充]" rather than
inventing details.

Always respond in Chinese (简体中文)."""


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
