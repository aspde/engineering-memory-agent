"""Code Review scenario — analyse PR diffs against project history."""

from __future__ import annotations

from backend.service.scenarios import invoke_scenario_agent

CODE_REVIEW_SYSTEM_PROMPT = """\
You are EMA's code review mode — 代码审查模式. Your task is to analyse a
pull request's diff and description, cross-reference against project history,
and produce review context that helps the human reviewer.

Steps:
1. Parse the PR diff to identify which files are changed.  For each changed
   file, use search_memories_tool to find past incidents, bug fixes, and
   decisions associated with that file or its related entities.
2. Use query_entity_tool to look up the entities (technologies, modules,
   components) mentioned in the changed files.
3. Read the PR description.  Search for ADRs (Architecture Decision Records)
   and decision memories that may conflict with or support the PR's stated goal.
4. For each changed file, assess risk:
   - 🔴 High: file involved in 2+ production incidents in the last 6 months
   - 🟡 Medium: file involved in 1 incident or several bug fixes
   - 🟢 Low: no incident or significant bug history

Then compose the review context in this structure:

## PR 概述
One paragraph summarising what this PR does and which areas it touches.

## 高风险文件
| 文件 | 风险等级 | 历史故障 | 相关实体 |
|------|---------|---------|---------|
| path/to/File.java | 🔴 | 2 次生产故障 | entity-A, entity-B |

## 决策一致性检查
- Does this PR contradict any existing ADR or documented decision?
- If yes, flag the specific ADR/decision with its memory ID.
- If no conflicts found, state that explicitly.

## 审查建议
3–5 specific things the reviewer should pay attention to, grounded in
project history (not generic advice).

Format: Markdown.  Be specific — use memory IDs, entity names, file paths.
Do not invent history — if no historical data is found, say so.

Always respond in Chinese (简体中文)."""


async def compose_review_context(pr_diff: str = "", pr_description: str = "") -> str:
    """Compose a review context for the given PR.

    Args:
        pr_diff: The diff content of the PR (unified diff format).
        pr_description: The PR description / body text.

    Returns:
        Formatted review context in Markdown.
    """
    parts: list[str] = ["请审查以下 Pull Request：\n"]

    if pr_description:
        desc = pr_description[:2000]
        if len(pr_description) > 2000:
            desc += "\n... (描述已截断)"
        parts.append(f"## PR 描述\n{desc}\n")

    if pr_diff:
        diff = pr_diff[:8000]
        if len(pr_diff) > 8000:
            diff += "\n... (diff 已截断)"
        parts.append(f"## 变更文件 (diff)\n```diff\n{diff}\n```")

    if not pr_diff and not pr_description:
        parts.append(
            "\n用户未提供 PR diff 或描述。请在回复中友善地提示用户提供这些信息，"
            "并说明你可以如何帮助他们审查代码。"
        )
    else:
        parts.append("\n请先搜索变更文件和实体的历史记忆，然后按模板输出审查上下文。")

    user_message = "\n".join(parts)
    return await invoke_scenario_agent(CODE_REVIEW_SYSTEM_PROMPT, user_message)
