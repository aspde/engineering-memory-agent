"""Code Review scenario — analyse PR diffs against project history."""

from __future__ import annotations

from backend.service.prompts import get_prompt
from backend.service.scenarios import invoke_scenario_agent

# Prompt text lives in the central registry; re-exported for compatibility.
CODE_REVIEW_SYSTEM_PROMPT = get_prompt("scenario.code_review")[1]


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
