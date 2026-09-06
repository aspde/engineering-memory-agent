"""Tech Debt Radar scenario — scan for unresolved workarounds and doc gaps."""

from __future__ import annotations

import logging

from backend.runner.scenarios import execute_scenario, invoke_scenario_agent
from backend.service.prompts import get_prompt

# Prompt text lives in the central registry; re-exported for compatibility.
TECH_DEBT_SYSTEM_PROMPT = get_prompt("scenario.tech_debt")[1]

logger = logging.getLogger(__name__)


async def compose_tech_debt_report() -> str:
    """Compose a tech debt report.

    Designed to be triggered manually or by the weekly patrol scheduler.

    Returns:
        Formatted tech debt report in Markdown.
    """
    user_message = (
        "请扫描项目知识库，生成技术债报告。\n"
        "1. 搜索标记为 workaround/temporary/临时方案/待优化/TODO 的记忆\n"
        "2. 检查哪些超过 3 个月未有跟进\n"
        "3. 识别文档缺口（记忆多但无文档的模块）\n"
        "4. 检测近期 commit 是否已解决了某些临时方案\n"
        "按模板格式输出完整报告。"
    )

    return await invoke_scenario_agent(TECH_DEBT_SYSTEM_PROMPT, user_message)


async def run_tech_debt_scan() -> None:
    """Run the weekly tech-debt scan through the shared executor.

    The scheduler entry point (backend.main wires its weekly slot here).
    Same execution path as the manual route and the event trigger:
    concurrency slot, SCENARIO_TIMEOUT_SECONDS deadline, ``scenario_runs``
    persistence, JSON contract parsing, typed error recording.  Calling
    ``compose_tech_debt_report()`` directly would skip all of those — the
    cron path is the only unattended tech-debt surface, so an unrecorded
    weekly report is invisible.  Failures are logged, never raised: the
    scheduler treats a raising callback as a crash, and a failed scan is
    already terminal-state on its run row.
    """
    try:
        outcome = await execute_scenario("tech_debt", trigger="weekly_patrol")
        logger.info(
            "Tech debt scan %s (run %s, contract_ok=%s)",
            outcome["status"],
            outcome["run_id"],
            outcome["contract_ok"],
        )
    except Exception:
        logger.exception("Tech debt scan failed")
