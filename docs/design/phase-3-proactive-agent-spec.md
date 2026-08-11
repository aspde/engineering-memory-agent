# Phase 3: 主动 Agent — 功能规格

> **实现状态（2026-08）**：本 spec 是 Phase 3 的原始设计蓝图，正文保留设计时的意图。
> 与当前代码的差异：
> - 定时巡检 `daily` / `weekly` 已实现——weekly 内含全量矛盾扫描与过期记忆扫描（`contradictions` / `stale_memories` / `entity_coverage`），**不再有独立的 `contradiction_scan` 类型**；
> - 事件驱动响应（CI 失败 / Jira 解决）从未接线，已在精简中移除（含对应 prompt）；
> - 巡检输出修复重试（repair）已移除——输出不合 JSON 契约时直接记 `failed` 并保留原始输出；
> - Slack 通知未实现，生产用飞书 webhook 推送；`notify_slack` tool 从未实现；
> - 手动触发 `POST /api/patrol/trigger` 返回真实 `patrol_id`（异步运行）。
> 当前 `patrol_type` 枚举为 `daily | weekly`。

## Problem Statement

EMA 当前是完全被动的——用户发消息才动作，用户不发消息就静静等着。即使知识库已经积累了数百条记忆、连接器在持续接入外部事件、实体关系图谱已经形成，EMA 不会主动告诉团队任何事情。

这导致三个盲区：一是**模式重复**——同样的故障模式可能在一段时间后重演，但直到有人手动搜索才会发现关联；二是**知识盲区**——团队不知道自己不知道什么，比如核心依赖 PostgreSQL 关联了 41 条记忆却没有一条关于备份恢复的；三是**知识腐化**——过期（从未召回或久未召回）的记忆没有人定期检查和清理，矛盾结论没有被发现和仲裁。

EMA 需要从"等你问"变成"主动告诉你"——在后台持续观察、关联、发现，在你还不知道有问题之前，告诉你哪里该看一眼。

## Solution

新增三层主动能力：**定时巡检**（固定频率的全面扫描）、**事件驱动响应**（外部事件触发即时分析）、**主动通知**（将发现推送到 Slack/Web UI）。每层都是同一个 Agent Graph 配上不同的触发方式和 System Prompt——架构不变，只换装配方式。

## User Stories

### 定时巡检

1. As a team, I want EMA to run a daily patrol scanning the last 24 hours of new memories, cross-referencing them with historical knowledge, and producing a morning brief, so that we start the day knowing what changed and what needs attention.

2. As a team, I want the daily brief to include pattern matches — "Jira EMA-58 'API 响应变慢' has symptoms highly similar (0.89) to the March connection pool outage" — so that recurring issues are flagged early.

3. As a team, I want the daily brief to highlight knowledge blind spots — "PostgreSQL is our core dependency with 41 related memories, but zero about backup/recovery" — so that we know where to focus documentation effort.

4. As a team, I want the daily brief to surface newly discovered entities — "Kafka appeared in 12 Slack messages this week but has zero memories" — so that emerging technologies are noticed before they become unmanaged dependencies.

5. As a team, I want a weekly deeper patrol that runs entity-level contradiction scanning — "Memory #47 and #89 about microservice splitting granularity have opposite conclusions and have never been flagged as a conflict" — so that unresolved disagreements surface without anyone manually comparing memories.

6. As a team, I want the weekly patrol to include a stale-memory report — "23 memories have never been recalled or not recalled for 90+ days, suggesting they are no longer relevant. 5 are candidates for archival" — so that the knowledge base stays healthy without manual curation.

### 事件驱动响应

7. As a team, I want EMA to react to CI build failures in real time — when a build fails, EMA should search for similar past failures and push a summary, so that we don't waste time re-diagnosing known issues.

8. As a team, I want EMA to react to a Jira issue being marked "resolved" — EMA should search for related historical issues with the same root cause and flag if this looks like a repeat, so that we notice when we're fixing the same thing repeatedly.

9. As a team, I want EMA to detect when a CI configuration value (e.g., connection pool size) matches a value that has caused a past production incident, and alert immediately, so that misconfigurations are caught before they hit production.

10. As a team, I want event-driven responses to be rate-limited — if the same CI job fails 50 times in an hour, EMA should alert once, not 50 times, so that notification fatigue doesn't make us ignore it.

### 主动通知

11. As a team, I want patrol findings and event alerts pushed to a designated Slack channel, so that we receive insights without checking EMA's Web UI.

12. As a team, I want the notification level to be configurable per finding type — pattern match = critical, knowledge gap = warning, stale memory = info — so that urgent findings don't get buried in low-priority noise.

13. As a team member, I want to be able to dismiss a finding — "I've seen this, don't notify again for this specific item" — so that acknowledged findings don't keep re-alerting.

14. As a team, I want a "Patrol Log" page in the Web UI showing all past patrol runs, their findings, and which were acted upon, so that we can track the value EMA's proactive behavior is delivering.

### 手动触发

15. As a power user, I want to manually trigger a patrol from the Web UI — "run a contradiction scan right now" — without waiting for the next scheduled run.

16. As a power user, I want to specify the scope of a manual patrol — "scan only memories related to PostgreSQL" or "scan only the last 7 days" — so that I can focus on a specific concern.

### 配置

17. As an operator, I want to configure patrol schedules via environment variables — `PATROL_DAILY_HOUR=8`, `PATROL_WEEKLY_DAY=1` — so that schedules fit the team's timezone and rhythm.

18. As an operator, I want to toggle individual patrol types on/off — contradiction scanning may be too noisy for a small team — so that teams can adopt proactive features incrementally.

## Implementation Decisions

### 架构原则

主动 Agent 不对 Agent Graph 做任何改动。每次巡检是一次独立的 `agent.ainvoke()` 调用——同一个 `build_agent_graph()` 产物，只是触发方式不同，System Prompt 不同：

| 模式 | 触发方式 | System Prompt 导向 |
|------|---------|-------------------|
| 对话 | 用户发消息 | "回答用户问题，需要时搜索记忆" |
| 每日巡检 | Cron 定时 | "扫描过去 24h 的新记忆，找出模式匹配、知识盲区、新实体" |
| 每周巡检 | Cron 定时 | "扫描所有记忆的矛盾、过期记忆（从未召回/久未召回）、实体覆盖度" |
| 事件响应 | Webhook 触发 | "CI 构建失败，搜索相似历史故障，判断是否需要告警" |

### 调度器

调度器是一个轻量级的异步任务执行器——不引入 APScheduler 或 Celery：

```python
# 核心是一个 background task + asyncio.sleep 循环
# 应用启动时注册，shutdown 时取消
async def patrol_loop():
    while True:
        now = datetime.now()
        next_run = calculate_next_run(now, schedule_config)
        await asyncio.sleep((next_run - now).total_seconds())
        await run_patrol(patrol_type="daily")
```

不需要持久化任务队列——巡检任务的调度规则简单（每天固定时间、每周固定天），`asyncio.sleep` + 循环即可。重启时最多丢失一次巡检，可以接受。

### 巡检 Prompt 模板

每种巡检类型有自己的 System Prompt 模板——不通过 LLM 生成，而是**预定义的固定文本**，确保巡检行为稳定、可预测：

```
DAILY_PATROL_PROMPT = """
You are EMA's daily patrol mode. Your task is to scan recent memories
and produce a structured briefing.

Steps:
1. Search for memories created in the last 24 hours
2. For each new memory, find similar historical memories (pattern match)
3. Identify knowledge gaps — entities with high memory count but missing domains
4. Identify new entities that appeared this week but have no documentation

Output format:
{ "pattern_matches": [...], "knowledge_gaps": [...], "new_entities": [...] }

Rules:
- Pattern match similarity threshold: 0.85
- Only report matches that are actionable — not every similarity is a pattern
- Knowledge gaps: flag when entity has >10 memories but 0 in category "documentation"
"""
```

Agent 使用已有的 `search_memories_tool` 和 Phase 1 的 `query_entity_tool` 完成巡检。不需要新 tool。巡检结果以 JSON 结构化返回，前端和通知系统消费同一份数据。

### 新增 notify_slack tool

Agent 完成巡检后，如果需要推送，调用新 tool：

```
notify_slack(channel: str, message: str, blocks: list[dict] | None)
```

这个 tool 的唯一职责是将结果格式化并通过 Slack Webhook 发送。前端巡检结果展示走现有 API，不经过 Slack。

### 新增 patrol_logs 表

```sql
CREATE TABLE patrol_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patrol_type TEXT NOT NULL,        -- daily | weekly
    trigger TEXT NOT NULL,            -- cron | webhook | manual
    status TEXT NOT NULL DEFAULT 'running',  -- running | completed | failed
    findings JSONB,                   -- 结构化巡检结果
    dismissed_findings TEXT[],         -- 用户已忽略的 finding 键（<group>-<index>，矛盾用 memory 对）
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);
```

### API 新端点

- `POST /api/patrol/trigger` — 手动触发巡检。body: `{"patrol_type": "daily" | "weekly", "scope": "all" | "entity:{name}"}`
- `GET /api/patrol/logs` — 巡检历史记录列表（分页，最近 50 条）
- `GET /api/patrol/logs/{id}` — 单次巡检详情（findings 全文）
- `POST /api/patrol/findings/{id}/dismiss` — 忽略某个 finding

### Agent Tool 修改

无需修改现有 tool。新增一个 tool：

| Tool | 职责 |
|------|------|
| `notify_slack` | 将巡检结果推送到 Slack。Web UI 展示不走此 tool |

### 前端新增

- `src/pages/Patrol.tsx`（路由 `/patrol`）— 巡检日志列表 + 最近 findings 展示
- 首页/仪表盘增加"今日简报"卡片，展示最近一次 daily patrol 的关键发现
- 手动触发入口：按钮"立即巡检"

### Schema 迁移

- 新增 `patrol_logs` 表
- 无破坏性变更
- 无新 Python 包依赖——`asyncio` 是标准库

## Testing Decisions

### 测试原则

- 巡检逻辑的核心是 prompt 指导下的 Agent.ainvoke() 调用——测试只验证**调度器正确触发**和**巡检 prompt 模板正确性**，不测试 LLM 输出内容
- `notify_slack` tool 只测试格式化逻辑和参数传递——Mock Slack webhook 调用
- 调度器逻辑测试：验证时间计算正确、任务取消正确、失败重试策略

### 测试模块

| 测试文件 | 内容 | 参考 |
|---------|------|------|
| `tests/unit/test_scheduler.py` | 调度器时间计算、任务编排、取消逻辑 | 新——纯逻辑测试 |
| `tests/unit/test_patrol_prompts.py` | prompt 模板包含必要的 instructions 和 output format | 新——确定性文本测试 |
| `tests/unit/test_notify_slack.py` | 消息格式化、channel 参数传递 | test_agent_tools.py |
| `tests/api/test_patrol_routes.py` | 手动触发、日志查询、finding dismiss | test_memory_routes.py |
| `tests/unit/test_agent_tools.py` | notify_slack tool 返回格式 | 已有测试扩展 |
| `src/pages/Patrol.test.tsx` | 巡检结果渲染、dismiss 操作 | StatsDashboard.test.tsx |

### 测试用例速览

- `test_scheduler_calculates_next_daily_run` — 时间计算
- `test_scheduler_skips_when_disabled` — 配置关闭时跳过
- `test_run_patrol_creates_log_entry` — 巡检记录写入
- `test_daily_prompt_includes_all_required_sections` — prompt 完整性
- `test_notify_slack_formats_message_correctly` — 消息格式化
- `test_manual_patrol_trigger_returns_accepted` — 手动触发 API
- `test_dismiss_finding_removes_from_active_list` — Finding 忽略
- `test_agent_notify_slack_tool_handles_failure` — Slack 调用失败降级

## Out of Scope

- 任务持久化 / 消息队列（Celery、Redis、Bull）——只用 asyncio.sleep + 循环
- 分布式调度（单实例运行足够）——多实例时用 leader election 或只用 cron trigger
- 巡检结果自动修复——只发现问题，不自动修改记忆（HITL 需求）
- 机器学习模型预测——不做"预测下周会出什么故障"，只做基于已有知识的模式匹配
- 通知渠道扩展（邮件、短信、PagerDuty）——首批只做 Slack + Web UI
- 巡检流式展示——巡检是后台任务，前端查询结果

## Further Notes

- 本 spec 对应 [ADR-006](../decisions/ADR-006-extension-roadmap.md) 的 Phase 3
- 调度器实现决策（内嵌主进程、不引入 APScheduler / Celery）见 [ADR-007](../decisions/ADR-007-patrol-in-process-scheduler.md)
- 主动 Agent 的价值严重依赖 Phase 1（实体归一化）和 Phase 2（多源数据持续输入）。没有前两个 Phase，主动 Agent 的眼界太窄，发现不了有价值的东西
- 巡检频率初始设置为：每日（早 8:00 扫前 24h）+ 每周（周一早 9:00 全量扫描）。这些是默认值，通过环境变量覆盖
- 巡检结果中的"模式匹配"阈值 (0.85) 是起点——上线后根据误报率调整
