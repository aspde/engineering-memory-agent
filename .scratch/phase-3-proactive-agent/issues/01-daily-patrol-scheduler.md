# 01 — 每日巡检基础设施

**What to build:** 搭建后台巡检调度器、`patrol_logs` 表、以及每日巡检的完整执行链路。调度器在 FastAPI lifespan 中启动一个 asyncio 后台循环，按环境变量配置的时间（默认早 8:00）触发每日巡检。巡检调用 `agent.ainvoke()` 使用预定义的 DAILY_PATROL_PROMPT，Agent 通过已有的 `search_memories_tool` 和 `query_entity_tool` 扫描过去 24h 的新记忆，输出结构化 JSON（pattern_matches、knowledge_gaps、new_entities），结果写入 `patrol_logs` 表。巡检失败时状态标记为 `failed` 并记录错误信息，不影响后续调度周期。调度器支持优雅关闭（lifespan shutdown 时取消 asyncio Task）。

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `patrol_logs` 表创建完成：`id UUID PK DEFAULT gen_random_uuid()`、`patrol_type TEXT NOT NULL`（daily/weekly/event_driven/manual）、`trigger TEXT NOT NULL`（cron/webhook/manual）、`status TEXT NOT NULL DEFAULT 'running'`、`findings JSONB`、`dismissed_findings UUID[]`、`started_at TIMESTAMPTZ DEFAULT now()`、`completed_at TIMESTAMPTZ`
- [ ] 表通过 `backend/db/schema.py` 的 `init_db()` 创建
- [ ] `backend/service/scheduler.py`：`PatrolScheduler` 类，支持 `schedule_daily(hour: int)`、`schedule_weekly(day: int, hour: int)`，内部用 `asyncio.sleep` 循环计算下次运行时间，`start()` 启动所有已注册的调度任务，`stop()` 取消所有任务
- [ ] `backend/service/patrol.py`：`run_patrol(patrol_type, trigger, system_prompt, scope=None)` 函数，调用 `agent.ainvoke()` 并写入 patrol_log（含 started_at 和 completed_at 记录）
- [ ] `DAILY_PATROL_PROMPT` 模板定义为固定文本常量，包含：搜索过去 24h 新记忆、pattern_match 相似度阈值 0.85、knowledge_gap 阈值（实体 >10 条记忆但 category "documentation" 为 0）、new_entity 检测（本周出现在消息中但无文档覆盖）
- [ ] `backend/shared/config.py` 新增 `PATROL_DAILY_HOUR`（默认 8，int）、`PATROL_ENABLED`（默认 true，bool）配置项，从环境变量读取
- [ ] `.env.example` 新增 patrol 相关环境变量及注释说明
- [ ] `backend/main.py` lifespan startup 中创建 PatrolScheduler 实例、注册每日巡检、调用 `scheduler.start()`；shutdown 中调用 `scheduler.stop()`
- [ ] 调度器在 `PATROL_ENABLED=false` 时跳过所有巡检（不启动 asyncio Task）
- [ ] 单元测试：`tests/unit/test_scheduler.py` — `test_scheduler_calculates_next_daily_run`（验证时间计算正确）、`test_scheduler_skips_when_disabled`（验证 PATROL_ENABLED=false 时不创建 task）
- [ ] 单元测试：`tests/unit/test_patrol_prompts.py` — `test_daily_prompt_includes_all_required_sections`（验证 prompt 文本包含 pattern_matches、knowledge_gaps、new_entities 三个 required sections 和 output format 要求）
- [ ] 集成测试：`tests/unit/test_patrol.py` — `test_run_patrol_creates_log_entry`（Mock agent.ainvoke 返回固定 JSON，验证 patrol_log 写入的各字段正确）、`test_run_patrol_marks_failed_on_error`（Mock agent.ainvoke 抛出异常，验证 status='failed' 且 completed_at 被记录）
