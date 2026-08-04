# 03 — 每周深度巡检

**What to build:** 在调度器中新增每周巡检（默认周一早 9:00），使用 WEEKLY_PATROL_PROMPT 执行全量扫描。每周巡检关注三类发现：(1) 矛盾检测——搜索语义相似但结论相反的 memory pairs；(2) 衰减健康——统计 decay_factor 低于阈值的记忆，建议归档；(3) 实体覆盖度——检查核心实体是否有关联记忆覆盖了关键知识域。巡检结果与每日巡检写入同一 `patrol_logs` 表（patrol_type=weekly），前端和通知系统统一消费。

**Blocked by:** 01（需要调度器和 run_patrol() 模式）

**Status:** ready-for-agent

- [ ] `WEEKLY_PATROL_PROMPT` 模板定义：包含矛盾扫描 instructions（搜索语义相似度 >0.8 但结论相反的 memory pairs，标记为 contradiction）、衰减健康报告 instructions（统计 decay_factor < 0.01 的记忆数量和 ID 列表，标记建议归档的候选）、实体覆盖度检查 instructions（对 memory 数量 top-20 的实体检查是否有关键知识域的记忆覆盖）
- [ ] `backend/shared/config.py` 新增 `PATROL_WEEKLY_DAY`（默认 1 即周一，int 0-6）、`PATROL_WEEKLY_HOUR`（默认 9，int）
- [ ] `backend/shared/config.py` 新增 `PATROL_WEEKLY_ENABLED`（默认 true，bool）独立开关，可单独关闭每周巡检
- [ ] `PatrolScheduler` 新增 `schedule_weekly(day, hour)` 方法，在 `backend/main.py` lifespan 中注册
- [ ] 每周巡检调用 `run_patrol(patrol_type="weekly", trigger="cron", system_prompt=WEEKLY_PATROL_PROMPT)`
- [ ] `.env.example` 新增每周巡检相关环境变量
- [ ] 单元测试：`tests/unit/test_scheduler.py` — `test_scheduler_calculates_next_weekly_run`（验证周期间隔正确，例如周一 9:00 到下周周一 9:00）、`test_weekly_patrol_skips_when_disabled`（验证 PATROL_WEEKLY_ENABLED=false 时不注册 weekly task）
- [ ] 单元测试：`tests/unit/test_patrol_prompts.py` — `test_weekly_prompt_includes_contradiction_instructions`、`test_weekly_prompt_includes_decay_health_instructions`、`test_weekly_prompt_includes_entity_coverage_instructions`
