# 05 — 事件驱动响应

**What to build:** 让 EMA 能响应外部事件（CI 构建失败、Jira issue resolved）并执行即时巡检。在现有 webhook 基础设施上新增事件驱动的 patrol 触发：CI webhook 携带 build status=failure 时，触发 CI_FAILURE_PROMPT 巡检——搜索相似历史故障并判断是否需要告警；Jira webhook 携带 issue status=resolved 时，触发 JIRA_RESOLVED_PROMPT 巡检——搜索同根因的历史问题判断是否重复。实现简单速率限制：同一事件源（如相同 CI job name）在时间窗口内（默认 1h）只触发一次巡检，使用内存字典 + asyncio.Lock（不引入 Redis）。

**Blocked by:** 01（需要 run_patrol() 函数和 patrol_logs 表）

**Status:** ready-for-agent

- [ ] `CI_FAILURE_PATROL_PROMPT` 模板：Instructions 包括搜索与当前 CI job name / error message 相似的历史故障记忆、判断是否匹配已知问题模式、如匹配则调用 notify_slack 推送（含历史解决方案链接）
- [ ] `JIRA_RESOLVED_PATROL_PROMPT` 模板：Instructions 包括搜索与当前 issue title/description 相似的历史 issue 记忆、判断同根因重复修复（"this looks like issue #X which was also caused by Y"）
- [ ] `backend/service/patrol.py`：`run_event_patrol(trigger_source, event_payload, system_prompt)` 函数，封装 `patrol_type="event_driven"` + `trigger="webhook"` 的巡检调用
- [ ] `backend/service/rate_limiter.py`：`RateLimiter` 类，`is_allowed(key: str, window_seconds: int = 3600) -> bool`，基于内存 dict（`dict[str, float]` 存上次触发时间戳）+ `asyncio.Lock`，定期清理过期条目（可选，避免内存泄漏）
- [ ] 现有 webhook 路由集成：CI PingCode/connector 的 failure 事件处理中，检查 rate limiter（key=job_name）→ 调用 `run_event_patrol`；Jira connector 的 issue status change 事件处理中，status=resolved 时检查 rate limiter（key=issue_key）→ 调用 `run_event_patrol`
- [ ] 速率限制被触发时返回 200 正常响应，可选 response header `X-Rate-Limited: true`，不报错不抛异常
- [ ] 事件巡检结果写入 `patrol_logs`（patrol_type=event_driven, trigger=webhook）
- [ ] 单元测试：`tests/unit/test_rate_limiter.py` — `test_allows_first_call`、`test_blocks_within_window`、`test_allows_after_window_expires`、`test_different_keys_independent`（不同 job_name 独立计数）
- [ ] 单元测试：`tests/unit/test_patrol_prompts.py` — `test_ci_failure_prompt_includes_search_instructions`、`test_jira_resolved_prompt_includes_root_cause_check`
- [ ] 集成测试：`tests/api/test_webhook_routes.py` — 扩展已有测试，验证 CI failure webhook 触发 patrol（Mock agent.ainvoke，验证 patrol_log 写入）和速率限制行为（连续两次相同 webhook，第二次不触发 patrol）
