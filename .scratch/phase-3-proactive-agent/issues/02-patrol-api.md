# 02 — 巡检 API + 手动触发

**What to build:** 提供巡检日志查询、手动触发巡检、Finding 忽略的 REST API。4 个端点：`POST /api/patrol/trigger` 接受 `patrol_type`（daily/weekly/contradiction_scan）和可选 `scope`（all / entity:{name}），异步启动巡检并返回 202 Accepted；`GET /api/patrol/logs` 分页返回最近 50 条巡检记录；`GET /api/patrol/logs/{id}` 返回单次巡检的完整 findings；`POST /api/patrol/findings/{id}/dismiss` 将 finding ID 加入 `dismissed_findings` 数组。所有端点遵循现有 API 的错误处理模式（404 for missing log, 422 for invalid input）。

**Blocked by:** 01（需要 patrol_logs 表和 run_patrol() 函数）

**Status:** ready-for-agent

- [ ] `backend/api/routes/patrol_routes.py`：创建路由文件，4 个端点实现
- [ ] `POST /api/patrol/trigger`：请求 body `{"patrol_type": "daily" | "weekly" | "contradiction_scan", "scope?": "all" | "entity:<name>"}`，Pydantic 模型校验，异步执行（asyncio.create_task）不阻塞响应，返回 202 + `{"patrol_id": "<uuid>", "status": "accepted"}`，无效 patrol_type 返回 422
- [ ] `GET /api/patrol/logs`：支持 `?limit=`（默认 50，max 100）和 `?offset=`（默认 0）分页，按 `started_at DESC` 排序，返回列表不含 findings 全文（仅 id、patrol_type、trigger、status、started_at、completed_at、finding 数量），支持 `?patrol_type=` 过滤
- [ ] `GET /api/patrol/logs/{id}`：返回完整 patrol_log 记录含 findings JSONB，不存在时返回 404
- [ ] `POST /api/patrol/findings/{id}/dismiss`：body `{"finding_id": "<uuid>"}`，将 finding_id 追加到 `dismissed_findings[]` 数组（使用 PostgreSQL array append），幂等（同一 finding_id 重复 dismiss 不报错），log 不存在返回 404
- [ ] 路由注册到 `backend/api/router.py`（prefix `/patrol`）
- [ ] Pydantic 请求/响应模型定义在 `backend/api/routes/patrol_routes.py` 或 `backend/model/patrol.py`
- [ ] 单元测试：`tests/api/test_patrol_routes.py` — `test_manual_patrol_trigger_returns_accepted`、`test_trigger_invalid_type_returns_422`、`test_list_logs_returns_paginated`、`test_get_log_not_found_returns_404`、`test_dismiss_finding_adds_to_dismissed_list`、`test_dismiss_duplicate_finding_is_idempotent`、`test_dismiss_nonexistent_log_returns_404`
