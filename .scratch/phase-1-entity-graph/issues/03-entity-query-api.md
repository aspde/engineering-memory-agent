# 03 — 实体查询与搜索 API

**What to build:** 三个新 REST 端点，让用户和前端能查询实体图谱：按 ID 获取实体档案、获取一度关系、按名称搜索实体。

**Blocked by:** 01 — 需要 entities/memory_entities 表存在。

**Status:** ready-for-agent

- [ ] `GET /api/entities/{entity_id}` — 返回实体档案：`id`、`name`、`canonical_name`、`type`、`memory_count`、`source_breakdown`（按 source_type 分组计数）、`first_seen_at`
- [ ] `GET /api/entities/{entity_id}/relations` — 返回一度关系：实体信息 + `related_entities`（关联实体 ID、名称、类型、关系类型、关联记忆数）+ `recent_memories`（最近 5 条关联记忆的摘要、来源、时间）
- [ ] `GET /api/entities/search?q=postgres&type=technology` — 按名称模糊搜索实体（`canonical_name ILIKE %q%` 或 `name ILIKE %q%`），支持 `type` 过滤，返回匹配实体列表
- [ ] 新路由模块 `backend/api/routes/entity_routes.py`，前缀 `/entities`，注册到 `backend/api/router.py`
- [ ] 所有端点处理 404（实体不存在）和参数校验
- [ ] API 测试覆盖：正常响应结构、404、空结果、type 过滤
