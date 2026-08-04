# 04 — Stats API 增加 entity_graph 指标

**What to build:** `GET /api/memory/stats` 响应中新增 `entity_graph` 字段，提供三个轻量级指标，为 Phase 3 主动巡检提供数据基础。

**Blocked by:** 01 — 需要 entities/memory_entities 表存在。

**Status:** ready-for-agent

- [ ] `entity_coverage_ratio` — 已链接归一化实体的记忆数 / 总记忆数（一条 SQL JOIN 查询）
- [ ] `entity_growth_rate` — 过去 7 天新增实体数 / 总实体数
- [ ] `entity_density` — 每记忆平均关联实体数 = `COUNT(memory_entities) / COUNT(memories)`
- [ ] `total_entities` — 实体总数
- [ ] 四个指标统一放在 `entity_graph` 嵌套字段中返回，匹配 spec 定义的 JSON 结构
- [ ] `MemoryStatsResponse` Pydantic 模型更新，新增 `entity_graph` 字段
- [ ] 前端 `StatsDashboard` 展示新增指标（可选 KPI 卡片或整合到现有布局）
- [ ] 测试覆盖：统计数据正确性
