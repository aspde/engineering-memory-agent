# 05 — Agent 实体感知能力

**What to build:** Agent 的两个工具增强——搜索结果附带归一化实体信息，新增 `query_entity_tool` 让 Agent 可直接查询实体图谱。

**Blocked by:** 02（归一化实体数据已存在）, 03（查询 API 可用）

**Status:** ready-for-agent

- [ ] `search_memories_tool` 返回的 JSON 中每条记忆增加 `entities` 字段：该记忆关联的归一化实体列表（entity_id + canonical_name + type）
- [ ] 新增 `query_entity_tool`：接受 `entity_name: str` 参数，调用 `GET /api/entities/search` 查找实体，再调用 `GET /api/entities/{id}/relations` 获取关系，返回格式化的实体档案 + 关联实体 + 最近记忆摘要
- [ ] `query_entity_tool` 注册到 `ALL_TOOLS` 列表
- [ ] `search_memories_tool` 的实体信息通过 `memory_entities` JOIN 查询获取（在 `query_memories()` 或工具层实现）
- [ ] Agent 工具测试：`query_entity_tool` 返回 JSON 格式正确（已存在实体 / 不存在实体）、`search_memories_tool` 返回含 entities 字段
