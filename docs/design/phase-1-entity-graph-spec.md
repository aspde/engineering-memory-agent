# Phase 1: 知识图谱化 — 功能规格

## Problem Statement

EMA 目前已从原始内容中提取实体和关系（三阶段提取管线），但实体名称以自由文本形式锁在每个 memory 的 JSONB 字段中。用户无法跨记忆查询"PostgreSQL 在我们系统中关联了哪些决策和故障"，因为 "PostgreSQL"、"Postgres"、"pg16" 三个字符串无法被识别为同一个东西。同理，关系只能在一段记忆内部有意义，不能跨记忆追踪因果链。

EMA 需要一种能力——让分散在数百条记忆中的实体和关系**联成一张网**，使得用户不仅能用向量搜索"找到关于 X 的记忆"，还能用图查询"理解 X 在团队知识体系中的位置、关联和演进"。

## Solution

新增**实体归一化层**——在记忆写入时，自动将提取到的实体名与已有实体匹配并链接到同一个规范化实体。允许用户在 Web UI 中以实体为中心浏览所有关联记忆，以交互式图谱可视化实体的直接关系。

## User Stories

### 实体归一化

1. As a user, I want EMA to recognize that "PostgreSQL" and "Postgres" and "pg" are the same entity across different memories, so that I don't get fragmented results when searching.

2. As a user, I want EMA to link extracted entities to an existing canonical entity when they are determined to be equivalent (similar name embedding + LLM confirmation), so that all knowledge about the same thing is connected without me manually curating.

3. As a developer, I want the entity normalization to happen automatically as part of the memory write pipeline, so that I don't need an extra step to maintain the entity graph.

4. As a developer, I want entity normalization to be idempotent — if it fails (LLM down, embedding unavailable), the memory still gets written without orphaned entity links, so that memory ingestion doesn't depend on entity normalization reliability.

### 实体查询

5. As a user, I want to retrieve an entity by name or ID and get all its related memories, grouped by relation type (decision, fault, dependency), so that I can see the full picture of how this entity fits in our technical landscape.

6. As a user, I want to see all entities directly related to a given entity, with the relationship type and count, so that I can navigate the knowledge graph step by step.

7. As a user, I want to filter a memory search by entity name — "find memories related to PostgreSQL in the past 30 days" — so that I can narrow down broad searches.

8. As a user, I want to see the total count of memories an entity is associated with, and how many came from each source type (git, conversation, jira), so that I know how rich the knowledge is around this entity.

### 关系查询

9. As a user, I want to query one-degree relations for an entity — "what directly depends on PostgreSQL?" / "what does PostgreSQL depend on?" — so that I can understand dependency chains.

10. As a user, I want to find `contradicts` relations — "do we have conflicting conclusions about microservice splitting?" — across multiple memories, so that I can spot unresolved disagreements.

11. As a user, I want to find `causes` chains — "what faults were caused by connection pool misconfiguration?" — so that I can trace root causes to downstream consequences.

### Agent 行为

12. As a user, I want the Agent to use entity-graph context when answering questions — "what do we know about PostgreSQL's performance?" should pull in relations and related entities, not just semantically similar text — so that the answer includes associations I didn't explicitly search for.

13. As a user, I want the Agent's memory search tool to return normalized entity information alongside memory summaries, so that I can see which entities each result relates to without drilling into the memory detail.

### 前端可视化

14. As a user, I want a new "Graph" page in the Web UI where I can search for an entity and see it as the center of a node-edge graph, so that I can visually explore the knowledge network rather than reading a flat list.

15. As a user, I want to click on a node in the graph to expand its direct relations, so that I can navigate the knowledge network interactively.

16. As a user, I want to click on a memory connected to an entity in the graph to see its full content, so that I can drill down from entity-level overview to memory-level detail.

17. As a user, I want to search for an entity in the chat page and have a "View in Graph" link, so that I can seamlessly switch between conversational search and visual exploration.

18. As a user, I want to see different colors/sizes for different entity types (technology vs. decision vs. fault), so that I can visually distinguish classes of knowledge at a glance.

## Implementation Decisions

### 数据模型

**新增 `entities` 表**：

```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    type TEXT NOT NULL,  -- technology | person | project | decision | event | file | concept
    embedding vector(1024),
    first_seen_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(canonical_name, type)
);
```

- `name` 是首次出现时的原始名称
- `canonical_name` 是 LLM 确认的标准名称
- `embedding` 用于查找语义相似的已有实体
- 实体通过 `(canonical_name, type)` 唯一约束防止重复

**新增 `memory_entities` 关联表**：

```sql
CREATE TABLE memory_entities (
    memory_id UUID REFERENCES memories(id) ON DELETE CASCADE,
    entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);
```

- 多对多关系：一条记忆可关联多个实体，一个实体可出现在多条记忆中
- `ON DELETE CASCADE` 保证删除记忆或实体时关联自动清理

**`memories` 表保持不变**：现有的 `entities` 和 `relations` JSONB 字段作为**来源记录**继续保留。`memory_entities` 是额外的索引层，不是替代。

### 实体归一化流程

```
write_memory() 写入成功后
    │
    ▼
normalize_entities(memory_id, extracted_entities)
    │
    │  对每个实体:
    │    1. embed(entity["name"]) → 向量
    │    2. 在 entities 表中搜索 top-3 相似实体 (cosine > 0.85)
    │    3. 如果找到候选 → LLM 判断是否为同一实体
    │    4. 如果是 → 链接到已有 entity
    │    5. 如果否或无候选 → INSERT 新 entity
    │
    ▼
  所有实体 → INSERT INTO memory_entities (memory_id, entity_id)
```

关键设计约束：
- 归一化在 `write_memory()` **成功后**异步触发，失败不回滚记忆写入
- LLM 判断使用现有 `LLMProvider.chat()` 接口，prompt 极简（"Are `Postgres` and `PostgreSQL` the same technology entity? Reply YES or NO."）
- 不引入缓存层——实体表数据量有限（＜10 万实体），embedding 搜索直接用 pgvector

### API 新端点

**`GET /api/entities/{entity_id}`**：
- 返回单个实体的完整档案：`{id, name, canonical_name, type, memory_count, source_breakdown}`

**`GET /api/entities/{entity_id}/relations`**：
- 返回该实体的一度关系
- 响应格式：

```json
{
  "entity": {"id": "...", "canonical_name": "PostgreSQL", "type": "technology"},
  "memory_count": 23,
  "source_breakdown": {"git_commit": 12, "conversation": 8, "doc": 3},
  "related_entities": [
    {"entity_id": "...", "name": "pgvector", "type": "technology", "relation_type": "depends_on", "memory_count": 5},
    {"entity_id": "...", "name": "连接池耗尽故障", "type": "event", "relation_type": "causes", "memory_count": 1}
  ],
  "recent_memories": [
    {"memory_id": "...", "summary": "...", "source_type": "git_commit", "created_at": "2026-08-01T00:00:00Z"}
  ]
}
```

**`GET /api/entities/search?q=postgres&type=technology`**：
- 按名称搜索实体，返回匹配的实体列表
- 支持按 entity type 过滤

### Schema 迁移

- 无破坏性变更——`memories` 表结构不变
- 新增两张表：`entities` 和 `memory_entities`
- 迁移脚本包含**回溯填充**逻辑：对现有 memories 中的 entities JSONB 进行归一化，填充新表

### Agent Tool 修改

**修改 `search_memories_tool`**：返回结果中包含 `entities` 字段（每个 memory 关联的归一化实体列表），用户可以看到每条记忆涉及哪些实体。

**新增 `query_entity_tool`**：Agent 可直接按实体名查询实体档案和关联记忆，用户不需要手动切换到 Web 图谱页。

### 前端新增

新增 `EntityGraph` 页面（路由：`/graph`），使用轻量的 Canvas/SVG 图渲染——不引入 D3 或任何图表框架。只渲染一度关系图，可点击节点展开。

组件结构沿用现有模式（`src/components/EntityGraph.tsx` + 对应 hook），API 调用放在 `src/api/` 目录中。

## Testing Decisions

### 测试原则

- Mock 所有外部依赖（LLM、Embedding Provider）
- 测试业务逻辑和数据流转，不测试框架行为
- 测试数据使用固定 UUID 和确定性输入，保证可重复
- 只测试外部可观察行为，不测试内部实现细节

### 测试模块与参考

| 测试文件 | 层级 | 内容 | 参考模式 |
|---------|------|------|---------|
| `tests/unit/test_entity.py` | Unit | `normalize_entities()` 逻辑，LLM mock，embedding mock | `tests/unit/test_memory.py`（同类 service 测试） |
| `tests/unit/test_entity.py` | Unit | `query_entity_relations()` SQL 查询正确性 | `tests/unit/test_retrieval.py`（同类 SQL 查询测试） |
| `tests/unit/test_memory.py` | Unit | 验证 `write_memory()` 在写入后触发 `normalize_entities()` 调用 | 已有测试文件中扩展 |
| `tests/api/test_entity_routes.py` | API | 新端点正常响应、404 处理、参数校验 | `tests/api/test_memory_routes.py`（同类 API 测试） |
| `tests/unit/test_agent_tools.py` | Agent | `query_entity_tool` 返回 JSON 格式正确 | `tests/unit/test_agent_tools.py`（已有 tool 测试） |
| `src/components/EntityGraph.test.tsx` | Frontend | 组件渲染、交互 | `src/components/StatsDashboard.test.tsx`（同类可视化组件测试） |

### 测试用例速览

- `test_normalize_new_entity_creates_record` — 新实体创建
- `test_normalize_duplicate_entity_links_existing` — 相似实体链接到已有
- `test_normalize_entity_llm_confirms_match` — LLM 确认匹配
- `test_normalize_entity_llm_rejects_match` — LLM 拒绝匹配（创建新实体）
- `test_normalize_entity_llm_fails_safe` — LLM 失败时创建新实体（降级）
- `test_query_entity_relations_one_degree` — 一度关系查询返回正确结构
- `test_query_entity_relations_empty` — 无关联实体的实体返回空列表
- `test_api_get_entity_not_found_404` — 不存在的实体返回 404
- `test_write_memory_triggers_normalization` — 写入记忆后自动触发归一化
- `test_search_memories_includes_normalized_entities` — 搜索结果包含归一化实体

### 可观测性指标（为 Phase 3 铺路）

Phase 1 新增三个轻量级指标，不引入新表，纯 SQL 查询或内存计数器。存入现有 `GET /api/memory/stats` 响应体中，为 Phase 3 的主动巡检提供数据基础：

**指标 1: `entity_coverage_ratio`**

已链接到归一化实体的记忆数 / 总记忆数。反映知识图谱的覆盖完整度。Phase 3 的每周巡检需要此指标判断"图谱够不够密"——覆盖率低于阈值时，巡检结论的可信度下降。

```sql
-- 轻量计算：一次查询
SELECT
  COUNT(DISTINCT me.memory_id)::float / NULLIF(COUNT(DISTINCT m.id), 0) AS coverage
FROM memories m
LEFT JOIN memory_entities me ON me.memory_id = m.id
WHERE m.deleted_at IS NULL;
```

**指标 2: `entity_growth_rate`**

过去 7 天新增实体数 / 总实体数。为 Phase 3 的每日巡检提供异常检测基线——如果某一天实体增长率突然飙升（例如连接器首次接入外部系统），巡检应自动调整扫描范围，避免"变化太多看不完"。

**指标 3: `entity_density`**

每记忆的平均关联实体数 = `COUNT(memory_entities) / COUNT(memories)`。反映提取管线的实体产出效率。为 Phase 3 的巡检提供判断依据——如果某批记忆的实体密度异常低（接近 0），可能意味着提取管线出了问题（LLM 降级、JSON 解析失败）而没有被发现。

三个指标统一在 `GET /api/memory/stats` 响应中新增一个 `entity_graph` 字段：

```json
{
  "entity_graph": {
    "coverage_ratio": 0.78,
    "growth_rate_7d": 0.05,
    "density": 3.2,
    "total_entities": 142
  }
}
```

无新表，无新依赖，只有 2 条新增 SQL 查询。Phase 3 巡检 prompt 中直接引用这些指标值。

## Out of Scope

- 多跳图遍历（2 跳以上）
- 图算法（中心性分析、社区发现、最短路径）
- 图数据库（Neo4j 等）——延续 ADR-004
- 实体关系编辑 UI（Phase 1 只做读，不做人工编辑）
- 批量回溯迁移的执行——脚本有，但手动运行
- ChatOps / IDE 插件中的图谱展示——只用 Web UI
- 新依赖引入（无新 Python 包、无新 npm 包、无新数据库）

## Further Notes

- 本 spec 对应 [ADR-006](./decisions/ADR-006-extension-roadmap.md) 的 Phase 1
- 领域术语定义见 [domain-model.md](./domain-model.md)
- Phase 1 完成后，Phase 2 多源连接器依赖实体的归一化层将外部数据链接到已有知识
- 回溯迁移脚本需要一个**确认机制**：首次运行时列出所有 LLM 判断的实体匹配，由用户确认后再执行批量插入
- 实体嵌入使用现有的 `EmbeddingProvider`（BGE-M3），不需要新的 embedding 策略
