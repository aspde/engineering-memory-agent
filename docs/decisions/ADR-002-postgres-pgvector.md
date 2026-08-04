# ADR-002: Use PostgreSQL + pgvector

**日期**: 2026-07-18
**更新**: 2026-08-04（补充向量数据库选型分析、当前实施细节）

**状态**: 已接受

## Context

系统需要同时存储业务数据（conversations、memories、chunks）、Memory 数据和 Embedding 向量，需要结构化查询和向量搜索两种能力。此外，LangGraph Agent 的对话状态需要 checkpoint 持久化。

在选择存储方案时，评估了三类：纯向量数据库、向量+关系分离组合、单数据库（PostgreSQL + pgvector）。

## Decision

采用 **PostgreSQL + pgvector**，统一管理关系数据、向量数据和 Agent checkpoints。不引入专用向量数据库。

## Vector Database Selection Analysis

### 候选方案

| 方案 | 类型 | 成熟度 | 运维复杂度 | 向量能力 | 适用场景 |
|------|------|--------|-----------|---------|---------|
| **PostgreSQL + pgvector** | 关系 + 向量扩展 | 生产级 | 低（已有 PG 部署） | 中（百万级向量） | 向量搜索+结构化查询混合 |
| Milvus | 专用向量数据库 | 生产级 | 高（独立部署+etcd+MinIO） | 高（十亿级向量） | 纯向量搜索，大规模 |
| Qdrant | 专用向量数据库 | 生产级 | 中（独立部署） | 高 | 向量搜索+payload 过滤 |
| Weaviate | 向量+对象存储 | 生产级 | 中 | 高（内置 embedding） | 全托管向量搜索 |
| Chroma | 嵌入式向量库 | 轻量 | 低 | 低-中 | 原型验证，本地开发 |
| Pinecone | 托管向量服务 | 生产级 | 零（SaaS） | 高 | 不想管基础设施 |
| pgvector + Neo4j | 双数据库 | 视组合而定 | 高（双写+一致性） | 中 | 图+向量混合查询 |

### 逐一排除

**Milvus** — 已拒绝。当时在 0.1 版本 EMA 中评估过。Milvus 需要独立部署整套基础设施（etcd 协调 + MinIO 对象存储 + Milvus 本身），对于 EMA 的数据规模（初始几千条记忆，远期几万到几十万条）是严重过度设计。Milvus 的优势（十亿级 ANN、GPU 索引、分段存储）在 EMA 的场景中完全用不到。

**Qdrant** — 未采用。Qdrant 的 payload 过滤能力比 pgvector 原生的 `WHERE` 子句更灵活，但 EMA 的过滤需求简单（按 `document_id`、`project`、`source_type` 过滤），SQL `WHERE` 完全够用。引入 Qdrant 增加一个独立服务 + 独立数据备份 + 独立监控的运维负担，收益不匹配。

**Weaviate** — 未采用。Weaviate 的内置 embedding 模块（自动向量化输入的文本）听起来方便，但 EMA 需要精确控制 embedding 管线（选择 BGE-M3、控制 batch size、未来切换到 OpenAI embedding）。内置 embedding 是便利也是耦合——EMA 的 embedding 策略需要通过 `EmbeddingProvider` 接口独立演进。

**Chroma** — 未采用。Chroma 的开发者体验好（几行代码启动），但设计目标是原型验证而非生产持久化。EMA 从设计之初就以生产使用为目标，Chroma 的持久化机制和数据完整性保障不及 PostgreSQL。

**Pinecone** — 未采用。SaaS 托管零运维，但有三个硬伤：(1) 数据在外部——EMA 的记忆是团队的私有知识资产，保存在外部服务上有合规风险；(2) 成本不可控——向量存储按 pod 计费，长期运行成本远高于自托管 PG；(3) 与业务数据分离——EMA 的记忆需要结构化字段（source_type、decay_factor、entities）和向量在同一个查询中，Pinecone 只能存向量+metadata，结构化查询能力弱。

**pgvector + Neo4j** — 已拒绝。详见 [ADR-005](./ADR-005-no-neo4j.md)。双写一致性负担、两套部署运维、两套查询语法，换不来可验证的用户价值。

### 为什么 pgvector 胜出

pgvector 的决定性优势不是"向量搜索性能最强"，而是**三种数据（结构化、向量、Agent checkpoints）一个数据库**：

```
PostgreSQL
  ├── pgvector 扩展 → 向量存储 + ANN 索引
  ├── 关系表       → conversations, memories, chunks, entities
  └── checkpoints  → LangGraph thread 状态持久化
```

这消除了多数据库之间的一致性问题和运维负担。对于 EMA 的数据规模，pgvector 的性能瓶颈远未到来。

## Current Implementation

### 数据表

| 表 | 向量列 | 维度 | 索引 | 用途 |
|----|--------|------|------|------|
| `chunks` | `embedding` | 1024 | IVFFlat, cosine | 文档片段向量检索 |
| `memories` | `embedding` | 1024 | IVFFlat, cosine | 记忆语义搜索 |
| `entities` | `embedding` | 1024 | IVFFlat, cosine | Phase 1 实体归一化匹配 |

### 索引策略

全部使用 **IVFFlat** 索引 + **cosine distance** (`<=>` 操作符)：

```sql
-- 余弦相似度检索
SELECT id, content, 1 - (embedding <=> :vec ::vector) AS similarity
FROM chunks
WHERE 1 - (embedding <=> :vec ::vector) > :threshold
ORDER BY embedding <=> :vec ::vector
LIMIT :limit;
```

选择 IVFFlat 而非 HNSW 的理由：
- IVFFlat 构建快，内存占用低，写入性能好——EMA 的写入是间歇性的（每次摄入一批 chunks、每次写入一条 memory），不是持续高吞吐
- HNSW 在查询性能上领先，但构建时间更长、内存占用更高。EMA 的数据量（数千条记忆 + 数万条 chunks）尚未触及 IVFFlat 的瓶颈
- HNSW 适合高频查询场景（QPS > 100），EMA 当前是低频查询（用户手动提问）——IVFFlat 的查询延迟（< 10ms）完全在可接受范围内

后续数据量增长到十万级时，可无缝迁移到 HNSW——只需重建索引，不需要改 SQL。

### 检索模式

两套检索管道，都走 pgvector：

| 管道 | 入口 | 流程 |
|------|------|------|
| Chunk 检索 | `retrieve(query)` | embed → vector_search(IVFFlat) → rerank(cross-encoder) → 返回 |
| Memory 检索 | `query_memories(query)` | embed → search_memories(decay_weighted) → rerank(cross-encoder) → update_decay → 返回 |

两套共用同一个 `EmbeddingProvider`（BGE-M3 / 1024 维）和同一个 rerank 策略。

### 写入时的向量操作

每次 `write_memory()` 触发一次 embedding 调用（摘要 → 向量）+ 一次相似度查询（向量 → 已有记忆）：

```sql
-- 四级相似度阈值判断
SELECT id, summary, 1 - (embedding <=> :vec ::vector) AS similarity
FROM memories
WHERE deleted_at IS NULL
  AND 1 - (embedding <=> :vec ::vector) > :threshold  -- 0.60
ORDER BY embedding <=> :vec ::vector
LIMIT 1;
```

四级阈值：≥0.92 合并、0.75-0.92 冲突检测、0.60-0.75 补充关联、<0.60 新插入。全部在 SQL 层面完成，不需要应用层后处理。

## Trade-offs & Limitations

### 已知限制

| 限制 | 影响 | 缓解 |
|------|------|------|
| IVFFlat 在 10 万+ 向量时召回率下降 | 搜索结果可能漏掉相关记忆 | 扩大 `top_k`（检索 4×top_k 候选再 rerank 到 top_k）+ 后续数据增长时迁移 HNSW |
| 向量 + 结构化字段在同一行的存储压力 | memories 表行宽较大（summary + entities JSONB + relations JSONB + embedding + meta） | 当前行宽仍在 PG 合理范围内（单行 < 10KB）。未来若膨胀，entities/relations 移到独立表（Phase 1 已做） |
| 无 GPU 加速 | BGE-M3 嵌入计算在 CPU 上运行 | BGE-M3 1024 维的单条嵌入耗时 < 50ms，批量 32 条 < 500ms，Phase 2 batch_mode 将批量嵌入合并后进一步降低开销 |
| 向量维度假定为 1024 | 切换到不同维度的 embedding 模型需要 ALTER TABLE + 重建索引 | 1024 是 BGE-M3 的维度，已硬编码于 schema 中。若切换到 OpenAI (1536/3072) 或其他模型，需要迁移 |

### 不会做的事

- **不用 pgvector 的 HNSW 索引**（当前数据量下 IVFFlat 足够，后续按需迁移）
- **不引入 pgvector 以外的向量存储**（保持单数据库约束）
- **不做向量量化压缩**（1024 维 × float32 = 4KB/向量，数据量级不需要）
- **不扩展维度到超过 BGE-M3 的 1024**（当前模型够用）

## Consequences

- ✅ 架构简单——一个 `docker compose up` 启动所有存储，零额外服务
- ✅ 结构化查询 + 向量搜索在同一个 SQL 事务中完成，无一致性负担
- ✅ LangGraph checkpoints 复用同一 PG 实例，对话持久化零额外部署
- ✅ 运维成本低——备份一个 PG 实例 = 备份全部数据
- ⚠️ 向量索引调优需要 PG 知识（IVFFlat lists 参数、HNSW m/ef_construction 参数等），不在标准 DBA 技能范围内
- ⚠️ 超大规模向量检索能力有限（百万级以上需要评估 HNSW 迁移或引入专用向量数据库）
- ⚠️ 与 PG 版本绑定——pgvector 的 PG16 支持最完整，跨 PG 大版本升级需要注意 pgvector 兼容性
