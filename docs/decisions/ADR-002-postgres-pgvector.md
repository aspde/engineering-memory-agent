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

**Pinecone** — 未采用。SaaS 托管零运维，但有三个硬伤：(1) 数据在外部——EMA 的记忆是团队的私有知识资产，保存在外部服务上有合规风险；(2) 成本不可控——向量存储按 pod 计费，长期运行成本远高于自托管 PG；(3) 与业务数据分离——EMA 的记忆需要结构化字段（source_type、recall_count、entities）和向量在同一个查询中，Pinecone 只能存向量+metadata，结构化查询能力弱。

**pgvector + Neo4j** — 已拒绝。详见 [ADR-004](./ADR-004-no-neo4j.md)。双写一致性负担、两套部署运维、两套查询语法，换不来可验证的用户价值。

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
| `chunks` | `embedding` | 1024 | HNSW, cosine | 文档片段向量检索 |
| `memories` | `embedding` | 1024 | HNSW, cosine | 记忆语义搜索 |
| `entities` | `embedding` | 1024 | HNSW, cosine | Phase 1 实体归一化匹配 |

### 索引策略

全部使用 **HNSW** 索引 + **cosine distance** (`<=>` 操作符)：

```sql
-- 余弦相似度检索
SELECT id, content, 1 - (embedding <=> :vec ::vector) AS similarity
FROM chunks
WHERE 1 - (embedding <=> :vec ::vector) > :threshold
ORDER BY embedding <=> :vec ::vector
LIMIT :limit;
```

**演进说明**：早期实施用 IVFFlat（lists=100），小语料上暴露了聚类依赖问题——数据量 < 千条时 lists 把探针散布到大量空聚类，召回率下降（见 seed-010 的教训）。已迁移到 HNSW（pgvector ≥ 0.5）：HNSW 无聚类质心依赖，小库也稳，且建索引用默认参数。基线迁移直接用 HNSW 建索引；历史 ivfflat 库开发期重建即可（`init_db` 不做运行时索引替换）。`search_memories` 单段按 HNSW 索引顺序取 `top_k`（索引只服务 `ORDER BY embedding <=> :vec` 的扫描），不再需要两段候选窗 + Python 重排——早期两段重排是配合「相似度 × decay_factor」加权排序，decay 已移除（见 [ADR-009](./ADR-009-decay-weighting-removed.md) 与 decision-faq 第 7 节）。

### 检索模式

两套检索管道，都走 pgvector：

| 管道 | 入口 | 流程 |
|------|------|------|
| Chunk 检索 | `retrieve(query)` | embed → vector_search(HNSW) → （可选 rerank） → 返回 |
| Memory 检索 | `query_memories(query)` | embed → search_memories(相似度) → （可选 rerank） → record_recalls → 返回 |

两套共用同一个 `EmbeddingProvider`（BGE-M3 / 1024 维）。**rerank 默认关闭**（`use_cross_encoder=False` / `use_llm_rerank=False`，opt-in）——eval 显示小语料下 cross-encoder rerank 延迟高 ~90 倍且 recall 更低（0.967 vs 1.000），见 [eval-report.md](../../tests/eval/reports/eval-report.md)。召回统计为 `record_recalls`（单条 `UPDATE ... WHERE id = ANY(:ids)` 批量递增 `recall_count`/`recalled_at`，无 N+1、并发不丢计数），只作元数据、不参与排序。

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

> **2026-08-11 更正**：阈值已标定调整（[threshold_calibration_report.md](../../tests/eval/reports/archive/threshold_calibration_report.md)）——0.92 高到同义改写对一半漏 merge，改后为 ≥0.85 合并、0.72-0.85 冲突检测、0.60-0.72 补充关联、<0.60 新插入（`backend/service/memory.py`）。本条保留原始决策值供追溯。

## Trade-offs & Limitations

### 已知限制

| 限制 | 影响 | 缓解 |
|------|------|------|
| HNSW 在极端高维大数据量下的内存占用 | 百万级向量时索引内存开销上升 | EMA 数据量（数千记忆 + 数万 chunks）远未触及；`search_memories` 两阶段召回（先 HNSW 取 `top_k×8` 候选再 Python 重排）兼顾质量与内存 |
| 向量 + 结构化字段在同一行的存储压力 | memories 表行宽较大（summary + entities JSONB + relations JSONB + embedding + meta） | 当前行宽仍在 PG 合理范围内（单行 < 10KB）。未来若膨胀，entities/relations 移到独立表（Phase 1 已做） |
| 无 GPU 加速 | BGE-M3 嵌入计算在 CPU 上运行 | BGE-M3 1024 维的单条嵌入耗时 150-230ms（CPU，locust 实测），批量嵌入合并降低开销 |
| 向量维度假定为 1024 | 切换到不同维度的 embedding 模型需要重建数据库 | 1024 是 BGE-M3 的维度，由配置映射（见 `backend/shared/config.py`）。切换模型时 `init_db()` 检测维度不一致会启动失败，需 `python -m scripts.recreate_db` 重建后重摄取（自动迁移会清空向量列，故明确不做） |

### 不会做的事

- **不引入 pgvector 以外的向量存储**（保持单数据库约束）
- **不做向量量化压缩**（1024 维 × float32 = 4KB/向量，数据量级不需要）
- **不扩展维度到超过 BGE-M3 的 1024**（当前模型够用）

## Consequences

- ✅ 架构简单——一个 `docker compose up` 启动所有存储，零额外服务
- ✅ 结构化查询 + 向量搜索在同一个 SQL 事务中完成，无一致性负担
- ✅ LangGraph checkpoints 复用同一 PG 实例，对话持久化零额外部署
- ✅ 运维成本低——备份一个 PG 实例 = 备份全部数据
- ⚠️ 向量索引调优需要 PG 知识（HNSW m/ef_construction 参数等），不在标准 DBA 技能范围内
- ⚠️ 超大规模向量检索能力有限（百万级以上需要评估 HNSW 迁移或引入专用向量数据库）
- ⚠️ 与 PG 版本绑定——pgvector 的 PG16 支持最完整，跨 PG 大版本升级需要注意 pgvector 兼容性
