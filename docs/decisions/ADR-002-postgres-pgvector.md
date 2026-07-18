# ADR-002: Use PostgreSQL + pgvector

## Status

Accepted

## Context

系统需要同时存储业务数据、Memory 数据和 Embedding 向量，需要结构化查询和向量搜索两种能力。

## Decision

采用 **PostgreSQL + pgvector**，统一管理关系数据和向量数据。

## Alternatives

### Milvus

Rejected — 运维复杂，MVP 阶段过重。

### ChromaDB

Rejected as production storage — 适合快速验证，不适合企业级长期存储。

## Consequences

- ✅ 架构简单，一个数据库解决问题，运维成本低
- ⚠️ 超大规模向量检索能力有限；不适合十亿级以上向量场景
