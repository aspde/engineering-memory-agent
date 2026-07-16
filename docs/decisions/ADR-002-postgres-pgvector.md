# ADR-002 Use PostgreSQL + pgvector


## Status

Accepted


## Context

系统需要同时存储：

- 业务数据
- Memory 数据
- Embedding 向量


需要：

- 结构化查询
- 向量搜索


---

## Decision

采用：

PostgreSQL + pgvector


统一管理：

- Relational Data
- Vector Data


---

## Alternatives


### Milvus

Rejected。


原因：

- 运维复杂
- MVP 阶段过重


### ChromaDB

Rejected as production storage。


原因：

适合快速验证，不适合作为长期企业存储。


---

## Consequences


优势：

- 架构简单
- 数据一致
- 降低运维成本


限制：

超大规模向量检索能力有限。