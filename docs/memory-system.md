# Memory System Design

## Goal

构建研发团队长期知识记忆，保存：

- 技术决策
- 代码变更
- Bug 经验
- Review 记录
- 项目上下文

## Memory Layers

| Layer | Storage | Purpose | Status |
|-------|---------|---------|--------|
| Short Term | Session state | 当前对话上下文 | 设计完成，待实现 |
| Long Term | PostgreSQL | 结构化知识、元数据 | 设计完成，待实现 |
| Vector Memory | pgvector | Embedding 语义检索 | 设计完成，待实现 |

## Memory Pipeline

```
Input → Chunk → Embedding → Store (pgvector)
                                ↓
Query → Embedding → Vector Search → Rerank → Context Assembly → LLM
```

## Retrieval

- 向量检索：pgvector 相似度查询
- Rerank：可选 bge-reranker 提升相关性（后期加入）
- Context Assembly：将检索结果拼装为 LLM 上下文

每个环节可控、可调优、可替换：Chunk 策略、Embedding 模型、Retrieval 算法、Rerank 策略、Context Assembly 方式。

## Embedding

- BGE-M3 (sentence-transformers)，本地部署
- 通过 `EmbeddingProvider` 抽象接口，可替换为 OpenAI Embedding 等

## Knowledge Sources — Git

GitPython 解析 Git 仓库，提取 Commit 记录（message、author、date）、Diff 变更、文件变更列表、Branch/Tag 信息、文件历史。

```
Git Repo → GitPython → Parse → Chunk → Embedding → Store → Retrieval
```

分阶段实现：

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | commit message、author、changed files | 待实现 |
| Phase 2 | Commit diff 解析 + LLM 总结 + 自动 Memory | 待实现 |
| Phase 3 | Issue、Pull Request、Code Review、ADR 扩展 | 待实现 |

## Design Requirements

- Memory 作为独立能力模块
- Retrieval 独立封装
- Embedding 模型可替换
- 不绑定单一模型或存储方案
