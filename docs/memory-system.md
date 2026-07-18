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
| Short Term | Session state | 当前对话上下文 | 🔜 |
| Long Term | PostgreSQL | 结构化知识、元数据 | 🔜 |
| Vector Memory | pgvector | Embedding 语义检索 | 🔜 |

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

## Design Requirements

- Memory 作为独立能力模块
- Retrieval 独立封装
- Embedding 模型可替换
- 不绑定单一模型或存储方案
