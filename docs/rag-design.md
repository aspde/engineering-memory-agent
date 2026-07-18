# RAG Design

## Architecture

自行控制 RAG 全流程，不完全依赖黑盒框架：

```
LangGraph Orchestrator
       +
Custom Retriever
       +
PostgreSQL + pgvector
```

## Design Principle

每个环节可控、可调优、可替换：

- Chunk 策略
- Embedding 模型
- Retrieval 算法
- Rerank 策略
- Context Assembly 方式

## Retrieval Pipeline

```
Query → Embedding → Vector Search (pgvector) → Reranker → Context → LLM
```

## Embedding

当前实现：

- BGE-M3 (sentence-transformers)，本地部署
- 通过 `EmbeddingProvider` 抽象接口，可替换为 OpenAI Embedding 等

## Reranker

后期加入 bge-reranker，提升召回结果相关性。

## Goals

- 可解释的检索过程
- 可调优的检索流程
- 企业级 RAG 架构
