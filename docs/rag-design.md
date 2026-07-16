# RAG Design


## Architecture


```
LangGraph

+

Custom Retriever

+

PostgreSQL pgvector
```


---

# Design Principle


不完全依赖黑盒 RAG。


自行控制：

- Chunk
- Embedding
- Retrieval
- Rerank
- Context Assembly


---

# Retrieval Pipeline


```
Query

 ↓

Embedding

 ↓

Vector Search

 ↓

Reranker

 ↓

LLM
```


---

# Reranker


后期加入：

bge-reranker


作用：

提升召回结果相关性。


---

# Goals


实现：

- 可解释检索
- 可调优流程
- 企业级 RAG 架构