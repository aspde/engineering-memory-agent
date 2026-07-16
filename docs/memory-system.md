# Memory System Design


## Goal

构建研发团队长期知识记忆。


保存：

- 技术决策
- 代码变化
- Bug经验
- Review记录
- 项目上下文


---

# Memory Layers


## Short Term Memory

当前对话上下文。


## Long Term Memory

PostgreSQL存储。


## Vector Memory

pgvector存储embedding。


## Cache

Redis缓存热点数据。


---

# Memory Pipeline


Input

↓

Chunk

↓

Embedding

↓

Store

↓

Retrieve

↓

Context Assembly

↓

LLM