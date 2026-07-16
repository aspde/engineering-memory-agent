# System Architecture


## Overview

研发协作长期记忆智能体。


目标：

将研发过程中的：

- 代码知识
- Git 历史
- 技术决策
- 故障经验

转化为可检索、可复用的长期记忆。


最终形成：

```
代码知识
+
提交历史
+
设计决策
+
故障经验

=

研发团队长期记忆
```


---

# High Level Architecture


```
User

 ↓

Frontend

 ↓

FastAPI Backend

 ↓

LangGraph Agent

 ↓

Retriever

 ↓

PostgreSQL + pgvector

 ↓

LLM

 ↓

Response
```


---

# Core Components


## Backend

FastAPI


## Agent

LangGraph


## Storage

PostgreSQL + pgvector


## Cache

Redis


## Frontend

MVP:

Streamlit


Future:

React