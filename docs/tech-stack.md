# Technology Stack

## Backend

### Language

Python 3.12

### Framework

FastAPI


## Reason

项目核心不是传统高并发业务系统，而是 AI Agent 编排。

Python 拥有成熟的 AI 工程生态：

- LangChain
- LangGraph
- LlamaIndex
- OpenAI SDK
- Embedding 模型生态

因此选择 Python 作为主要开发语言。


---

# Agent Framework

## Selected

LangGraph


## Reason

项目需要支持：

- 状态管理
- 多步骤流程
- Memory
- Tool Calling


典型流程：

```
用户问题
 ↓
判断问题类型
 ↓
查询代码
 ↓
查询历史
 ↓
查询 Memory
 ↓
生成答案
```


LangGraph 适合复杂 Agent Workflow 和状态管理。


---

## Design Principle

采用：

```
OpenAI SDK
+
LangGraph
+
自研 Memory System
```

避免完全依赖黑盒 Agent 框架。


需要能够解释：

- 为什么这样设计 Memory
- 如何避免上下文爆炸
- 如何实现长期记忆召回
- 如何管理 Agent 状态
- Redis 和向量数据库如何分工


后续如果需要更多 RAG 组件或工具生态，再考虑引入 LangChain。


---

# LLM

## Selected Model

DeepSeek


## Design Requirement

业务代码不能绑定具体模型。


统一抽象：

```python
llm = ModelProvider()
```


支持未来切换：

- DeepSeek
- OpenAI
- Claude
- Local LLM


---

# Vector Database

## Selected

PostgreSQL + pgvector


## Reason

使用一个数据库同时解决：

- 结构化数据存储
- 向量数据检索


存储：

- 用户信息
- 项目信息
- Memory
- Embedding


适合企业级应用。


---

## Alternatives


### ChromaDB

用途：

MVP 快速验证。


不作为最终生产存储。


---

### Milvus

不选择。


原因：

- 部署复杂
- 运维成本高
- MVP 阶段过重


适合：

大规模纯向量检索场景。


---

# Embedding


## Options


### OpenAI Embedding

优势：

- 效果稳定
- 快速验证 MVP


### BGE

优势：

- 本地部署
- 数据不外传
- 成本可控


---

## Design


通过：

```
EmbeddingService
```

进行抽象。


禁止业务代码直接依赖具体 Embedding 模型。


---

## Evolution


### Phase 1

OpenAI Embedding

目标：

快速验证系统。


### Phase 2

BGE 本地部署。


优化：

- 成本
- 隐私
- 数据安全