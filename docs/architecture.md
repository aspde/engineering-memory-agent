# System Architecture

## Overview

EMA (Engineering Memory Agent) — 面向研发团队的长期记忆智能体。

将研发过程中的代码知识、Git 历史、技术决策、故障经验转化为可检索、可复用的长期记忆。

## High Level Architecture

```
User → Frontend (Streamlit) → FastAPI Backend → Agent Layer (LangGraph)
                                                    ↓
                                              Memory Layer
                                                    ↓
                                         PostgreSQL + pgvector
                                                    ↓
                                               LLM Provider
                                                    ↓
                                               Response
```

## Layers

| Layer | Technology | Status |
|-------|-----------|--------|
| Frontend | Streamlit (MVP) / React (Production) | Streamlit 骨架已就绪 |
| Backend | FastAPI + Python 3.12 | API 骨架 + provider 抽象已实现 |
| Agent | LangGraph | 设计完成，待实现 |
| Memory | PostgreSQL + pgvector | 数据库已就绪，记忆逻辑待实现 |
| Storage | PostgreSQL + pgvector | docker-compose 已就绪 |
| LLM | OpenAI SDK / Anthropic SDK | provider 抽象已实现 |
| Embedding | BGE-M3 (sentence-transformers) | 已实现 |

## Layer Responsibilities

- **Frontend**: 用户交互、请求提交、结果展示
- **Backend**: API 接口、请求生命周期、调用 Agent
- **Agent**: 工作流编排、状态管理、Tool/Memory 调用
- **Memory**: 长期记忆管理、检索、上下文构建
- **Storage**: 业务数据 + 向量存储
- **LLM**: 统一模型调用封装，支持多 provider 切换

## Technology Stack

### Backend

- **Language**: Python 3.12
- **Framework**: FastAPI
- **Async**: async/await + httpx

Python 拥有成熟的 AI 工程生态（LangGraph、OpenAI SDK、sentence-transformers 等），适合 Agent 编排场景。

### Agent Framework

- **Selected**: LangGraph
- **Reason**: 需要状态管理、条件分支、Tool Calling、Memory 集成

设计原则：`OpenAI SDK + LangGraph + 自研 Memory System`，不依赖黑盒 Agent 框架。

### LLM

通过 `LLMProvider` 抽象接口统一封装，业务代码不依赖具体模型：

| Provider | SDK | Config Switch |
|----------|-----|---------------|
| OpenAI 兼容 (DeepSeek 等) | openai | default |
| Anthropic Claude | anthropic | `LLM_PROVIDER=anthropic` |

支持通过环境变量切换 provider，无需改代码。

### Embedding

通过 `EmbeddingProvider` 抽象接口统一封装：

| Provider | Model | Deployment |
|----------|-------|------------|
| BGE (local) | BAAI/bge-m3 | sentence-transformers, 本地 |

未来可扩展 OpenAI Embedding 等 provider。

### Vector Database

- **Selected**: PostgreSQL + pgvector
- **Reason**: 一个数据库同时解决结构化数据和向量检索

### Frontend

- **MVP**: Streamlit
- **Production**: React (planned)

### Key Dependencies

| Category | Library | Purpose |
|----------|---------|---------|
| Web | fastapi, uvicorn, httpx | API server |
| Agent | langgraph | Workflow orchestration |
| LLM | openai, anthropic | Provider SDKs |
| Embedding | sentence-transformers | BGE-M3 |
| Database | pgvector, psycopg2-binary, SQLAlchemy | Storage |
| Frontend | streamlit | MVP UI |
| Git | GitPython | Repository analysis |
| Testing | pytest, pytest-asyncio | Test framework |

## Design Principles

详见 [.claude/rules/constraints.md](../.claude/rules/constraints.md)。
