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
| Frontend | Streamlit (MVP) / React (Production) | ✅ Streamlit skeleton |
| Backend | FastAPI + Python 3.12 | ✅ API scaffold |
| Agent | LangGraph | 🔜 Planned |
| Memory | PostgreSQL + pgvector | ✅ DB ready, 🔜 memory logic |
| Storage | PostgreSQL + pgvector | ✅ docker-compose |
| LLM | OpenAI SDK / Anthropic SDK | ✅ provider abstraction |
| Embedding | BGE-M3 (sentence-transformers) | ✅ implemented |

## Layer Responsibilities

- **Frontend**: 用户交互、请求提交、结果展示
- **Backend**: API 接口、请求生命周期、调用 Agent
- **Agent**: 工作流编排、状态管理、Tool/Memory 调用
- **Memory**: 长期记忆管理、检索、上下文构建
- **Storage**: 业务数据 + 向量存储
- **LLM**: 统一模型调用封装，支持多 provider 切换

## Design Principles

- 简单优先：清晰边界、避免过度设计
- 组件可替换：LLM、Embedding 通过抽象接口切换
- 单 Agent 架构：不引入 Multi-Agent
