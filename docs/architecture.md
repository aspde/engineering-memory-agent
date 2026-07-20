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
| Frontend | Streamlit (MVP) / React (Production) | Streamlit 骨架已就绪，交互界面待实现 |
| Backend | FastAPI + Python 3.12 | 5 个 API 端点已实现 |
| Agent | LangGraph (手动 StateGraph) | ReAct 循环已实现 (call_llm → tools ⇄ generate_final) |
| Memory | PostgreSQL + pgvector | 记忆写入/检索/衰减/去重全链路已实现 |
| Storage | PostgreSQL + pgvector | docker-compose 已就绪 |
| LLM | OpenAI SDK / Anthropic SDK | provider 抽象 + chat_raw 工具调用已实现 |
| Embedding | BGE-M3 (sentence-transformers) | 本地离线模式已实现 |

## Layer Responsibilities

- **Frontend**: 用户交互、请求提交、结果展示
- **Backend**: API 接口、请求生命周期、调用 Agent
- **Agent**: ReAct 工具调用循环、状态管理、Tool/Memory 编排
- **Memory**: 长期记忆管理、检索、上下文构建、衰减加权
- **Storage**: 业务数据 + 向量存储
- **LLM**: 统一模型调用封装，支持多 provider 切换，支持工具调用 (chat_raw)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/memory/ingest` | 文档分块 → 嵌入 → 存入 chunks 表 |
| POST | `/api/memory/search` | 语义搜索：嵌入 → 向量检索 → rerank |
| POST | `/api/memory/memories/write` | 结构化记忆写入：提取 → 相似度分级 → 合并/冲突/新插入 |
| POST | `/api/memory/memories/search` | 记忆搜索：衰减加权 → rerank → 更新 decay |
| POST | `/api/agent/chat` | Agent 对话：ReAct 循环 + 工具调用 + 上下文生成 |

## Technology Stack

### Backend

- **Language**: Python 3.12
- **Framework**: FastAPI
- **Async**: async/await + httpx

### Agent Framework

- **Selected**: LangGraph
- **Approach**: 手动 `StateGraph` ReAct 循环，不使用预建 Agent
- **Reason**: 为 Human-in-the-Loop (`interrupt`)、对话持久化 (`PostgresSaver`)、多步工作流等未来场景预留入口

### LLM

通过 `LLMProvider` 抽象接口统一封装，业务代码不依赖具体模型：

| Provider | SDK | Config Switch |
|----------|-----|---------------|
| OpenAI 兼容 (DeepSeek 等) | openai | `LLM_PROVIDER=deepseek` |
| Anthropic Claude | anthropic | `LLM_PROVIDER=anthropic` |

Agent 工具调用通过新增的 `chat_raw(messages, tools, **kwargs) → dict` 方法，返回结构化响应 `{content, tool_calls}`。

### Embedding

通过 `EmbeddingProvider` 抽象接口统一封装：

| Provider | Model | Deployment |
|----------|-------|------------|
| BGE (local) | BAAI/bge-m3 | sentence-transformers, 本地离线模式 |

未来可扩展 OpenAI Embedding 等 provider。

### Vector Database

- **Selected**: PostgreSQL + pgvector
- **Reason**: 一个数据库同时解决结构化数据 + 向量检索 + Agent 对话 checkpoints

### Memory System

详见 [memory-system.md](memory-system.md)。核心能力：

- **文档索引与检索**：分块 → 嵌入 → pgvector，双 reranker（cross-encoder / LLM）
- **三阶段记忆提取**：摘要 + 实体并行提取 → 关系提取
- **四级相似度去重**：≥0.92 合并，0.75–0.92 冲突检测，0.60–0.75 补充关联，<0.60 新插入
- **艾宾浩斯遗忘衰减**：`R = e^(-t/S)`，召回时自动更新

### Agent

详见 [agent-design.md](agent-design.md)。核心能力：

- **ReAct 工具调用循环**：6 个 tool 封装记忆检索、文档搜索、记忆写入、知识提取、Git 摄取、文档摄取
- **对话连续性**：thread_id 维持跨轮次上下文
- **容错降级**：LLM 调用失败不终止图执行

### Frontend

- **MVP**: Streamlit (骨架已就绪)
- **Production**: React (planned)

### Key Dependencies

| Category | Library | Purpose |
|----------|---------|---------|
| Web | fastapi, uvicorn, httpx | API server |
| Agent | langgraph | Workflow orchestration + checkpoints |
| LLM | openai, anthropic | Provider SDKs |
| Embedding | sentence-transformers | BGE-M3 |
| Database | pgvector, asyncpg, SQLAlchemy | Storage + vector search |
| Frontend | streamlit | MVP UI |
| Git | pygit2 | Repository history ingestion |
| Testing | pytest, pytest-asyncio | Test framework |

## Design Principles

详见 [.claude/rules/constraints.md](../.claude/rules/constraints.md)。
