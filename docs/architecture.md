# System Architecture

## Overview

EMA (Engineering Memory Agent) — 面向研发团队的长期记忆智能体。

将研发过程中的代码知识、Git 历史、技术决策、故障经验转化为可检索、可复用的长期记忆。

## High Level Architecture

```
User → Frontend (React) → FastAPI Backend → Agent Layer (LangGraph)
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
| Frontend | React + TypeScript + Vite + Tailwind CSS | 聊天页、记忆库页、HITL 审批流已实现 |
| Backend | FastAPI + Python 3.12 | 11 个 API 端点已实现（含 SSE 流式 + HITL） |
| Agent | LangGraph (手动 StateGraph) | ReAct 循环已实现 (call_llm → tools ⇄ generate_final) |
| Memory | PostgreSQL + pgvector | 记忆写入/检索/衰减/去重全链路已实现 |
| Entity Graph | PostgreSQL + pgvector | 实体归一化、一度关系查询、图谱可视化已实现 |
| Storage | PostgreSQL + pgvector | docker-compose 已就绪 |
| LLM | OpenAI SDK / Anthropic SDK | provider 抽象 + chat_raw 工具调用 + chat_json 结构化输出已实现 |
| Embedding | BGE-M3 (local) / OpenAI (API) | 本地离线 + OpenAI 兼容 API 双模式 |

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
| POST | `/api/agent/chat` | Agent 对话（非流式）：ReAct 循环 + 工具调用 |
| POST | `/api/agent/chat/stream` | Agent 对话（SSE 流式）：逐 token 输出 + interrupt |
| GET | `/api/agent/threads` | 获取对话历史列表 |
| GET | `/api/agent/thread/{thread_id}` | 获取指定对话消息历史 |
| POST | `/api/memory/ingest` | 文档分块 → 嵌入 → 存入 chunks 表 |
| POST | `/api/memory/search` | 语义搜索：嵌入 → 向量检索 → rerank |
| POST | `/api/memory/memories/write` | 结构化记忆写入：提取 → 相似度分级 → 合并/冲突/新插入 |
| POST | `/api/memory/memories/search` | 记忆搜索：衰减加权 → rerank → 更新 decay |
| GET | `/api/memory/memories/{memory_id}` | 通过 ID 获取单条记忆 |
| DELETE | `/api/memory/memories/{memory_id}` | 软删除记忆（设置 deleted_at） |
| GET | `/api/memory/stats` | 记忆库统计信息（总数、来源分布、高频实体、知识图谱指标等） |
| GET | `/api/entities/{entity_id}` | 获取实体档案：名称、类型、关联记忆数、来源分布 |
| GET | `/api/entities/{entity_id}/relations` | 获取实体一度关系：关联实体 + 最近记忆 |
| GET | `/api/entities/search?q=&type=` | 按名称搜索实体，支持类型过滤 |

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

结构化输出（实体/关系提取、矛盾检测、实体匹配）通过 `chat_json(messages, json_schema, **kwargs) → str` 强制合法 JSON——OpenAI 兼容用 `response_format=json_object`，Anthropic 用 forced `tool_use`。`chat_structured` 统一负责解析、`jsonschema` 校验、退避重试，重试耗尽抛 `LLMStructuredError`。

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

- **React + TypeScript + Vite + Tailwind CSS**
- 纯客户端 SPA，2 个页面：聊天页、记忆库页
- SSE 流式聊天、Human-in-the-Loop 审批/冲突解决

### Key Dependencies

| Category | Library | Purpose |
|----------|---------|---------|
| Web | fastapi, uvicorn, httpx | API server |
| Agent | langgraph | Workflow orchestration + checkpoints |
| LLM | openai, anthropic | Provider SDKs |
| Embedding | sentence-transformers | BGE-M3 |
| Database | pgvector, asyncpg, SQLAlchemy | Storage + vector search |
| Frontend | react, react-router-dom, tailwindcss | SPA UI |
| Frontend build | vite, typescript | Build toolchain |
| Git | pygit2 | Repository history ingestion |
| Testing | pytest, pytest-asyncio | Test framework |

## Design Principles

详见 [.claude/rules/constraints.md](../.claude/rules/constraints.md)。
