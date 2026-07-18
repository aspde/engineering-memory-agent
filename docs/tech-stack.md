# Technology Stack

## Backend

- **Language**: Python 3.12
- **Framework**: FastAPI
- **Async**: async/await + httpx

Python 拥有成熟的 AI 工程生态（LangGraph、OpenAI SDK、sentence-transformers 等），适合 Agent 编排场景。

## Agent Framework

- **Selected**: LangGraph
- **Reason**: 需要状态管理、条件分支、Tool Calling、Memory 集成

设计原则：`OpenAI SDK + LangGraph + 自研 Memory System`，不依赖黑盒 Agent 框架。

## LLM

通过 `LLMProvider` 抽象接口统一封装，业务代码不依赖具体模型：

| Provider | SDK | Config Switch |
|----------|-----|---------------|
| OpenAI 兼容 (DeepSeek 等) | openai | default |
| Anthropic Claude | anthropic | `LLM_PROVIDER=anthropic` |

支持通过环境变量切换 provider，无需改代码。

## Embedding

通过 `EmbeddingProvider` 抽象接口统一封装：

| Provider | Model | Deployment |
|----------|-------|------------|
| BGE (local) | BAAI/bge-m3 | sentence-transformers, 本地 |

未来可扩展 OpenAI Embedding 等 provider。

## Vector Database

- **Selected**: PostgreSQL + pgvector
- **Reason**: 一个数据库同时解决结构化数据和向量检索

## Frontend

- **MVP**: Streamlit (skeleton 已就绪)
- **Production**: React (planned)

## Dependencies (key)

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
