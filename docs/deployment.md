# Deployment

## Container

Docker Compose 编排：

```yaml
services:
  postgres:      # PostgreSQL + pgvector
```

当前仅包含数据库容器。后续将加入 backend、frontend 等服务。

## Services

| Service | Image | Status |
|---------|-------|--------|
| postgres | pgvector/pgvector:pg16 | 已就绪 |
| backend | FastAPI + Python 3.12 | 10 个 API 端点已实现，待容器化 |
| frontend | React + TypeScript + Vite | SPA 已实现（聊天页、记忆库页），待容器化 |

## Configuration

通过 `.env` 文件管理环境变量：

- `LLM_*` — LLM provider 配置
- `EMBEDDING_*` — Embedding 模型配置
- `DATABASE_URL` — PostgreSQL 连接
- `MAX_AGENT_STEPS` — Agent 最大工具调用次数
- `APP_ENV` — 运行环境 (development / test / production)

## Runtime Architecture

```
React SPA (Frontend) → FastAPI (Backend) → LangGraph Agent
                                                 ↓
                                           Memory Layer
                                                 ↓
                                      PostgreSQL + pgvector
                                      (chunks + memories + conversations + checkpoints)
                                                 ↓
                                           LLM Provider
                                                 ↓
                                             Response
```

## Development

```bash
# Start database
docker compose up -d

# Run backend (auto-creates pgvector extension + tables + checkpoint tables)
uvicorn backend.main:app --reload

# Run frontend (dev server with hot reload)
cd frontend && npm run dev

# Build frontend for production (served by FastAPI as static files)
cd frontend && npm run build

# Run tests
pytest
```

## Production

Backend 在生产模式下直接托管前端构建产物（`frontend/dist/`），无需单独运行前端开发服务器：

```bash
# 1. Build frontend
cd frontend && npm run build

# 2. Start backend (serves both API and SPA)
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
