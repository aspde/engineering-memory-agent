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
| backend | FastAPI | 5 个 API 端点已实现，待容器化 |
| frontend | Streamlit | 骨架已就绪，交互界面待实现 |

## Configuration

通过 `.env` 文件管理环境变量：

- `LLM_*` — LLM provider 配置
- `EMBEDDING_*` — Embedding 模型配置
- `DATABASE_URL` — PostgreSQL 连接
- `MAX_AGENT_STEPS` — Agent 最大工具调用次数
- `APP_ENV` — 运行环境 (development / test / production)

## Runtime Architecture

```
Streamlit (Frontend) → FastAPI (Backend) → LangGraph Agent
                                               ↓
                                         Memory Layer
                                               ↓
                                    PostgreSQL + pgvector
                                    (chunks + memories + checkpoints)
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

# Run frontend
streamlit run frontend/app.py

# Run tests
pytest
```
