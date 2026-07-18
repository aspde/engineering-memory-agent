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
| backend | FastAPI | 待容器化 |
| frontend | Streamlit | 待容器化 |

## Configuration

通过 `.env` 文件管理环境变量：

- `LLM_*` — LLM provider 配置
- `EMBEDDING_*` — Embedding 模型配置
- `DATABASE_URL` — PostgreSQL 连接
- `APP_ENV` — 运行环境 (development / test / production)

## Runtime Architecture

```
FastAPI (Backend)
    ↓
PostgreSQL + pgvector (Storage)
    ↓
LLM Provider (External API)
```

## Development

```bash
# Start database
docker compose up -d

# Run backend
uvicorn backend.main:app --reload

# Run frontend
streamlit run frontend/app.py

# Run tests
pytest
```
