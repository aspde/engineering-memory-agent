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
| backend | FastAPI + Python 3.12 | 11 个 API 端点已实现，待容器化 |
| frontend | React + TypeScript + Vite | SPA 已实现（聊天页、记忆库页），待容器化 |

## Configuration

通过 `.env` 文件管理环境变量：

- `LLM_*` — LLM provider 配置（含传输韧性 `LLM_RETRY_*` / `LLM_CIRCUIT_BREAKER_*`、结构化重试 `LLM_STRUCTURED_*`）
- `EMBEDDING_*` — Embedding 模型配置
- `DATABASE_URL` — PostgreSQL 连接
- `MAX_AGENT_STEPS` — Agent 最大工具调用次数
- `AGENT_TIMEOUT` — Agent 单回合总超时（秒）
- `PATROL_*` — 巡检调度与超时（`PATROL_ENABLED` / `PATROL_DAILY_HOUR` / `PATROL_WEEKLY_*` / `PATROL_TIMEOUT`）
- `LOG_LEVEL` — 日志级别（DEBUG / INFO / WARNING / ERROR）
- `APP_ENV` — 运行环境 (development / test / production)

## Health & Startup

- **启动配置校验**：启动时运行 `validate_config()`，检查巡检时间范围（`PATROL_DAILY_HOUR` 等 0-23 / 0-6）、重试与熔断参数边界、`LLM_API_KEY` 非空（`APP_ENV=test` 豁免）。配置非法直接拒绝启动，而非等首个请求才报错或静默产生无效调度。
- **健康检查**：`GET /health` 返回 `{"status": "ok", "database": "ok"}`；数据库不可达时返回 503 `{"status": "degraded", "database": "unreachable"}`，供负载均衡 / 容器探活使用。
- **巡检兜底**：启动时把上次进程残留的 `running` 巡检日志标记为 `failed`（`mark_stale_patrols_failed`）；单次巡检受 `PATROL_TIMEOUT`（默认 600s）约束，同类型巡检互斥（已在运行时直接跳过）。

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
