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

> **pgvector 版本要求**：向量索引使用 HNSW（`USING hnsw`），需要 **pgvector ≥ 0.5**。上述 `pgvector/pgvector:pg16` 镜像满足要求；若使用发行版自带扩展或自建镜像，请先确认 `SELECT extversion FROM pg_extension WHERE extname = 'vector'` 不低于 0.5，否则 `init_db()` 建索引会失败（旧库的 ivfflat 索引会在启动迁移中被替换为 HNSW）。

## Configuration

通过 `.env` 文件管理环境变量：

- `LLM_*` — LLM provider 配置（含传输韧性 `LLM_RETRY_*` / `LLM_CIRCUIT_BREAKER_*`、结构化重试 `LLM_STRUCTURED_*`、可选故障转移 `LLM_FALLBACK_*`——设置 `LLM_FALLBACK_PROVIDER` 后，主 provider 的可重试失败或熔断会在该次调用上改走备用 provider，留空则关闭）
- `EMBEDDING_*` — Embedding 模型配置（含可选故障转移 `EMBEDDING_FALLBACK_*`——设置 `EMBEDDING_FALLBACK_PROVIDER` 后，主 provider 失败（重试耗尽/熔断/本地模型损坏）会在该次调用上改走备用 provider，留空则关闭；备用模型维度必须与主模型一致）
- `DATABASE_URL` — PostgreSQL 连接
- `MAX_AGENT_STEPS` — Agent 最大工具调用次数
- `MAX_AGENT_CONCURRENCY` — 同时运行的交互式 Agent 会话上限（默认 4；超过的 chat 请求返回 503，防止并发 ReAct 循环一起打满 provider 限流）
- `AGENT_TIMEOUT` — Agent 单回合总超时（秒）
- `PATROL_*` — 巡检调度与超时（`PATROL_ENABLED` / `PATROL_DAILY_HOUR` / `PATROL_WEEKLY_*` / `PATROL_TIMEOUT`）
- `USAGE_*` — LLM 用量追踪（`USAGE_ENABLED` / `USAGE_FLUSH_INTERVAL_SECONDS` / `USAGE_BUFFER_MAX` / `USAGE_SAMPLE_RATE`——采样率决定多少成功调用把 prompt/response 文本存入 `llm_usage` 供事后质量分析，error 调用一律采样；`/api/usage/samples` 查询。`USAGE_SAMPLE_RETENTION_DAYS`——采样文本保留天数，到期由 flusher 清空文本列、元数据保留，默认 30）
- `ALERT_*` — LLM 健康告警（`ALERTS_ENABLED` / `ALERT_ERROR_RATE_THRESHOLD` / `ALERT_CHECK_INTERVAL_SECONDS` / `ALERT_FEISHU_ENABLED`——错误率/结构化失败/熔断超阈值写日志；飞书推送需显式开启）
- `LOG_LEVEL` — 日志级别（DEBUG / INFO / WARNING / ERROR）
- `APP_ENV` — 运行环境 (development / test / production)

## Health & Startup

- **启动配置校验**：启动时运行 `validate_config()`，检查巡检时间范围（`PATROL_DAILY_HOUR` 等 0-23 / 0-6）、重试与熔断参数边界、`LLM_API_KEY` 非空（`APP_ENV=test` 豁免）。配置非法直接拒绝启动，而非等首个请求才报错或静默产生无效调度。
- **健康检查**：`GET /health` 返回 `{"status", "database", "llm", "embedding"}`——除数据库连通性外，廉价上报 LLM/Embedding provider 的配置状态与熔断器开关（`circuit: closed/open/n/a`，不做真实 LLM 调用，避免探活烧 token）；数据库不可达时返回 503 `{"status": "degraded", "database": "unreachable"}`，供负载均衡 / 容器探活使用。
- **巡检兜底**：启动时把上次进程残留的 `running` 巡检日志标记为 `failed`（`mark_stale_patrols_failed`）；单次巡检受 `PATROL_TIMEOUT`（默认 600s）约束，同类型巡检互斥（已在运行时直接跳过）。此外，若某类巡检已有计划历史、但最近一个计划槽（daily/hour 或 weekly/day+hour）内没有运行记录——即进程在计划时刻停机——启动时补跑一次（`trigger=cron_catchup`），避免重启静默丢一轮巡检；全新部署（无历史）不触发补跑。

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
