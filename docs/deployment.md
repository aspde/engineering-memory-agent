# Deployment

## Container

`docker compose up -d` 一键启动完整栈（PostgreSQL + pgvector + EMA 后端 + 前端）：

```yaml
services:
  postgres:      # PostgreSQL + pgvector（含健康检查）
  backend:       # EMA 单镜像（后端 + 构建产物 frontend/dist，见 Dockerfile）
```

`backend` 使用根目录 `Dockerfile` 构建**单镜像**——FastAPI 同时托管前端构建产物（`backend/main.py` 挂载 `frontend/dist` 并对非 API 路由回退到 index.html），因此不需要单独的 frontend 服务。构建流程：

- **Stage 1**（`node:22-alpine`）：`npm ci && npm run build` 产出 SPA
- **Stage 2**（`python:3.12-slim`）：装依赖 + 拷贝前端产物（镜像不含 embedding 模型，见下文"Embedding 模型挂载"）

**torch 分步安装**：`torch==2.13.0+cpu` 只发布在 PyTorch 官方 CPU index（`+cpu` 后缀在 PyPI 不存在），因此 Dockerfile 先用 `--no-deps --index-url https://download.pytorch.org/whl/cpu` 单独装 torch，再用纯 PyPI 装其余依赖——不能合并为一个 `--extra-index-url`，否则 PyTorch index 的旧包快照会污染 charset-normalizer 等共享包的解析。

## Embedding 模型挂载

BGE-M3 等 embedding 模型**不烘焙进镜像**，而是放在宿主机、通过 **bind mount** 挂载进容器（不复制数据，宿主机与容器共享同一份文件）：

```yaml
services:
  backend:
    volumes:
      - ${HF_CACHE_DIR:-./docker/models}:/home/ema/.cache/huggingface
```

- 容器运行时 `HOME=/home/ema`（uid 10001），`backend/service/embedding_service.py` 用 `local_files_only=True` 加载，直接命中该挂载的 HF 缓存；运行期仍完全离线（`HF_HUB_OFFLINE=1`）。
- **推荐：复用已有 HF 缓存**。宿主机已有 `~/.cache/huggingface`（含目标模型）时，`.env` 中设 `HF_CACHE_DIR` 指向它即可零复制：
  ```
  HF_CACHE_DIR=C:/Users/<you>/.cache/huggingface
  ```
  挂载源目录结构须为 HF hub 缓存：`hub/models--BAAI--bge-m3/...`（`.env` 不在版本库，此变量仅本机生效）。
- **默认（新部署）**：不设 `HF_CACHE_DIR` 时挂载项目内 `./docker/models/`（相对 compose 文件解析），用已构建的镜像一次性下载到该目录（示例）：
  ```bash
  docker run --rm -u root -v "$(pwd)/docker/models:/models" \
    -e HF_HOME=/models -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
    ema-backend python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('BAAI/bge-m3')"
  ```
  `docker/models/` 已加入 `.gitignore` 与 `.dockerignore`，模型不进版本库、不进构建上下文。
- 换模型只需改 `.env` 的 `EMBEDDING_MODEL` 并准备对应缓存，无需重建镜像。

## Services

| Service | Image | Status |
|---------|-------|--------|
| postgres | pgvector/pgvector:pg16 | 已就绪 |
| backend | `Dockerfile` 构建（后端 + 前端产物单镜像） | 已容器化，`docker compose up -d` 一键启动 |

> **pgvector 版本要求**：向量索引使用 HNSW（`USING hnsw`），需要 **pgvector ≥ 0.5**。上述 `pgvector/pgvector:pg16` 镜像满足要求；若使用发行版自带扩展或自建镜像，请先确认 `SELECT extversion FROM pg_extension WHERE extname = 'vector'` 不低于 0.5，否则 `init_db()` 建索引会失败（旧库的 ivfflat 索引会在启动迁移中被替换为 HNSW）。

## Configuration

通过 `.env` 文件管理环境变量：

- `LLM_*` — LLM provider 配置（含传输韧性 `LLM_RETRY_*` / `LLM_CIRCUIT_BREAKER_*`、结构化重试 `LLM_STRUCTURED_*`、可选故障转移 `LLM_FALLBACK_*`——设置 `LLM_FALLBACK_PROVIDER` 后，主 provider 的可重试失败或熔断会在该次调用上改走备用 provider，留空则关闭）
- `EMBEDDING_*` — Embedding 模型配置（含可选故障转移 `EMBEDDING_FALLBACK_*`——设置 `EMBEDDING_FALLBACK_PROVIDER` 后，主 provider 失败（重试耗尽/熔断/本地模型损坏）会在该次调用上改走备用 provider，留空则关闭；备用模型维度必须与主模型一致）
- `EMA_API_KEY` — API 接入认证 key。设置后所有 `/api` 请求须携带 `Authorization: Bearer <EMA_API_KEY>`（见下文 Authentication）；不设置时服务仍可启动，但任何 `/api` 请求都会收到 401
- `VITE_EMA_API_KEY` — 前端注入的同一 key（构建期替换 `import.meta.env.VITE_EMA_API_KEY`）；未配置时前端请求不携带认证头（向后兼容）。应与 `EMA_API_KEY` 保持一致

  > **容器构建注意**：Vite 在 **build 阶段**（`docker build` Stage 1）读取该变量，而根 `.env` 被 `.dockerignore` 排除、`env_file` 只注入容器**运行时**，Vite 看不到。docker-compose 已通过 `build.args.VITE_EMA_API_KEY` 从根 `.env` 传入，直接 `docker compose build` 即可；修改 key 后须重新构建镜像，否则浏览器端仍携带旧 key 会收到 401。构建参数会留在镜像元数据（`docker history`）中，因该 key 本就是分发给前端的公开值，可接受；真正的服务端密钥（`EMA_API_KEY`）不要走 build-arg。
- `DATABASE_URL` — PostgreSQL 连接
- `MAX_AGENT_STEPS` — Agent 最大工具调用次数
- `MAX_AGENT_CONCURRENCY` — 同时运行的交互式 Agent 会话上限（默认 4；超过的 chat 请求返回 503，防止并发 ReAct 循环一起打满 provider 限流）
- `AGENT_TIMEOUT` — Agent 单回合总超时（秒）
- `PATROL_*` — 巡检调度与超时（`PATROL_ENABLED` / `PATROL_DAILY_HOUR` / `PATROL_WEEKLY_*` / `PATROL_TIMEOUT`）
- `USAGE_*` — LLM 用量追踪（`USAGE_ENABLED` / `USAGE_FLUSH_INTERVAL_SECONDS` / `USAGE_BUFFER_MAX` / `USAGE_SAMPLE_RATE`——采样率决定多少成功调用把 prompt/response 文本存入 `llm_usage` 供事后质量分析，error 调用一律采样；`/api/usage/samples` 查询。`USAGE_SAMPLE_RETENTION_DAYS`——采样文本保留天数，到期由 flusher 清空文本列、元数据保留，默认 30）
- `ALERT_*` — LLM 健康告警（`ALERTS_ENABLED` / `ALERT_ERROR_RATE_THRESHOLD` / `ALERT_CHECK_INTERVAL_SECONDS` / `ALERT_FEISHU_ENABLED`——错误率/结构化失败/熔断超阈值写日志；飞书推送需显式开启）
- `METRICS_ENABLED` — 运行健康指标（Prometheus，默认 true）。开启后 `GET /metrics` 暴露进程内时间序列：HTTP 请求数与延迟分位数、LLM 调用数/延迟/token、熔断器状态与打开/拒绝计数、Agent 并发槽位占用与 503 拒绝、ReAct 循环步数分布。与 `USAGE_ENABLED` 独立——这是进程本地健康观测，`llm_usage` 是持久化成本行。关闭则 `/metrics` 返回 404
- `LOG_LEVEL` — 日志级别（DEBUG / INFO / WARNING / ERROR）
- `APP_ENV` — 运行环境 (development / test / production)

## Authentication

API 接入认证为共享 key（`Authorization: Bearer <EMA_API_KEY>`），实现见 `backend/api/auth.py`，作为全局依赖覆盖所有 `/api` 路由：

```bash
# 服务端：设置 key 后启动（未设置时所有 /api 请求返回 401）
EMA_API_KEY=<your-key> uvicorn backend.main:app

# 客户端：请求头携带同一 key
curl -H "Authorization: Bearer <your-key>" http://localhost:8000/api/memory/stats
```

- `APP_ENV=test` 豁免该守卫（API 测试套件用 mock，不需 key）。
- key 比较为常量时间（`secrets.compare_digest`），不匹配返回无细节的通用 401。
- 前端构建期用 `VITE_EMA_API_KEY` 注入同一 key；开发模式未配置时前端请求不携带认证头（后端返回 401）。
- 边界：单 key 共享、无用户身份与角色——这是有意取舍（与 ADR 004 不做多租户一致），非生产多用户场景的完整认证方案。

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

### 方式 A（推荐）：Docker Compose

```bash
# 1. 构建镜像（后端 + 前端产物单镜像，约 10-30 分钟，视网络）
docker compose build

# 2. 启动完整栈（postgres + backend，backend 依赖 postgres 健康检查通过）
docker compose up -d

# 3. 访问
#    API + SPA:  http://localhost:8000
#    OpenAPI:    http://localhost:8000/docs

# 4. 查看日志
docker compose logs -f backend
```

- `.env` 通过 `env_file` 注入；compose 内 `DATABASE_URL` 覆盖指向 `postgres` 服务名（`.env` 里的 localhost 不适用容器网络）。
- Embedding 模型经 `./docker/models` 挂载进容器，运行期完全离线（`HF_HUB_OFFLINE=1`），见上文"Embedding 模型挂载"。
- **Linux 容器顺带解决了 Windows 平台的 checkpointer 限制**：`AsyncPostgresSaver`（psycopg 异步）在 Linux 原生可用，对话 checkpoint 落 PostgreSQL；Windows 下降级为 `InMemorySaver` 仅是开发期兼容（见 `backend/main.py`）。

### 方式 B：裸进程（开发 / 调试）

Backend 在生产模式下直接托管前端构建产物（`frontend/dist/`），无需单独运行前端开发服务器：

```bash
# 1. Build frontend
cd frontend && npm run build

# 2. Start backend (serves both API and SPA)
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## 单实例部署约束

**当前版本只支持单进程 / 单副本部署，不支持 `uvicorn --workers N` 或多实例副本。** 以下状态全部保存在进程内，不跨实例共享：

| 状态 | 位置 | 多副本后果 |
|------|------|-----------|
| LLM / Embedding 熔断器（开关状态、半开探测） | `backend/shared/resilience.py` | 每个副本各自计数熔断、半开探测互相打散，一个实例熔断不保护其他实例；/health 只反映本副本状态 |
| auto-memory 节流（per-thread 间隔/上限、进程级滚动窗口） | `backend/agent/nodes.py` | 每个副本独立计数，实际写入频率放大到上限的 N 倍 |
| LLM usage 缓冲（`_pending`，由后台 flusher 批量入库） | `backend/service/usage.py` | 缓冲只记录本副本的调用；写入 `llm_usage` 的行不重复，但 `USAGE_BUFFER_MAX` 兜底语义按副本独立 |
| 会话压缩摘要缓存 / in-flight 去重 | `backend/agent/nodes.py` | 跨副本完全失效，每副本各自付一次压缩 LLM 调用（结果一致，仅浪费 token） |
| 对话 checkpoint（`AsyncPostgresSaver` 连接池） | `backend/service/agent_service.py` | 唯一已持久化的状态；`checkpoints` 表在 PG 中，多副本下对话历史仍一致，但同一 `thread_id` 的并发续写由哪副本处理不可控 |

巡检互斥（`patrol_logs` 中 `status='running'` 的查重）在数据库层，多副本下仍有效；其余状态均受上述约束。

**改造方向**（当需要横向扩展时）：将熔断计数与 auto-memory 节流计数迁到共享存储（PostgreSQL 表或 Redis），usage 缓冲保持内存态但接受 `llm_usage` 行由多副本各自 flush。在此之前，扩容请垂直扩容（加大单实例资源）而非加副本。
