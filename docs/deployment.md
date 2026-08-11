# Deployment

## Container

`docker compose up -d` 一键启动完整栈（PostgreSQL + pgvector + EMA 后端 + 前端 + 备份 + Prometheus/Grafana 监控）：

```yaml
services:
  postgres:      # PostgreSQL + pgvector（含健康检查）
  backend:       # EMA 单镜像（后端 + 构建产物 frontend/dist，见 Dockerfile）
  backup:        # 每小时 pg_dump 全库到 ./backups（见下文 Backup & Restore）
  prometheus:    # 抓取 backend /metrics 的 ema_* 时序（见下文 Monitoring）
  grafana:       # 渲染 EMA 运行健康看板（登录 admin / GF_ADMIN_PASSWORD）
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
| backup | pgvector/pgvector:pg16（复用 pg_dump） | 每小时 pg_dump -Fc 全库，保留最近 14 份 |
| prometheus | prom/prometheus:v2.53.0 | 抓取 backend `/metrics`，15s 间隔 |
| grafana | grafana/grafana:11.1.0 | EMA 运行健康看板（`http://localhost:3000`） |
| caddy | caddy:2.8-alpine | TLS 终止 + 反代（默认 `http://localhost:8080`，公网配域名后自动 HTTPS） |

> **pgvector 版本要求**：向量索引使用 HNSW（`USING hnsw`），需要 **pgvector ≥ 0.5**。上述 `pgvector/pgvector:pg16` 镜像满足要求；若使用发行版自带扩展或自建镜像，请先确认 `SELECT extversion FROM pg_extension WHERE extname = 'vector'` 不低于 0.5，否则建索引会失败。历史 ivfflat 库开发期重建即可（`init_db` 不做运行时索引替换）。

## Schema Migration (Alembic)

Schema 演进用 **Alembic 版本化**（`alembic_version` 表 + `migrations/versions/` 下成对的 upgrade/downgrade 迁移）：

- **统一入口**：`backend/db/schema.py` 的 `init_db()`（FastAPI 启动 lifespan / 测试 / eval 脚本都走它）先跑 `alembic upgrade head`（新库全量建表，旧库跳过已存在表并 stamp 版本），再**校验** embedding 列维度与配置一致——不一致启动失败，需 `python -m scripts.recreate_db` 重建（维度依赖配置、不进迁移，但绝不自动清空）。`APP_ENV=test` 下 pytest 每次会话 `DROP SCHEMA public CASCADE` 后经 `init_db()` 重建，行为不变。
- **新增列 / 表 / 索引**：写一个新迁移而不是改 `init_db`：
  ```bash
  alembic revision -m "add something"
  # 手写 upgrade() / downgrade()（项目 schema 是 raw SQL，无 ORM metadata，
  # 不用 autogenerate——`op.execute("ALTER TABLE ... ADD COLUMN ...")`）
  alembic upgrade head   # 本地验证
  ```
  CI（`.github/workflows/ci.yml`）跑 `test_migrations.py`，它会用临时库验证 `upgrade head` / `downgrade base` / 幂等，以及 `init_db` 对维度不匹配的拒绝。
- **配置**：`alembic.ini` 不含数据库 URL——`migrations/env.py` 从 app 自己的 `DATABASE_URL` 读（单一来源），并用同步 psycopg 驱动跑迁移（async URL 自动转 `postgresql+psycopg://`）。
- **回滚**：`alembic downgrade -1`（回退一个迁移）或 `alembic downgrade base`（清空所有业务表）。
- **首次引入的兼容性**（2026-08-11 已实测）：旧库（由历史 `init_db` 建成）首次 `upgrade head` 时 `CREATE TABLE IF NOT EXISTS` 跳过已有表、仅 stamp 到 head；全新库全量建 9 张业务表。embedding 列用占位维度 1024；若配置模型维度不同，`init_db` 启动校验失败，需 `python -m scripts.recreate_db` 重建后重摄取。

## Backup & Restore

> **已实测**（2026-08-11）：备份容器启动即 dump 一次；恢复演练通过——pg_restore 到新库后 `memories` / `chunks` 计数与主库完全一致。

### 备份策略

compose 的 `backup` 容器（复用 `pgvector/pgvector:pg16` 镜像自带的 pg_dump）执行 [scripts/backup.sh](../scripts/backup.sh)：**启动时 dump 一次，之后每 `BACKUP_INTERVAL`（默认 3600s）一次**，全库导出为 pg_dump **custom 格式**（压缩、可 `pg_restore` 选择性恢复），写入宿主 `./backups/`（已 gitignore），**滚动保留最近 `BACKUP_KEEP`（默认 14）份**。容器内脚本用 `sed 's/\r$//'` 去除可能的 CRLF，兼容 Windows checkout。

- 备份失败只写日志、进入下一个间隔重试，不阻塞其他服务；磁盘写满/权限错误通过容器日志暴露（`docker compose logs -f backup`）。
- 立即手动备份一次：`docker compose exec backup sh -c "sh /backup.sh"`（脚本内含启动即 dump 逻辑，也可直接 `docker compose exec -T backup pg_dump -h postgres -U ema -d ema_dev -Fc -f /backups/manual_$(date +%Y%m%d_%H%M%S).dump`）。
- 调间隔/保留数：`.env` 设 `BACKUP_INTERVAL` / `BACKUP_KEEP` 后 `docker compose up -d backup`。
- 备份与主库走同一 compose 网络，不占用宿主端口；若需把备份落异地，把 `./backups` 换成挂载的 NAS/对象存储目录或加一层 rsync 即可。

> **局限**：这是逻辑备份（pg_dump），非物理备份 + WAL 归档——RPO 为一个备份间隔（默认 1 小时），一次 pg_dump 期间的数据变更可能不在上一个 dump 内。对开发/演示场景足够；生产高可用场景需在此基础上加 WAL 归档（`archive_mode=on` + 连续归档）或托管库快照。

### 恢复 runbook

```bash
# 1. 找到要恢复的备份（最新的在前）
ls -1t backups/ema_*.dump | head -1
#    backups/ema_20260811_015147.dump

# 2. （可选）先确认备份可读、看内容清单
docker compose exec -T backup pg_restore -l /backups/<文件名>.dump | head -20

# 3. 恢复到一个新库，保留原库用于对比/回滚（推荐）：
docker compose exec -T backup psql -h postgres -U ema -d postgres \
  -c "create database ema_restore;"
docker compose exec -T backup pg_restore -h postgres -U ema \
  -d ema_restore --no-owner /backups/<文件名>.dump

#   或就地覆盖现有库（先停 backend 防止写冲突）：
#   docker compose stop backend
#   docker compose exec -T backup pg_restore -h postgres -U ema \
#     -d ema_dev --clean --if-exists --no-owner /backups/<文件名>.dump
#   docker compose start backend

# 4. 校验行数与主库/预期一致
docker compose exec -T backup psql -h postgres -U ema -d ema_restore \
  -c "select count(*) from memories; select count(*) from chunks;"

# 5. 校验通过后，把新库指为实际使用的库，再重启 backend：
#    a) 在 .env 设 DATABASE_URL 指向新库（compose 网络内用 postgres 服务名，
#       如 postgresql://ema:<口令>@postgres:5432/ema_restore）后
#       docker compose up -d backend；或
#    b) 直接 drop 原库后把新库 rename 回 ema_dev（数据库侧操作，无需改配置）
#       docker compose exec -T backup psql -h postgres -U ema -d postgres \
#         -c "drop database if exists ema_dev; alter database ema_restore rename to ema_dev;"
#       docker compose restart backend
```

> **Windows 注意**：在 Git Bash 下执行容器内路径参数（如 `/backups/xxx.dump`）需先 `MSYS_NO_PATHCONV=1`（或 `MSYS2_ARG_CONV_EXCL='*'`），否则 Git Bash 会把 POSIX 路径改写成 Windows 路径导致文件找不到。示例：`MSYS_NO_PATHCONV=1 docker compose exec -T backup pg_restore ... /backups/ema_xxx.dump`。

### 监控（Prometheus + Grafana）

`backend` 已暴露 `GET /metrics`（`ema_*` 时序，见 `backend/shared/runtime_metrics.py`）；compose 的 `prometheus` 服务每 15s 抓取一次，`grafana` 服务渲染 **"EMA — Runtime Health"** 看板（9 个面板：HTTP QPS / 延迟 P50·P95 / 5xx 错误率 / LLM 调用·token·延迟 / 熔断器状态 / Agent 并发槽位与 503 拒绝 / ReAct 步数均值）。

- 访问：Grafana `http://localhost:3000`（登录 **admin** / `GF_ADMIN_PASSWORD`，默认 `admin`，首次登录请修改）；Prometheus `http://localhost:9090`（query 界面可手查 `ema_*`）。
- 配置位置：抓取配置 `docker/prometheus.yml`；Grafana datasource / dashboard 载入 `docker/grafana/provisioning/`，看板 JSON `docker/grafana/dashboards/ema.json`（改后 `docker compose restart grafana` 生效）。
- 数据持久化：`prometheus_data` / `grafana_data` 为 named volume，重启不丢；Grafana 的 datasource 通过固定 uid `prometheus` 与看板绑定。
- **告警边界**：当前无进程内 LLM 健康告警——`backend/service/alerts.py` 的进程内告警循环已移除。其四个 check 中，结构化失败增长迁移为 Prometheus 的 `ema_structured_failures_total`（按 scenario 计数，见 `backend/shared/runtime_metrics.py`）；错误率、重试抖动、熔断器状态三个信号本就在 `runtime_metrics.py` 的同一咽喉点以时序暴露（LLM 调用 success/error 计数与延迟直方图、熔断器 open/closed gauge 与拒绝计数），并配结构化 WARNING 日志。原循环默认仅写日志、飞书推送 opt-in（默认关），与观测栈信号重叠且无独有覆盖，故删除以避免并行代码路径漂移。Prometheus/Grafana 目前负责**观测与可视化**，尚未配置 Prometheus alert rules / Alertmanager——基础设施级告警（服务不可达、错误率突增）是下一步，不是现状。

## HTTPS (TLS) 与反向代理

compose 的 `caddy` 服务在 backend 前面做 TLS 终止 + 反向代理。配置只有两个变量：

- `CADDY_SITE_ADDRESS` — 站点地址。**默认 `http://localhost:8080`**（本地纯 HTTP 测试，零证书配置）；**公网部署设为你的域名**（如 `https://ema.example.com`），Caddy 自动申请 Let's Encrypt 证书、托管 443/80、HTTP→HTTPS 重定向。
- `ACME_EMAIL` — 证书到期通知邮箱（可选，公网部署建议填）。

```bash
# 本地测试（默认）：代理入口 http://localhost:8080
docker compose up -d caddy
curl http://localhost:8080/health

# 公网部署：.env 设 CADDY_SITE_ADDRESS=https://ema.example.com 后重启
docker compose up -d caddy   # 自动获取证书，https://ema.example.com 生效
```

- **前端无需改动**：SPA 的 API 调用全部走相对路径（`frontend/src/api/client.ts` 的 `BASE_URL=''`），经代理访问时同源转发，无 CORS / 基址问题；FastAPI 自托管 SPA + API（单镜像），代理层只需整站转发 `backend:8000`。
- **已实测**（2026-08-11）：默认配置下 `/health` 经代理返回 200，`/api/connectors` 无认证返回 401（认证守卫正常透传），`/api/agent/chat/stream` 的 SSE 流式响应正常转发（Caddy 对流式响应无缓冲）。
- **安全说明**：代理只接收 `CADDY_SITE_ADDRESS` / `ACME_EMAIL` 两个变量，不注入整个 `.env`——API key 等敏感值不会进入边缘容器。backend 的 `8000:8000` 宿主映射保留用于本地直接访问；公网环境建议只暴露代理端口（80/443），在 `.env` 层面或防火墙收紧 8000。
- 需更复杂路由（按路径分流、WAF、限速）时，把 `docker/caddy/Caddyfile` 的 site 块替换为对应指令即可，无需改应用。

## 日志（结构化）

EMA 日志统一走 **stdout**（容器 `docker compose logs`），格式由 `LOG_FORMAT` 控制（默认 `text`，人类可读；设 `json` 输出结构化 JSON）：

```json
{"ts":"2026-08-11T02:11:03.042+00:00","level":"WARNING","logger":"backend.api.routes.agent_routes","msg":"agent_chat refused — concurrency cap reached (max=4)","trace_id":"…","thread_id":"…","func":"agent_chat","line":412}
```

- **字段**：`ts`（UTC ISO-8601，可被时间索引直接解析）、`level`、`logger`、`msg`、`func` / `line`（调用点）、`trace_id` / `thread_id`（同一 Agent 运行的所有日志行共享一个 id，可整段回放）、`exc`（异常堆栈，仅 error 行）。
- **采集**：stdout 即所有收集器（Loki / ELK / vector）的标准输入；单行 JSON 无需 grok 正则即可索引，`docker compose logs -f backend | grep '"trace_id":"…"'` 即可跟一条 Agent 运行。当前 compose 未内置日志收集器（保持精简）——集中化平台由部署侧承担，`LOG_FORMAT=json` 已就绪。
- **实现**：`backend/shared/logging_config.py`（`setup_logging()` 在 `main.py` lifespan 调用，幂等替换根 handler）；`trace_id` / `thread_id` 复用 usage 追踪的同一 contextvar（`backend/shared/config.py`）。
- 测试与开发不受影响：pytest 用自己的 `caplog` handler，`LOG_FORMAT` 只在应用启动时读取。

## Configuration

通过 `.env` 文件管理环境变量：

- `LLM_*` — LLM provider 配置（含传输韧性 `LLM_RETRY_*` / `LLM_CIRCUIT_BREAKER_*`、结构化重试 `LLM_STRUCTURED_*`、可选故障转移 `LLM_FALLBACK_*`——设置 `LLM_FALLBACK_PROVIDER` 后，主 provider 的可重试失败或熔断会在该次调用上改走备用 provider，留空则关闭）
- `EMBEDDING_*` — Embedding 模型配置（含可选故障转移 `EMBEDDING_FALLBACK_*`——设置 `EMBEDDING_FALLBACK_PROVIDER` 后，主 provider 失败（重试耗尽/熔断/本地模型损坏）会在该次调用上改走备用 provider，留空则关闭；备用模型维度必须与主模型一致；另含 CPU 并发控制 `EMBEDDING_MAX_CONCURRENCY` / `EMBEDDING_TORCH_THREADS`——两者乘积≈核数时零超卖，消除并发 embed 延迟长尾，冷路径压测 10 并发 P95 19s→690ms，见 gap-remediation.md §5.3.2）
- `EMA_API_KEY` — API 接入认证 key。设置后所有 `/api` 请求须携带 `Authorization: Bearer <EMA_API_KEY>`（见下文 Authentication）；不设置时服务仍可启动，但任何 `/api` 请求都会收到 401
- `VITE_EMA_API_KEY` — 前端注入的同一 key（构建期替换 `import.meta.env.VITE_EMA_API_KEY`）；未配置时前端请求不携带认证头（向后兼容）。应与 `EMA_API_KEY` 保持一致

  > **容器构建注意**：Vite 在 **build 阶段**（`docker build` Stage 1）读取该变量，而根 `.env` 被 `.dockerignore` 排除、`env_file` 只注入容器**运行时**，Vite 看不到。docker-compose 已通过 `build.args.VITE_EMA_API_KEY` 从根 `.env` 传入，直接 `docker compose build` 即可；修改 key 后须重新构建镜像，否则浏览器端仍携带旧 key 会收到 401。构建参数会留在镜像元数据（`docker history`）中，因该 key 本就是分发给前端的公开值，可接受；真正的服务端密钥（`EMA_API_KEY`）不要走 build-arg。
- `DATABASE_URL` — PostgreSQL 连接串。本地直接运行用 localhost；compose 部署时默认由 `docker-compose.yml` 指向 `postgres` 服务名，此处设置会覆盖该默认值（改库/改口令见上方"改数据库口令"与恢复 runbook）
- `MAX_AGENT_STEPS` — Agent 最大工具调用次数
- `MAX_AGENT_CONCURRENCY` — 同时运行的交互式 Agent 会话上限（默认 4；超过的 chat 请求返回 503，防止并发 ReAct 循环一起打满 provider 限流）
- `AGENT_TIMEOUT` — Agent 单回合总超时（秒）
- `PATROL_*` — 巡检调度与超时（`PATROL_ENABLED` / `PATROL_DAILY_HOUR` / `PATROL_WEEKLY_*` / `PATROL_TIMEOUT`）
- `USAGE_*` — LLM 用量追踪（`USAGE_ENABLED` / `USAGE_FLUSH_INTERVAL_SECONDS` / `USAGE_BUFFER_MAX` / `USAGE_SAMPLE_RATE`——采样率决定多少成功调用把 prompt/response 文本存入 `llm_usage` 供事后质量分析，error 调用一律采样；`/api/usage/samples` 查询。`USAGE_SAMPLE_RETENTION_DAYS`——采样文本保留天数，到期由 flusher 清空文本列、元数据保留，默认 30）
- `METRICS_ENABLED` — 运行健康指标（Prometheus，默认 true）。开启后 `GET /metrics` 暴露进程内时间序列：HTTP 请求数与延迟分位数、LLM 调用数/延迟/token、熔断器状态与打开/拒绝计数、Agent 并发槽位占用与 503 拒绝、ReAct 循环步数分布。与 `USAGE_ENABLED` 独立——这是进程本地健康观测，`llm_usage` 是持久化成本行。关闭则 `/metrics` 返回 404
- `RATE_LIMIT_*` — API 限流（per-key 令牌桶，默认开启）：`RATE_LIMIT_ENABLED` / `RATE_LIMIT_CHAT_REQUESTS`（默认 30）/ `RATE_LIMIT_CHAT_WINDOW_SECONDS`（60，覆盖 `/api/agent/chat*` 与 `/api/scenarios*` 的 LLM 密集档）/ `RATE_LIMIT_GENERAL_REQUESTS`（120）/ `RATE_LIMIT_GENERAL_WINDOW_SECONDS`（60，其余 `/api` 路由）。按 `Authorization: Bearer` 的 token 分桶（无 token 归共享 `anonymous` 桶），超限返回 `429 + Retry-After`。实现见 `backend/api/ratelimit.py`；`APP_ENV=test` 豁免（与 auth 一致），`/health`、`/metrics`、静态资源不受限
- `LOG_LEVEL` — 日志级别（DEBUG / INFO / WARNING / ERROR）
- `LOG_FORMAT` — 日志格式（`text` 默认人类可读；`json` 输出单行结构化 JSON，含 UTC `ts` / `trace_id` / `thread_id`，可直接被 Loki/ELK 摄取，见上文"日志（结构化）"）
- `CADDY_SITE_ADDRESS` / `ACME_EMAIL` — 反代与 TLS 配置（见上文 "HTTPS (TLS) 与反向代理"；注意这些变量只注入 caddy 容器，不进入 backend）
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
- **巡检可靠性**：巡检是全库扫描，扫描步数与输出空间独立于交互式配置——`daily` 用 15 步、`weekly` 用 20 步（交互式 `MAX_AGENT_STEPS` 默认 5 会把巡检在扫描中途强停）；最终报告合成输出上限为 `PATROL_MAX_TOKENS`（默认 8000，独立于交互式 `LLM_MAX_TOKENS`），且 daily/weekly prompt 限制各类条目数，保证完整 JSON 能单次输出。输出不符合 JSON 契约（缺键或非 JSON）时巡检记 `failed` 并保留原始输出供排查，不重写重试。

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

- `.env` 通过 `env_file` 注入；compose 的 backend `DATABASE_URL` 默认指向 `postgres` 服务名（`docker-compose.yml` 内 `${DATABASE_URL:-…}`），`.env` 中设置的 `DATABASE_URL` 会覆盖该默认值——`.env` 里的 localhost 是本地直接运行用的，容器网络内不可用，部署时若设此变量须指向 `postgres` 服务名，或保持不设。
- **改数据库口令**：生产环境请修改默认口令 `ema123`。在 `.env` 设 `POSTGRES_PASSWORD`（postgres 容器与 backup 容器共用，单一来源）后 `docker compose up -d postgres backup` 重建；backend 的 `DATABASE_URL` 若设了连接串，其中内嵌的口令也要同步改（compose 不支持在一个变量内引用另一个，故 DATABASE_URL 需整体维护）。`ema123` 是默认值，暴露 5432 端口到公网时务必修改。
- Embedding 模型经 `./docker/models` 挂载进容器，运行期完全离线（`HF_HUB_OFFLINE=1`），见上文"Embedding 模型挂载"。
- **Linux 容器顺带解决了 Windows 平台的 checkpointer 限制**：`AsyncPostgresSaver`（psycopg 异步）在 Linux 原生可用，对话 checkpoint 落 PostgreSQL；Windows 下降级为 `InMemorySaver` 仅是开发期兼容（见 `backend/main.py`）。注意 psycopg 的纯 Python 实现依赖系统 `libpq` 运行库，`python:*-slim` 基础镜像没有它——Dockerfile 已 `apt-get install libpq5`，缺失时 `AsyncPostgresSaver` 会静默降级为 `InMemorySaver`（checkpoint 不持久）。

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
| LLM / Embedding 熔断器（开关状态、冷却窗口） | `backend/shared/resilience.py` | 每个副本各自计数熔断、冷却窗口互相打散，一个实例熔断不保护其他实例；/health 只反映本副本状态 |
| auto-memory 节流（per-thread 间隔/上限、进程级滚动窗口） | `backend/agent/nodes.py` | 每个副本独立计数，实际写入频率放大到上限的 N 倍 |
| LLM usage 缓冲（`_pending`，由后台 flusher 批量入库） | `backend/service/usage.py` | 缓冲只记录本副本的调用；写入 `llm_usage` 的行不重复，但 `USAGE_BUFFER_MAX` 兜底语义按副本独立 |
| API 限流令牌桶（per-key 的 `_buckets`） | `backend/api/ratelimit.py` | 每个副本各自计数——一个实例被限流不保护其他副本，实际允许的请求量放大到单副本限额的 N 倍 |
| 会话压缩摘要缓存 / in-flight 去重 | `backend/agent/nodes.py` | 跨副本完全失效，每副本各自付一次压缩 LLM 调用（结果一致，仅浪费 token） |
| 对话 checkpoint（`AsyncPostgresSaver` 连接池） | `backend/service/agent_service.py` | 唯一已持久化的状态；`checkpoints` 表在 PG 中，多副本下对话历史仍一致，但同一 `thread_id` 的并发续写由哪副本处理不可控 |

巡检互斥（`patrol_logs` 中 `status='running'` 的查重）在数据库层，多副本下仍有效；其余状态均受上述约束。

**改造方向**（当需要横向扩展时）：将熔断计数与 auto-memory 节流计数迁到共享存储（PostgreSQL 表或 Redis），usage 缓冲保持内存态但接受 `llm_usage` 行由多副本各自 flush。在此之前，扩容请垂直扩容（加大单实例资源）而非加副本。
