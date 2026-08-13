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
| Backend | FastAPI + Python 3.12 | 34 个 `/api` 路由 + `/health`/`/metrics` 等应用级端点已实现（含 SSE 流式 + HITL） |
| Agent | LangGraph (手动 StateGraph) | ReAct 循环已实现 (call_llm → tools ⇄ generate_final) |
| Memory | PostgreSQL + pgvector | 记忆写入/检索/召回统计/去重全链路已实现 |
| Entity Graph | PostgreSQL + pgvector | 实体归一化、一度关系查询、图谱可视化已实现 |
| Storage | PostgreSQL + pgvector | docker-compose 已就绪 |
| LLM | OpenAI SDK / Anthropic SDK | provider 抽象 + chat_raw 工具调用 + chat_json 结构化输出已实现 |
| Embedding | BGE-M3 (local) / OpenAI (API) | 本地离线 + OpenAI 兼容 API 双模式 |

## Layer Responsibilities

- **Frontend**: 用户交互、请求提交、结果展示
- **Backend**: API 接口、请求生命周期、调用 Agent
- **Agent**: ReAct 工具调用循环、状态管理、Tool/Memory 编排
- **Memory**: 长期记忆管理、检索、上下文构建、召回统计
- **Storage**: 业务数据 + 向量存储
- **LLM**: 统一模型调用封装，支持多 provider 切换，支持工具调用 (chat_raw)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/agent/chat` | Agent 对话（非流式）：ReAct 循环 + 工具调用 |
| POST | `/api/agent/chat/stream` | Agent 对话（SSE 流式）：逐 token 输出 + interrupt |
| GET | `/api/agent/threads` | 获取对话历史列表 |
| GET | `/api/agent/thread/{thread_id}` | 获取指定对话消息历史 |
| DELETE | `/api/agent/thread/{thread_id}` | 删除对话及 checkpoint 数据 |
| POST | `/api/memory/ingest` | 文档分块 → 嵌入 → 存入 chunks 表 |
| POST | `/api/memory/search` | 语义搜索：嵌入 → 向量检索（+稀疏）→ hybrid 默认跳过 rerank（rerank 仅服务层 opt-in，API 不暴露开关） |
| POST | `/api/memory/memories/write` | 结构化记忆写入：提取 → 相似度分级 → 合并/冲突/新插入 |
| POST | `/api/memory/memories/search` | 记忆搜索：相似度排序 → rerank → 记录召回 |
| GET | `/api/memory/memories/{memory_id}` | 通过 ID 获取单条记忆 |
| DELETE | `/api/memory/memories/{memory_id}` | 软删除记忆（设置 deleted_at） |
| GET | `/api/memory/stats` | 记忆库统计信息（总数、来源分布、高频实体、知识图谱指标等） |
| GET | `/api/entities/{entity_id}` | 获取实体档案：名称、类型、关联记忆数、来源分布 |
| GET | `/api/entities/{entity_id}/relations` | 获取实体一度关系：关联实体 + 最近记忆 |
| GET | `/api/entities/search?q=&type=` | 按名称搜索实体，支持类型过滤 |
| GET | `/api/connectors` | 列出已接入的连接器（PingCode/CI/飞书）与配置状态 |
| GET | `/api/connectors/{source}/logs` | 查看某连接器的投递日志 |
| GET | `/api/conflicts` | 列出待人工解决的记忆冲突（webhook/连接器 HITL 队列） |
| POST | `/api/conflicts/{id}/resolve` | 以 keep_existing/overwrite/merge/keep_both 之一解决冲突 |
| POST | `/api/conflicts/{id}/reopen` | 重新打开已解决的冲突 |
| POST | `/api/patrol/trigger` | 手动触发一次巡检 |
| GET | `/api/patrol/logs` | 巡检历史日志列表 |
| GET | `/api/patrol/logs/{log_id}` | 单次巡检详情（发现、冲突、告警） |
| POST | `/api/patrol/findings/{log_id}/dismiss` | 忽略一条巡检发现 |
| POST | `/api/patrol/findings/{log_id}/conflict` | 将巡检发现转入冲突队列 |
| GET | `/api/scenarios` | 列出可用垂直场景（复盘/审查/Onboarding/技术债） |
| POST | `/api/scenarios/{name}/run` | 运行指定场景 |
| POST | `/api/webhook/{source}` | 连接器事件入口（PingCode/CI/飞书 webhook） |
| GET | `/api/usage/summary?days=7` | LLM 调用按天汇总：调用数 / tokens / 错误数 / 平均延迟 / 估算成本 |
| GET | `/api/usage/scenarios?days=7` | LLM 调用按 scenario 汇总（各调用点的成本拆分） |
| GET | `/api/usage/models?days=7` | 按 provider/model 汇总 + 估算成本 |
| GET | `/api/usage/threads/{thread_id}` | 单个会话的 LLM 调用记录 |
| GET | `/api/usage/trace/{trace_id}` | 单次 trace 的 LLM 调用链（回放一次 agent 运行） |
| GET | `/api/usage/samples` | LLM 调用原始样本 |
| GET | `/health` | 存活探活：数据库连通性 + LLM/Embedding provider 配置与熔断器状态（DB 不可达返回 503 degraded） |

> **广度层按 flag 挂载（ADR-011）**：上表中的 connectors / webhook / patrol / scenarios 路由并非默认全部挂载。核心闭环（agent chat、memory、entities、conflicts、usage）恒挂载；connectors+webhook（`CONNECTORS_ENABLED`）、scenarios（`SCENARIOS_ENABLED`）、patrol（`PATROL_ENABLED`）默认关闭，flag 置 `true` 才挂载对应路由（未挂载返回 404）。三个 flag 在 `APP_ENV=test` 下视为开启，API 测试套件完整驱动全部路由。

## Technology Stack

### Backend

- **Language**: Python 3.12
- **Framework**: FastAPI
- **Async**: async/await + httpx

### API Security (Authentication)

所有 `/api` 路由（含 agent chat、memory 读写、ingest、connectors、patrol、webhooks）共享单一 API key 接入认证，实现于 `backend/api/auth.py`，作为全局依赖挂载在 `backend/api/router.py`（connectors/patrol/webhooks/scenarios 路由另受 ADR-011 的 `*_ENABLED` flag 控制，见上方端点表注释）：

- **认证模型**：请求须携带 `Authorization: Bearer <EMA_API_KEY>`，缺头 / 非 Bearer 方案 / key 不匹配均返回通用 401（`WWW-Authenticate: Bearer`），响应不携带任何可辅助攻击的细节。
- **常量时间比较**：key 校验用 `secrets.compare_digest`，避免时序侧信道；`APP_ENV=test` 完全豁免该守卫（API 测试套件依赖 mock，不需 key）。
- **前端注入**：构建期将 `VITE_EMA_API_KEY` 替换进 `import.meta.env.VITE_EMA_API_KEY`，前端请求自动携带该 Bearer。它与服务端 `EMA_API_KEY` 刻意不同——拆分仅保证运维用字符串不进入前端 bundle；**两把 key 在认证上完全等价**（`auth.py` 的 `require_api_key` 是唯一认证关卡，等同接受二者），bundle key 持有者拥有完整的读/写/删权限（见下方威胁模型）。未配置时前端不携带认证头（向后兼容，后端返回 401）。
- **边界与 ADR-005 的关系**：当前是"共享 key、无用户身份"——服务端不知请求来自谁、无登录/会话/角色/权限（两把 key 的 API 权限完全相同），这有意与 ADR 005 保持一致（不做多租户，故无用户→租户绑定）。用户级认证（JWT/session、多 key 按用户隔离）是引入多租户的前置条件，可分步引入（key 表 → 按 key 记 owner → 按 owner 过滤），迁移路径见 [ADR-005「用户级认证的迁移路径」](./decisions/ADR-005-no-multi-tenancy.md)。

#### 威胁模型

共享 key 认证的目标是**阻止未授权访问**，而不是**区分已授权者的权限**。边界：当前部署无公网暴露，使用方为单团队内部成员（ADR-005 前提）。

**防什么**
- 未携带有效 key 的请求（随机扫描、配置错误的客户端）——统一返回无细节 401
- key 校验的时序侧信道——`secrets.compare_digest` 常量时间比较
- 401 响应泄露可用于定向攻击的信息——不区分"缺 key / key 不匹配 / scheme 错误"

**不防什么（已知并接受）**
- **bundle key 持有者拥有全权**：`VITE_EMA_API_KEY` 打包进静态资源，任何拿到 bundle 的人都能提取，且与 `EMA_API_KEY` 权限完全相同——可读/写/删整个记忆库、触发巡检。两把 key 的差异仅是值不同，不是权限不同
- **请求无身份归属**：服务端不知请求来自哪个用户，无"谁写了/删了哪条记忆"的审计
- **key 泄漏无轮换粒度**：单 key 泄漏只能整体更换，无法按用户或按能力吊销

**为什么可接受 / 既有缓解**：单团队、无公网暴露是该设计的边界。在此边界内已有轻量缓解——记忆删除是软删除（`deleted_at`，误删可恢复）；agent 路径的写操作经 HITL 审批门（`check_approval_node`），不存在无人确认的批量写。即便 bundle key 被提取（公网暴露迟早发生，下述纵深防御在暴露后仍生效）：per-key 令牌桶（`backend/api/ratelimit.py`）限制滥用方用泄漏 key 刷 LLM 调用的成本；`usage.py` 的 trace_id 链路把 key 的每次调用记入 `llm_usage`，`/api/usage/trace/{id}` 可回放审计；写库前的 `_redact` 掩码（sk-/ghp_/AKIA、Bearer 头、PEM 块）保证明文敏感信息不落库。用户级认证 / 多 key 按用户隔离是引入多租户的前置条件，可分步引入（key 表 → 按 key 记 owner → 按 owner 过滤），迁移路径见 [ADR-005](./decisions/ADR-005-no-multi-tenancy.md)。

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

provider 层内置传输层韧性（`backend/shared/resilience.py`）：

- `chat` / `chat_raw` / `chat_sync` / `embed` / `embed_sync` 带 tenacity 指数退避重试（只重试瞬时错误：HTTP 429 / 5xx / 超时 / 连接错误，4xx 不重试），外加按 provider 命名的 in-memory 熔断器（连续失败达阈值后打开，冷却窗口内快速失败，冷却结束自动恢复探测）。
- `chat_json` 只挂熔断、不挂 tenacity 重试——该路径的传输重试由 `chat_structured` 的语义重试负责，避免两层重试嵌套（3×3）。
- 配置项：`LLM_RETRY_MAX_ATTEMPTS`、`LLM_RETRY_BACKOFF_BASE`、`LLM_RETRY_BACKOFF_MAX`、`LLM_CIRCUIT_BREAKER_THRESHOLD`、`LLM_CIRCUIT_BREAKER_COOLDOWN`。

### Observability (LLM usage tracing)

每次 LLM 调用都通过 provider 层（唯一咽喉点）记录一行观测，落 `llm_usage` 表（`backend/service/usage.py`）：

- **数据流**：provider 方法内同步、轻量地把观测值 append 进内存缓冲（线程安全、有界，溢出丢最旧并告警）→ 后台 flusher（`backend/main.py` lifespan 启动）每 `USAGE_FLUSH_INTERVAL_SECONDS`（默认 10s）批量 INSERT；进程优雅退出时再 flush 一次。观测代码绝不阻塞/拖垮 LLM 热路径，落库失败仅告警。同时为每次调用打一条带 trace_id 的结构化 `llm_call ...` 日志。
- **trace 链路**：入口处设置 `current_trace_id` contextvar（agent 的 `/chat`、`/chat/stream` 每请求一个新 uuid；patrol 用 `patrol_id`），provider 读取后为每条记录盖上 trace_id，`/api/usage/trace/{id}` 可端到端回放一次 agent 运行。会话维度由 `thread_id` 关联（webhook/连接器等未设 trace 的后台任务仍按 scenario 记录）。
- **字段**：trace_id / thread_id / scenario / provider / model / input|output|total_tokens / latency_ms / status(success|error) / error / prompt|response_chars / created_at（`seq` BIGSERIAL 保证插入顺序，trace 回放按它排序）。
- **成本估算**：`estimate_cost` 按模型内置价格表（deepseek / openai / claude 家族，$ 每 1M tokens）估算，未知模型用保守默认值——仅用于报告，真实账单以 provider 为准。
- 配置项：`USAGE_ENABLED`（默认 true）、`USAGE_FLUSH_INTERVAL_SECONDS`（默认 10）、`USAGE_BUFFER_MAX`（默认 5000）。

### Observability (Runtime health metrics — Prometheus)

除持久化成本行外，`backend/shared/runtime_metrics.py` 维护进程内健康时间序列，`GET /metrics`（`METRICS_ENABLED=true`，默认开）以 Prometheus 文本格式暴露，供 Prometheus 抓取 / Grafana 看板：

- **HTTP 层**：ASGI 中间件（`MetricsMiddleware`，挂在 `backend/main.py`）按路由模板路径记录请求数、延迟直方图（5ms–60s bucket）与状态码分布——`path` 取匹配到的 route 模板（`/api/agent/thread/{thread_id}` 而非具体 id），标签基数有界。
- **LLM 层**：与 `llm_usage` 同一咽喉点（`usage.record_call`）打点——按 scenario 的调用数（success/error）、延迟直方图、input/output/total token 计数。
- **韧性层**：`resilience.py` 的 `CircuitBreaker` 在状态转换处打点——open/closed 状态 gauge、进入 OPEN 次数、open 期间快速拒绝次数。
- **Agent 层**：交互式并发槽位占用 gauge + 超 `MAX_AGENT_CONCURRENCY` 的 503 拒绝计数（`agent_service.py`）；每次 chat 完成记录 ReAct 步数分布（`agent_routes.py`）——**task eval 测到的过度调用信号在产线持续可见**。
- **结构化输出降级**：实体/关系提取重试耗尽降级为 `[]` 时按 scenario 计数（`ema_structured_failures_total`）——**llm_usage 看不见这个信号**：降级的提取在 provider 层仍是 success 行（返回了 JSON、schema 校验在之后失败），只有该时序能暴露记忆质量的静默退化。
- 所有记录函数在 `config.metrics_enabled` 关闭时 no-op、异常吞掉，绝不阻塞/拖垮热路径；测试用 `reset_runtime_metrics()` 隔离。
- 与 `llm_usage` 的分工：**usage 表回答"花了多少钱、哪次调用贵"（历史查询）；Prometheus 回答"现在健不健康、慢在哪、并发满没满"（实时时序）**。二者由同一批 provider 打点驱动，互不依赖。
- **采集闭环已落地**（compose）：`prometheus` 服务每 15s 抓取 `/metrics`，`grafana` 服务渲染 "EMA — Runtime Health" 看板（9 面板，见 [deployment.md](./deployment.md) Monitoring）。当前为观测/可视化，Prometheus 告警规则 / Alertmanager 尚未配置。

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
- **四级相似度去重**：≥0.85 合并，0.72–0.85 冲突检测，0.60–0.72 补充关联，<0.60 新插入（阈值经标定，见 `tests/eval/reports/archive/threshold_calibration_report.md`）
- **召回统计**：检索按纯相似度排序，命中记忆记录 `recall_count`/`recalled_at` 作元数据；原艾宾浩斯衰减加权因 A/B 实测掉 recall（0.667 vs 无衰减 0.900，见 `tests/eval/reports/decay_ab_report.md`）已从排序路径移除，过期记忆归档改由 patrol LLM 读原始召回字段判断

### Agent

详见 [agent-design.md](agent-design.md)。核心能力：

- **ReAct 工具调用循环**：9 个 tool 封装记忆检索、文档搜索、实体查询、改写检索、记忆写入、知识提取、Git 摄取、文档摄取、飞书通知
- **自动知识捕获**：默认开启（`AUTO_MEMORY_ENABLED`），对话中涌现的实质性知识在回答交付后由后台任务自动提取入库——LLM 质量门判断"是否值得"、per-thread 间隔节流、内容哈希幂等防重复（见 [agent-design.md](agent-design.md) 自动知识捕获一节）
- **对话连续性**：thread_id 维持跨轮次上下文
- **容错降级**：LLM 调用失败不终止图执行

### Frontend

- **React + TypeScript + Vite + Tailwind CSS**
- 纯客户端 SPA，6 个页面：聊天、记忆库、实体图谱、连接器、巡检、冲突解决
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
