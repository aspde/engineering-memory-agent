# EMA — Engineering Memory Agent

面向研发团队的长期记忆智能体。将代码知识、Git 历史、技术决策和故障经验转化为可检索、可复用的长期记忆。

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 后端 | FastAPI + Python 3.12 | async 优先 |
| Agent | LangGraph (手动 StateGraph) | 单 Agent 架构，ReAct 循环 + 工具调用 |
| LLM | OpenAI SDK / Anthropic SDK | 抽象接口，支持 DeepSeek、OpenAI、Claude |
| Embedding | BGE-M3 (sentence-transformers) | 本地离线部署，可替换 |
| 数据库 | PostgreSQL 16 + pgvector | 结构化数据 + 向量检索 + 对话 checkpoints |
| 前端 | React + TypeScript + Vite + Tailwind CSS | SPA，6 页面（聊天 / 记忆库 / 实体图谱 / 连接器 / 巡检 / 冲突解决）+ HITL 审批流 |

## 快速开始

### 环境要求

- Python 3.12
- Docker

### 安装

```bash
git clone https://github.com/aspde/engineering-memory-agent.git
cd engineering-memory-agent

python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

pip install -r requirements.txt

# 创建 .env 并填入 LLM_API_KEY（参考 .env 示例）
```

### 启动数据库

```bash
docker compose up -d
```

### 启动服务

```bash
# 后端（自动创建 pgvector 扩展 + 表 + 对话 checkpoint 表）
uvicorn backend.main:app --reload --port 8000

# 前端
cd frontend && npm run dev
```

## 项目结构

```
ema/
├── backend/            # FastAPI + LangGraph Agent (api/agent/service/shared/db)
├── frontend/           # React + TypeScript + Vite SPA
├── tests/              # unit / integration / api
├── docs/               # 设计文档 & ADR
├── .claude/rules/      # Claude Code 规则
├── docker-compose.yml  # PostgreSQL + pgvector
├── .env                # 环境变量
└── requirements.txt
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/agent/chat` | Agent 对话：ReAct 循环 + 工具调用 |
| POST | `/api/agent/chat/stream` | Agent 对话（SSE 流式）：逐 token + interrupt |
| GET | `/api/agent/threads` | 获取对话历史列表 |
| GET | `/api/agent/thread/{thread_id}` | 获取指定对话消息历史 |
| DELETE | `/api/agent/thread/{thread_id}` | 删除对话及 checkpoint 数据 |
| POST | `/api/memory/ingest` | 文档分块 → 嵌入 → 入库 |
| POST | `/api/memory/search` | 语义搜索 chunks |
| POST | `/api/memory/memories/write` | 结构化记忆写入（提取 → 去重 → 合并） |
| POST | `/api/memory/memories/search` | 记忆搜索（衰减加权） |
| GET | `/api/memory/memories/{id}` | 通过 ID 获取单条记忆 |
| DELETE | `/api/memory/memories/{id}` | 软删除记忆 |
| GET | `/api/memory/stats` | 记忆库统计（总数、来源分布、高频实体） |
| GET | `/api/entities/search?q=&type=` | 按名称搜索实体 |
| GET | `/api/entities/{entity_id}` | 获取实体档案 |
| GET | `/api/entities/{entity_id}/relations` | 获取实体一度关系 |
| GET | `/api/conflicts` | 待人工解决的记忆冲突列表 |
| POST | `/api/conflicts/{id}/resolve` | 解决冲突（keep_existing/overwrite/merge/keep_both） |
| POST | `/api/patrol/trigger` | 手动触发巡检 |
| GET | `/api/patrol/logs` | 巡检历史日志 |
| POST | `/api/scenarios/{name}/run` | 运行垂直场景 |
| POST | `/api/webhook/{source}` | 连接器事件入口 |
| GET | `/api/usage/summary?days=7` | LLM 调用按天汇总（含成本估算） |
| GET | `/api/usage/trace/{trace_id}` | 单次 trace 的 LLM 调用链回放 |

## 开发

```bash
# 运行测试
pytest

# 跳过 BGE-M3 集成测试（更快）
pytest tests/unit/ tests/api/
```

## 文档

| 文档 | 内容 |
|---|---|---|
| [系统架构](docs/architecture.md) | 整体架构、分层设计、技术选型 |
| [Agent 设计](docs/agent-design.md) | LangGraph ReAct 循环、设计决策、工具目录 |
| [记忆系统](docs/memory-system.md) | 记忆架构：提取、去重、衰减、检索、Git 摄取 |
| [领域模型](docs/design/domain-model.md) | 核心概念、实体关系、演进路线 |
| [部署](docs/deployment.md) | Docker Compose 与运行配置 |
| [ADR](docs/decisions/) | 架构决策记录 |
| [扩展路线图](docs/decisions/ADR-006-extension-roadmap.md) | 四阶段扩展计划 |

## 许可

MIT
