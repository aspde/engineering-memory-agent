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
| 前端 MVP | Streamlit | 快速验证 |

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
streamlit run frontend/app.py --server.port 8501
```

## 项目结构

```
ema/
├── agent/              # LangGraph Agent (state, tools, nodes, graph)
├── backend/            # FastAPI (api/model/service/shared/db)
├── frontend/           # Streamlit MVP
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
| POST | `/api/memory/ingest` | 文档分块 → 嵌入 → 入库 |
| POST | `/api/memory/search` | 语义搜索 chunks |
| POST | `/api/memory/memories/write` | 结构化记忆写入（提取 → 去重 → 合并） |
| POST | `/api/memory/memories/search` | 记忆搜索（衰减加权） |
| POST | `/api/agent/chat` | Agent 对话（ReAct 循环 + 工具调用） |

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
| [部署](docs/deployment.md) | Docker Compose 与运行配置 |
| [ADR](docs/decisions/) | 架构决策记录 |

## 许可

MIT
