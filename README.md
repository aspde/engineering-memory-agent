# EMA — Engineering Memory Agent

面向研发团队的长期记忆智能体。将代码知识、Git 历史、技术决策和故障经验转化为可检索、可复用的长期记忆。

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 后端 | FastAPI + Python 3.12 | async 优先 |
| Agent | LangGraph | 单 Agent 架构，状态机编排 |
| LLM | OpenAI SDK / Anthropic SDK | 抽象接口，支持 DeepSeek、OpenAI、Claude |
| Embedding | BGE-M3 (sentence-transformers) | 本地部署，可替换 |
| 数据库 | PostgreSQL 16 + pgvector | 结构化数据 + 向量检索 |
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

# 创建 .env 并填入配置（参考下方配置说明）
```

### 启动数据库

```bash
docker compose up -d
```

### 启动服务

```bash
# 后端
uvicorn backend.main:app --reload --port 8000

# 前端
streamlit run frontend/app.py --server.port 8501
```

## 项目结构

```
ema/
├── backend/           # FastAPI (api/model/service/shared)
├── frontend/          # Streamlit MVP
├── tests/             # unit / integration / api
├── docs/              # 设计文档 & ADR
├── .claude/rules/     # Claude Code 规则
├── docker-compose.yml # PostgreSQL + pgvector
└── requirements.txt
```

## 开发

```bash
# 运行测试
pytest
```

## 文档

| 文档 | 内容 |
|---|---|
| [系统架构](docs/architecture.md) | 整体架构与分层设计 |
| [技术选型](docs/tech-stack.md) | 各层技术决策与理由 |
| [Agent 设计](docs/agent-design.md) | LangGraph 工作流设计 |
| [记忆系统](docs/memory-system.md) | 多层记忆架构 |
| [RAG 设计](docs/rag-design.md) | 检索增强生成管线 |
| [Git 知识提取](docs/git-knowledge.md) | 代码仓库数据摄入 |
| [部署](docs/deployment.md) | Docker Compose 与运行配置 |
| [ADR](docs/decisions/) | 架构决策记录 |

## 许可

MIT
