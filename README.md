# EMA — Engineering Memory Agent

研发团队长期记忆智能体。将代码知识、Git 历史、技术决策和故障经验转化为可检索、可复用的长期记忆。

## 架构

```
User → Streamlit/FastAPI → LangGraph Agent → Retriever → PostgreSQL + pgvector → LLM → Response
```

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 后端 | FastAPI | Python 3.12, async 优先 |
| Agent | LangGraph | 状态管理, 多步骤流程, Tool Calling |
| LLM | DeepSeek (OpenAI SDK) | 模型抽象层, 可切换 |
| 数据库 | PostgreSQL 16 + pgvector | 结构化数据 + 向量检索 |
| 缓存 | Redis | 热点数据, 会话状态 |
| 前端 MVP | Streamlit | 快速验证 |
| 前端生产 | React | 后续迁移 |

## 快速开始

### 环境要求

- Python 3.12
- Docker (PostgreSQL + pgvector)

### 安装

```bash
# 克隆项目
git clone https://github.com/aspde/engineering-memory-agent.git
cd engineering-memory-agent

# 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Keys
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
├── backend/           # FastAPI 服务 (api/service/repository/model)
├── agent/             # LangGraph Agent 定义
├── rag/               # RAG 检索管线 (Retriever/Reranker)
├── ingestion/         # 数据摄入
├── frontend/          # Streamlit UI
├── tests/             # pytest + pytest-asyncio
├── docs/              # 设计文档 & ADR
├── docker-compose.yml # PostgreSQL + pgvector
└── .claude/           # Claude Code 规则 & Skills
```

## 开发

### 运行测试

```bash
pytest
```

### 代码规范

- Python type hints
- async 优先
- 函数职责单一
- 单文件不超过 300 行
- 业务逻辑不在 controller 层

## 文档

| 文档 | 内容 |
|---|---|
| [技术选型](docs/tech-stack.md) | 各层技术决策与理由 |
| [系统架构](docs/architecture.md) | 整体架构与核心组件 |
| [Agent 设计](docs/agent-design.md) | LangGraph 工作流设计 |
| [RAG 设计](docs/rag-design.md) | 检索增强生成管线 |
| [记忆系统](docs/memory-system.md) | 多层记忆架构 |
| [Git 知识提取](docs/git-knowledge.md) | 代码仓库数据摄入 |
| [部署](docs/deployment.md) | Docker Compose 与缓存策略 |
| [ADR](docs/decisions/) | 架构决策记录 |

## 许可

MIT
