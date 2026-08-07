# EMA — Engineering Memory Agent

面向研发团队的长期记忆智能体。自动从 Git、代码和文档中提取知识，转化为可检索的长期记忆。

## 绝对约束

1. **Agent**：LangGraph 单 Agent 架构，禁止 Multi-Agent、禁止 LangChain Agent
2. **存储**：PostgreSQL + pgvector 作为唯一数据存储方案，禁止引入或替换为其他数据库
3. **LLM**：通过 LLMProvider 抽象接口调用，业务代码禁止直接依赖具体 SDK
4. **异步**：所有 IO 操作（API 调用、数据库访问、文件 IO）优先使用 async/await

## 开发环境

- **操作系统**：Windows 10 Pro，终端为 PowerShell。需要用户手动执行的命令（如交互式登录、启动本地服务）按 PowerShell 语法书写，不要假定 POSIX 命令（`mv` / `cat` / `grep` 等）可用
- **数据库**：本地 PostgreSQL（`postgresql://ema:ema123@localhost:5432/ema_dev`），含 pgvector 扩展
- **测试**：`python -m pytest`；测试环境用 NullPool，规避 Windows ProactorEventLoop + asyncpg 的连接池跨事件循环复用问题（见 `backend/db/__init__.py`）
- **Windows 已知兼容性问题**：psycopg3 异步驱动（AsyncPostgresSaver）在 Windows 下不可用，checkpointer 降级为 InMemorySaver，开发期可接受（见 `backend/main.py`）；生产部署目标为 Linux

## 规则

详见 `.claude/rules/`。

## 文档

系统设计与技术决策见 `docs/`。
