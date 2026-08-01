# EMA — Engineering Memory Agent

面向研发团队的长期记忆智能体。自动从 Git、代码和文档中提取知识，转化为可检索的长期记忆。

## 绝对约束

1. **Agent**：LangGraph 单 Agent 架构，禁止 Multi-Agent、禁止 LangChain Agent
2. **存储**：PostgreSQL + pgvector 作为唯一数据存储方案，禁止引入或替换为其他数据库
3. **LLM**：通过 LLMProvider 抽象接口调用，业务代码禁止直接依赖具体 SDK
4. **异步**：所有 IO 操作（API 调用、数据库访问、文件 IO）优先使用 async/await

## 模型策略（成本优化）

- **当前模型 (pro)**：负责架构设计、方案讨论、代码审查 — 需要深度思考的任务
- **flash 子智能体**：负责写代码、修 bug、运行测试 — 执行类任务
- **核心原则**：pro 负责"想"，flash 负责"做"；pro 会话不切模型以保护缓存，flash 不在意缓存重建成本
- **委派方式**：用 Agent 工具指定 `model: "haiku"` 将执行任务委派给 flash 子智能体

## 规则

详见 `.claude/rules/`。

## 文档

系统设计与技术决策见 `docs/`。
