# 文档索引

本目录包含 EMA 项目的设计文档和知识记录。供人类开发者阅读，也作为 Memory System 的知识检索来源。

## 文档列表

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | 系统架构、分层设计、技术选型 |
| [agent-design.md](agent-design.md) | LangGraph 单 Agent 工作流设计 |
| [memory-system.md](memory-system.md) | 记忆系统设计（含 RAG 检索、Git 知识提取） |
| [deployment.md](deployment.md) | Docker Compose 部署与运行配置 |
| [decisions/](decisions/) | 架构决策记录（ADR） |

## 维护约定

- 每个文档描述**当前系统状态**，非未来计划
- 只记录代码无法表达的信息：决策原因、设计权衡、系统边界
- 更新规则见 [.claude/rules/workflow.md](../.claude/rules/workflow.md#文档同步)
