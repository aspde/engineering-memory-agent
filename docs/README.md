# 文档索引

本目录包含 EMA 项目的设计文档和知识记录。供人类开发者阅读，也作为 Memory System 的知识检索来源。

## 文档列表

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | 系统架构、分层设计、技术选型 |
| [agent-design.md](agent-design.md) | LangGraph 单 Agent 工作流设计 |
| [memory-system.md](memory-system.md) | 记忆系统设计（含 RAG 检索、Git 知识提取） |
| [deployment.md](deployment.md) | Docker Compose 部署与运行配置 |
| [design/domain-model.md](design/domain-model.md) | 领域模型：核心概念、实体关系、演进路线 |
| [decisions/ADR-006-extension-roadmap.md](decisions/ADR-006-extension-roadmap.md) | 四阶段扩展路线图 |
| [decisions/ADR-007-patrol-in-process-scheduler.md](decisions/ADR-007-patrol-in-process-scheduler.md) | 巡检调度器内嵌主进程，不引入任务队列 |
| [design/phase-1-entity-graph-spec.md](design/phase-1-entity-graph-spec.md) | Phase 1 Spec: 知识图谱化 |
| [design/phase-2-connectors-spec.md](design/phase-2-connectors-spec.md) | Phase 2 Spec: 多源连接器 |
| [design/phase-3-proactive-agent-spec.md](design/phase-3-proactive-agent-spec.md) | Phase 3 Spec: 主动 Agent |
| [design/phase-4-vertical-scenarios-spec.md](design/phase-4-vertical-scenarios-spec.md) | Phase 4 Spec: 垂直场景孵化 |
| [decisions/](decisions/) | 架构决策记录（ADR） |
| [design/](design/) | 设计文档（前端测试、删除 API、React 迁移等） |

## 维护约定

- 每个文档描述**当前系统状态**，非未来计划
- 只记录代码无法表达的信息：决策原因、设计权衡、系统边界
- 更新规则见 [.claude/rules/workflow.md](../.claude/rules/workflow.md#文档同步)
