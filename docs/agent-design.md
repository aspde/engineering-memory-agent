# Agent Design

## Framework

LangGraph — 单 Agent 架构，状态机驱动的 workflow 编排。

## Agent Workflow

```
User Query
    ↓
Intent Classification → Router
    ↓
Tool Selection → Tool Execution
    ↓
Memory Retrieval → Context Assembly
    ↓
LLM Generation → Response
```

## State Management

LangGraph 管理 Agent 状态流转：

- 状态定义明确
- 节点职责清晰
- 支持条件分支和循环

## Capabilities

| Capability | Description | Status |
|-----------|-------------|--------|
| Code Search | 查询代码仓库 | 待实现 |
| Git History Search | Commit / Diff / Author / File History | 待实现 |
| Memory Search | 检索历史研发知识 | 待实现 |
| Answer Generation | 结合上下文生成回答 | 待实现 |

## Constraints

硬约束见 [.claude/rules/constraints.md](../.claude/rules/constraints.md)。补充约束：

- 避免依赖真实 LLM 响应进行测试
