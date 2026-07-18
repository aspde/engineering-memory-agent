# Agent Design

## Framework

LangGraph — 单 Agent 架构，状态机驱动的 workflow 编排。

## Agent Workflow (planned)

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

## Capabilities (planned)

| Capability | Description | Status |
|-----------|-------------|--------|
| Code Search | 查询代码仓库 | 🔜 |
| Git History Search | Commit / Diff / Author / File History | 🔜 |
| Memory Search | 检索历史研发知识 | 🔜 |
| Answer Generation | 结合上下文生成回答 | 🔜 |

## Constraints

- 单 Agent 架构，禁止 Multi-Agent
- 禁止使用 LangChain Agent（仅用 LangGraph）
- Tool 与 Agent 解耦，通过接口调用
- 避免依赖真实 LLM 响应进行测试
