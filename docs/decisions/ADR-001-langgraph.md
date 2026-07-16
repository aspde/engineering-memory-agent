# ADR-001 Use LangGraph


## Status

Accepted


## Context

系统需要构建具有：

- 状态管理
- 多步骤执行
- Tool Calling
- Memory

能力的 AI Agent。


---

## Decision

使用 LangGraph 作为 Agent Orchestration Framework。


---

## Alternatives


### LangChain Agent


Rejected。


原因：

- 黑盒程度较高
- 状态控制能力不足
- 调试困难


---

## Consequences


Positive:

- Workflow 清晰
- 状态显式管理
- 易于调试


Negative:

- 开发复杂度增加