# ADR-001: Use LangGraph

## Status

Accepted

## Context

系统需要构建支持状态管理、多步骤执行、Tool Calling、Memory 的 AI Agent。

## Decision

使用 **LangGraph** 作为 Agent 编排框架。单 Agent 架构，状态显式管理。

## Alternatives

### LangChain Agent

Rejected。
- 黑盒程度较高
- 状态控制能力不足
- 调试困难

## Consequences

- ✅ Workflow 清晰，状态显式管理，易于调试
- ⚠️ 相对单纯的 API 调用，开发复杂度增加
