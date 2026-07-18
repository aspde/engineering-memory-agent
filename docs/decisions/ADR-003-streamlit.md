# ADR-003: Use Streamlit for MVP

## Status

Accepted

## Context

项目核心价值在 Agent 能力、Memory 系统和 RAG 流程，而非复杂前端。MVP 阶段需要快速验证 AI 能力。

## Decision

使用 **Streamlit** 作为 MVP 前端，实现 Chat UI + Memory 查询 + 项目知识展示。

## Alternatives

### React + Tailwind + Ant Design

Rejected for MVP — 增加前端开发成本，降低 AI 能力验证速度。

## Future

正式版本迁移到 React + Tailwind CSS。

## Consequences

- ✅ 快速搭建，验证 AI 能力
- ⚠️ Streamlit 灵活性有限，不适合复杂交互
