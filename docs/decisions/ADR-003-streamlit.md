# ADR-003: Use Streamlit for MVP

## Status

Superseded (see ADR-003b below)

## Context

项目核心价值在 Agent 能力、Memory 系统和 RAG 流程，而非复杂前端。MVP 阶段需要快速验证 AI 能力。

## Decision

使用 **Streamlit** 作为 MVP 前端，实现 Chat UI + Memory 查询 + 项目知识展示。

## Alternatives

### React + Tailwind + Ant Design

Rejected for MVP — 增加前端开发成本，降低 AI 能力验证速度。

## Consequences

- ✅ 快速搭建，验证 AI 能力
- ⚠️ Streamlit 灵活性有限，不适合复杂交互

---

## ADR-003b: Migrate to React + TypeScript

### Status

Accepted

### Context

Streamlit MVP 验证了核心 AI 能力（Agent、Memory、RAG 流程）可行。产品进入正式开发阶段，Streamlit 的局限性（无客户端路由、无 SSE 流式、复杂交互受限、样式控制弱）成为瓶颈。

### Decision

迁移到 **React + TypeScript + Vite + Tailwind CSS**，纯客户端 SPA，通过 REST + SSE 与 FastAPI 后端通信。不再使用 Streamlit。

### Consequences

- ✅ 支持 SSE 流式聊天、Human-in-the-Loop 审批与冲突解决
- ✅ 客户端路由（聊天页 / 记忆库页），Tab 切换等复杂 UI 交互
- ✅ Tailwind CSS 精准控制样式
- ✅ TypeScript 类型安全
- ⚠️ 增加前端开发成本，但已由 Streamlit MVP 阶段吸收验证风险
