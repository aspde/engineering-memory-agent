# ADR-003: 前端选型 — Streamlit MVP 过渡至 React + TypeScript

## Status

Accepted

## Context

项目核心价值在 Agent 能力、Memory 系统和 RAG 流程，而非复杂前端。MVP 阶段为快速验证 AI 能力，选择 Streamlit 作为前端，实现 Chat UI + Memory 查询 + 项目知识展示。Streamlit 成功验证了核心 AI 能力（Agent、Memory、RAG 流程）可行。

产品进入正式开发阶段后，Streamlit 的局限成为瓶颈：

- 无客户端路由
- 无 SSE 流式
- 复杂交互受限
- 样式控制弱

## Decision

前端采用 **React + TypeScript + Vite + Tailwind CSS** 纯客户端 SPA，通过 REST + SSE 与 FastAPI 后端通信，不再使用 Streamlit。迁移保留 Streamlit MVP 阶段已验证的 AI 能力，前端定位为后端 API 的消费者之一，不引入独立前端架构。

## Alternatives

### Streamlit 作为最终前端

否决：无客户端路由、无 SSE 流式、复杂交互受限、样式控制弱，正式开发阶段无法支撑。

### 首版即上 React

否决：增加前端开发成本，降低 AI 能力验证速度——MVP 阶段先接受 Streamlit 的交互局限，代价在正式开发阶段以一次迁移偿还。

### React + Tailwind + Ant Design（MVP 时考虑的前端组合）

否决：已确定的 React 路线下用 Tailwind CSS 承担样式，不引入 Ant Design 组件库，最终为 React + TypeScript + Vite + Tailwind CSS。

## Consequences

- ✅ 快速搭建，验证 AI 能力（MVP 阶段）
- ✅ 支持 SSE 流式聊天、Human-in-the-Loop 审批与冲突解决
- ✅ 客户端路由（聊天页 / 记忆库页等），复杂 UI 交互
- ✅ Tailwind CSS 精准控制样式
- ✅ TypeScript 类型安全
- ⚠️ 增加前端开发成本，但已由 Streamlit MVP 阶段吸收验证风险——迁移不改变后端 API
