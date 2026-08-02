# Streamlit → React + TypeScript 前端迁移

## 问题描述

EMA 当前使用 Streamlit 作为 MVP 前端（`frontend/app.py` + `frontend/pages/chat.py` + `frontend/pages/memories.py`，共 ~40KB / ~980 行）。

Streamlit 在交互体验上存在几个根本性问题：

1. **全量 rerun 机制**：每次状态变更（`st.rerun()`）触发整个脚本重新执行，无法做到细粒度的 UI 更新。聊天界面中每次发送消息、审批操作、切换对话都要全量 rerun
2. **CSS hack 依赖**：聊天框固定底部、消息对齐等布局需求大量依赖 `data-testid` 选择器定位 DOM 元素，通过 JS 注入 `setInterval` 动态同步位置，脆弱且难以维护
3. **组件局限性**：无法实现自定义交互（消息编辑、代码块复制、对话框确认等）
4. **性能瓶颈**：WegRPC 双向通信增加延迟，大量历史消息时渲染变慢
5. **HITL 体验差**：审批/冲突解决流程需要 `st.rerun()` 全量刷新，无法做到就地更新

项目架构文档（`docs/architecture.md`）已明确规划 **"Frontend: Streamlit (MVP) / React (Production)"**，此迁移是按计划执行的生产化步骤。

## 目标

用 React + TypeScript + Vite + Tailwind CSS 重建前端，保持后端 API **零改动**。

## 方案选择

### 方案对比

| 维度 | 方案 A: Vite + React + Tailwind | 方案 B: Next.js + React + Tailwind |
|------|-------------------------------|-----------------------------------|
| 复杂度 | 低 — 纯客户端 SPA | 高 — SSR/SSG/Server Components |
| 部署 | 静态文件，任意 HTTP 服务器 | 需要 Node.js 运行时 |
| 构建速度 | 极快 (esbuild) | 中等 |
| 包大小 | ~50KB gzip | ~90KB+ gzip |
| 项目契合度 | ✅ "简单优先"原则 | ❌ 过度设计，2 页应用不需要 SSR |
| 路由 | React Router v7 | 内置 App Router |
| 生态 | 成熟稳定 | 最新但变化快 |

### 结论：选择方案 A

EMA 是客户端 SPA（2 个页面），无 SEO 需求，无服务端渲染需求。Vite 最简单，完全匹配项目的"简单优先"设计原则。3 个运行时依赖（React、ReactDOM、React Router），不引入组件库、状态管理库、HTTP 库。

## 架构图

```
                    ┌──────────────────────────┐
                    │   React + TypeScript SPA  │
                    │   (Vite dev / static)     │
                    │                          │
                    │  ┌─────┐  ┌───────────┐  │
                    │  │ Chat │  │ Memories  │  │
                    │  │ Page │  │   Page    │  │
                    │  └──┬───┘  └─────┬─────┘  │
                    │     └──────┬─────┘        │
                    │       AppContext           │
                    │     (useReducer)           │
                    │           │                │
                    │      API Layer             │
                    │  (fetch + ReadableStream)  │
                    └───────────┬──────────────┘
                                │ HTTP/SSE
                    ┌───────────┴──────────────┐
                    │   FastAPI Backend         │
                    │   (NO CHANGES)            │
                    │                           │
                    │  /api/agent/chat          │
                    │  /api/agent/chat/stream   │
                    │  /api/agent/threads       │
                    │  /api/agent/thread/{id}   │
                    │  /api/memory/*            │
                    └──────────────────────────┘
```

## 目录结构

```
frontend/
├── index.html                  # Vite entry HTML
├── package.json
├── tsconfig.json
├── vite.config.ts              # proxy /api → localhost:8000
├── tailwind.config.ts
├── postcss.config.js
├── src/
│   ├── main.tsx                # ReactDOM.createRoot
│   ├── App.tsx                 # Router + Sidebar + Context Provider
│   ├── index.css               # Tailwind directives + global styles
│   ├── api/
│   │   ├── client.ts           # fetch wrapper (base URL, error handling)
│   │   ├── agent.ts            # chat/chatStream/listThreads/getThread
│   │   └── memory.ts           # ingest/search/searchMemories/getMemory/stats
│   ├── context/
│   │   └── AppContext.tsx       # Global state: thread, messages, approval
│   ├── hooks/
│   │   ├── useChat.ts          # Streaming send + resume + state
│   │   └── useMemories.ts      # Stats fetch, search, ingest
│   ├── components/
│   │   ├── Sidebar.tsx         # Logo, new-chat button, thread list
│   │   ├── ChatArea.tsx        # Message list container
│   │   ├── MessageBubble.tsx   # User/assistant/system message render
│   │   ├── ApprovalCard.tsx    # Tool approval UI
│   │   ├── ConflictCard.tsx    # Conflict resolution UI
│   │   ├── SourcesPanel.tsx    # Clickable source references
│   │   ├── ToolCallPanel.tsx   # Formatted tool call results
│   │   ├── ChatInput.tsx       # Text input + send button
│   │   ├── StatsDashboard.tsx  # KPI cards + source dist + top entities
│   │   ├── IngestSection.tsx   # Text paste + file upload
│   │   ├── MemorySearch.tsx    # Search bar + result cards
│   │   └── MemoryCard.tsx      # Single memory display
│   ├── pages/
│   │   ├── ChatPage.tsx        # Chat page layout
│   │   └── MemoriesPage.tsx    # Memory library page layout
│   └── types/
│       └── index.ts            # All TypeScript interfaces
```

## 核心设计决策

### 1. 状态管理：React Context + useReducer

不使用 Redux/Zustand，因为全局状态仅 ~10 个字段，无中间件需求。

```typescript
interface AppState {
  threadId: string;
  messages: Message[];
  pendingInterrupt: Interrupt | null;
  waitingForApproval: boolean;
  threads: ThreadInfo[];
  loadedThreadId: string | null;
  memFilterId: string | null;
}

type Message = {
  role: 'user' | 'assistant' | 'system';
  content: string;
  _meta?: {
    toolCalls: ToolCall[];
    sources: Source[];
  };
};
```

### 2. SSE 流式处理：fetch + ReadableStream

使用 `fetch()` + `ReadableStream`（而非 EventSource），因为 EventSource 不支持 POST 请求。与后端 SSE 协议完全兼容。

Stream 事件处理：
- `token` → 追加到当前响应缓冲区，setState 触发增量渲染
- `node` → 显示状态提示（"思考中…" / "执行工具…"）
- `interrupt` → 停止流式，设置 `pendingInterrupt`，显示审批卡片
- `meta` → 保存 tool_calls + sources，关联到最终消息
- `error` → 显示错误 toast
- `done` → 流结束，持久化完整消息

### 3. HITL 流程

```
User sends message
  → SSE stream starts
  → tokens render incrementally
  → [if interrupt] stream stops, ApprovalCard shown
  → User clicks Approve/Reject
  → POST /api/agent/chat with resume_data (non-streaming)
  → AI response added to messages
  → User not blocked — can send next message

[if conflict]
  → ConflictCard shown with 4 resolution options
  → User selects resolution
  → POST /api/agent/chat with resume_data {"resolution": "merge|overwrite|..."}
  → AI response added to messages
```

### 4. Vite 开发代理

```typescript
// vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
});
```

### 5. 组件树 & 数据流

```
<App>
  <AppProvider>                       ← useReducer state
    <div className="flex h-screen">
      <Sidebar>                       ← reads threads[], threadId
        <NewChatButton />             ← dispatches NEW_CONVERSATION
        <NavLink to="/memories" />    ← React Router Link
        <ThreadList>                  ← reads threads[]
          {threads.map(t =>
            <ThreadItem              ← dispatches SET_THREAD_ID
              active={t.id===threadId}
            />
          )}
        </ThreadList>
      </Sidebar>
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/memories" element={<MemoriesPage />} />
        </Routes>
      </main>
    </div>
  </AppProvider>
</App>
```

## 接口定义

所有接口与后端 Pydantic models 一一对应。后端 API 完全不变。

### Agent API

| Method | Path | Request | Response |
|--------|------|---------|----------|
| GET | `/api/agent/threads` | — | `ThreadInfo[]` |
| GET | `/api/agent/thread/{thread_id}` | — | `ThreadMessagesResponse` |
| POST | `/api/agent/chat` | `ChatRequest` | `ChatResponse` |
| POST | `/api/agent/chat/stream` | `ChatRequest` | SSE stream |

### Memory API

| Method | Path | Request | Response |
|--------|------|---------|----------|
| POST | `/api/memory/ingest` | `IngestRequest` | `IngestResponse` |
| POST | `/api/memory/search` | `SearchRequest` | `SearchResponse` |
| POST | `/api/memory/memories/write` | `MemoryWriteRequest` | `MemoryWriteResponse` |
| POST | `/api/memory/memories/search` | `MemorySearchRequest` | `MemorySearchResponse` |
| GET | `/api/memory/memories/{memory_id}` | — | `MemoryGetResponse` |
| GET | `/api/memory/stats` | — | `MemoryStatsResponse` |

## 数据模型变更

无。后端 API 和数据模型完全不变。前端 TypeScript 类型是后端 Pydantic models 的镜像。

## 实施步骤

| # | 步骤 | 内容 | 文件数 |
|---|------|------|--------|
| 1 | **项目初始化** | Vite + React + TS 脚手架，安装 Tailwind、React Router | 5 |
| 2 | **类型定义** | `src/types/index.ts` — 完整 TypeScript 类型 | 1 |
| 3 | **API 层** | fetch 封装 + SSE 流式解析 + agent/memory 模块 | 3 |
| 4 | **状态管理** | AppContext with useReducer | 1 |
| 5 | **App Shell** | 路由、侧边栏、布局 | 2 |
| 6 | **聊天页** | 消息列表、气泡、输入框、useChat hook（含流式） | 5 |
| 7 | **审批 & 冲突** | 审批卡片、冲突解决面板、Sources 面板、ToolCall 面板 | 4 |
| 8 | **记忆库页** | 统计面板、摄入区、搜索、useMemories hook | 6 |
| 9 | **清理 & 文档** | 移动 Streamlit 到 backup，更新架构文档 | ~3 |

## 影响范围

- **后端**：无变更
- **Agent**：无变更
- **Memory**：无变更
- **Storage**：无变更
- **Frontend**：完全替换（Streamlit → React）
- **部署**：新增前端构建步骤（`npm run build` → 静态文件）
- **开发流程**：新增前端 dev server（`npm run dev`），通过 Vite proxy 连接后端

## 风险 & 缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| SSE 解析不完整 | token 丢失/乱码 | 使用 buffer + 行边界解析，同后端协议已充分测试 |
| 状态同步问题 | UI 与后端不一致 | useReducer 单一数据源，避免分散状态 |
| Streamlit 兼容期 | 需要同时维护两套前端 | Streamlit 文件仅移到 backup 保留，后端 API 不变 |
