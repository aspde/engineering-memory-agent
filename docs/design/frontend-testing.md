# 前端测试方案

## 问题描述

EMA 前端已从 Streamlit MVP 迁移到 React + TypeScript + Vite + Tailwind CSS，但当前前端零测试。所有测试都是 Python 后端测试。前端缺乏测试导致：

- 组件回归风险不可见
- SSE 流式处理、HITL 审批等核心交互无自动化验证
- reducer 状态逻辑复杂（14 种 action），纯手工验证易出错

## 方案对比

### 方案 A：Vitest + React Testing Library（推荐）

**核心思路**：Vite 原生测试框架 + DOM 测试工具。

| 维度 | 评价 |
|------|------|
| 配置成本 | 极低 — 与 Vite 共用 transform 管线，近乎零配置 |
| 速度 | 快 — 原生 ESM，HMR 级别重跑 |
| 兼容性 | 完美 — 与现有 vite.config.ts 共用插件 |
| 生态 | 成熟 — vitest-dom、@testing-library/react 全套 |
| 学习成本 | 低 — API 与 Jest 兼容，团队熟悉 pytest 可类比 |

### 方案 B：Jest + React Testing Library

**核心思路**：传统 Jest 生态 + DOM 测试工具。

| 维度 | 评价 |
|------|------|
| 配置成本 | 高 — 需额外 Babel/ts-jest 配置，与 Vite 管线重复 |
| 速度 | 中 — 基于转换后的代码运行，不如 Vitest |
| 兼容性 | 差 — 需维护两套 transform 配置 |
| 生态 | 最成熟 — 插件和文档最丰富 |
| 学习成本 | 低 — 业界标准 |

### 选择：方案 A — Vitest

Jest 在 Vite 项目中没有优势。Vitest 与 Vite 共用配置，零额外 transform 开销。API 与 Jest 兼容，迁移成本为零。React Testing Library 是组件测试的事实标准，不绑特定 runner。

## 包选型

```json
// devDependencies 新增
{
  "vitest": "^3.0.0",                          // 测试框架
  "@testing-library/react": "^16.0.0",          // React 组件测试
  "@testing-library/jest-dom": "^6.0.0",        // DOM 断言扩展
  "@testing-library/user-event": "^14.0.0",     // 用户交互模拟
  "jsdom": "^25.0.0"                            // DOM 环境模拟
}
```

## 配置文件

### vitest.config.ts

```ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
});
```

### src/test/setup.ts

```ts
import '@testing-library/jest-dom/vitest';
```

### package.json scripts

```json
"test": "vitest run",
"test:watch": "vitest"
```

## 测试结构

采用 **co-located** 方式，测试文件放在源文件旁边：

```
frontend/src/
  context/
    AppContext.tsx
    AppContext.test.ts          ← 与源文件同级
  hooks/
    useChat.ts
    useChat.test.ts
  api/
    client.ts
    client.test.ts
  components/
    MessageBubble.tsx
    MessageBubble.test.tsx
    ApprovalCard.tsx
    ApprovalCard.test.tsx
  test/
    setup.ts                    ← 唯一集中配置文件
```

**命名规则**：`<filename>.test.ts(x)`。

**不采用** `__tests__` 目录方案 — 对小型项目而言，co-located 让测试与源文件的关系更直观，import 路径更短。

## Mock 策略

### 分层 Mock

| 目标 | 方式 | 原因 |
|------|------|------|
| API 调用 (`apiGet`, `apiPost`, `apiSSE`) | `vi.mock('../api/client')` | 隔离网络，控制返回值 |
| 浏览器 API (`crypto.randomUUID`, `setInterval`) | `vi.stubGlobal()` / `vi.useFakeTimers()` | 确定性、可控 |
| Context (AppProvider) | 真实渲染，手动 dispatch | 测试集成行为 |
| react-router (`useNavigate`) | `MemoryRouter` 包裹 | 测试导航 |
| Tailwind 样式 | 不验证类名 | 样式属于视觉范畴 |

### 为什么不用 MSW

项目只有 4 个 API 端点，API 层很薄（`api/agent.ts`、`api/memory.ts` 都只是包装 `apiGet`/`apiPost`/`apiSSE`）。`vi.mock()` 足够覆盖所有 API 调用场景。MSW 的 Service Worker 架构对于这个规模的项目是过度设计。

### 通用 Test Wrapper

```tsx
// src/test/utils.tsx
import { ReactNode } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { AppProvider } from '../context/AppContext';
import { render, type RenderOptions } from '@testing-library/react';

function TestWrapper({ children }: { children: ReactNode }) {
  return (
    <MemoryRouter>
      <AppProvider>{children}</AppProvider>
    </MemoryRouter>
  );
}

function renderWithProviders(ui: ReactNode, options?: Omit<RenderOptions, 'wrapper'>) {
  return render(ui, { wrapper: TestWrapper, ...options });
}

export { renderWithProviders };
```

## 测试优先级 & 示例

### Tier 1 — 纯逻辑（最快，最有价值）

**1. `appReducer` 测试**

纯函数，无依赖，覆盖所有 14 种 action：

```ts
describe('appReducer', () => {
  it('SET_THREAD_ID clears messages and pending interrupt')
  it('ADD_MESSAGE appends to messages array')
  it('UPDATE_LAST_MESSAGE handles empty messages safely')
  it('UPDATE_LAST_MESSAGE appends content to last message')
  it('UPDATE_LAST_MESSAGE applies meta to last message')
  it('SET_INTERRUPT sets pendingInterrupt and waitingForApproval')
  it('CLEAR_INTERRUPT clears both fields')
  it('NEW_CONVERSATION resets to clean state with new threadId')
  it('SET_LOADED_THREAD sets loadedThreadId')
  it('INVALIDATE_THREADS resets threadsFetchedAt to 0')
  it('SET_MEM_FILTER / CLEAR_MEM_FILTER')
  it('unknown action returns state unchanged')
})
```

测试的是 `appReducer` 纯函数（导出或通过 import 直接访问），不需要 React。

**2. `api/client.ts` SSE 解析**

纯函数，无 DOM 依赖，测试字节流 → 事件映射：

```ts
describe('apiSSE', () => {
  it('yields token event from data: {"type":"token","content":"hello"}')
  it('yields node event')
  it('yields interrupt event with typed data')
  it('yields meta event with tool_calls and sources arrays')
  it('yields error event from data payload')
  it('handles multi-byte UTF-8 split across chunks')
  it('skips malformed JSON lines')
  it('skips non-data lines (comments, empty)')
  it('throws ApiError on non-2xx response')
  it('throws on network fetch failure')
})
```

需要 mock `fetch` 返回 `ReadableStream`。

### Tier 2 — Hooks（中等复杂度）

**3. `useChat` hook**

核心流式逻辑，需要 fake timers + mock API：

```ts
describe('useChat', () => {
  describe('sendMessage', () => {
    it('dispatches user and assistant messages immediately')
    it('streams tokens via SSE, dispatched as UPDATE_LAST_MESSAGE')
    it('flushes tokens on 50ms interval')
    it('appends node labels for non-generate_final nodes')
    it('handles interrupt — flushes, stops timer, dispatches SET_INTERRUPT')
    it('handles error event — appends error to last message')
    it('handles meta event — dispatches tool_calls and sources')
    it('handles network error — appends error message')
    it('aborts previous stream on new sendMessage call')
    it('does nothing on empty/whitespace input')
  })
  describe('resume', () => {
    it('calls chatNonStream with resume_data')
    it('handles second interrupt after resume')
    it('dispatches assistant message with response')
    it('handles network error gracefully')
  })
})
```

**4. `useMemories` hook**

```ts
describe('useMemories', () => {
  it('fetchStats loads and caches stats')
  it('fetchStats sets error state on failure')
  it('search returns results')
  it('ingest calls API and refreshes stats')
  it('getMemoryById returns single memory')
})
```

### Tier 3 — 组件（交互验证）

**5. `ApprovalCard`**

```tsx
describe('ApprovalCard', () => {
  it('renders tool name in Chinese')
  it('calls onResume({approved: true}) on approve button click')
  it('calls onResume({approved: false, reason: ...}) on reject button click')
  it('buttons are disabled when isResolving is true')
})
```

**6. `MessageBubble`**

```tsx
describe('MessageBubble', () => {
  it('renders user message right-aligned with blue bubble')
  it('renders assistant message left-aligned with grey bubble')
  it('renders typing indicator when streaming with empty content')
  it('renders system message with amber notice')
  it('shows ToolCallPanel when _meta.toolCalls present')
  it('shows SourcesPanel when _meta.sources present')
})
```

**7. `Sidebar`**

```tsx
describe('Sidebar', () => {
  it('shows loading skeletons on initial render')
  it('shows thread list after fetch')
  it('highlights active thread')
  it('NEW_CONVERSATION dispatches with new UUID and navigates to /')
  it('navigates to /memories on memory library button click')
})
```

### Tier 4 — 页面（集成测试，可选）

ChatPage 和 MemoriesPage 涉及多个 hook 和 API 调用的协调，复杂度高。优先保证 hook 和组件级别的测试覆盖，页面级测试在 Tier 3 完成后按需补充。

## 实施步骤

| 步骤 | 内容 | 预估 |
|------|------|------|
| 1 | 安装 vitest + @testing-library/react + jsdom 等依赖 | 1 次 |
| 2 | 创建 `vitest.config.ts`、`src/test/setup.ts`、`src/test/utils.tsx` | 3 个文件 |
| 3 | 在 `package.json` 添加 `test` / `test:watch` scripts | 2 行 |
| 4 | 写 `appReducer` 测试 (~15 cases) | Tier 1 |
| 5 | 写 `api/client.ts` SSE 解析测试 (~10 cases) | Tier 1 |
| 6 | 写 `useChat` 测试 (~12 cases) | Tier 2 |
| 7 | 写 `useMemories` 测试 (~4 cases) | Tier 2 |
| 8 | 写组件测试 (~20 cases across 6 components) | Tier 3 |
| 9 | 运行 `npm test` 验证全部通过 | — |

## 影响分析

- **对现有代码**：零修改。`appReducer` 需要从 `AppContext.tsx` 导出（目前是内部函数）。其他组件和 hook 已具备可测试结构。
- **对构建**：vitest 配置文件与 vite.config.ts 并列，不影响生产构建。
- **对 CI**：新增 `npm test` 步骤，与现有 `pytest` 并行运行。
- **对开发流程**：`npm run test:watch` 提供 watch 模式，与 Vite HMR 体验一致。
