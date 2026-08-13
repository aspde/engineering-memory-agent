# 前端测试

## 当前状态

前端测试已落地：Vitest + React Testing Library + jsdom，**18 个测试文件、185 个测试用例**，覆盖 `components/`、`hooks/`、`context/`、`api/`、`pages/` 五个层级。`npm test`（vitest run）在 CI 中与后端 pytest 并行运行。

配置见 `frontend/vitest.config.ts`：

```ts
test: {
  environment: 'jsdom',
  include: ['src/**/*.test.{ts,tsx}'],
  setupFiles: ['./src/test-setup.ts'],
  globals: true,
  css: false,
}
```

## 选型决策：Vitest 而非 Jest

| 维度 | Vitest | Jest |
|------|--------|------|
| 配置成本 | 低 — 与 Vite 共用 transform 管线，近乎零配置 | 高 — 需额外 Babel/ts-jest 配置，与 Vite 管线重复 |
| 速度 | 快 — 原生 ESM，HMR 级别重跑 | 中 — 基于转换后的代码运行 |
| 兼容性 | 完美 — 与 vite.config.ts 共用插件 | 差 — 需维护两套 transform 配置 |

Jest 在 Vite 项目中没有优势；React Testing Library 是组件测试的事实标准，不绑特定 runner。依赖：`vitest`、`@testing-library/react`、`@testing-library/jest-dom`、`@testing-library/user-event`、`jsdom`。

一个真实的配置陷阱（`vitest.config.ts` 中已有注释）：测试依赖（`@testing-library/react`）会被 hoist 到仓库根 `node_modules`，而应用依赖在 `frontend/node_modules`，不强制单实例会导致 React hooks 在渲染时报错。用 `resolve.alias` 把 `react`/`react-dom` 指回 `frontend/node_modules` 解决。

## 结构与约定

- **co-located**：测试文件与源文件同级、同名 `*.test.{ts,tsx}`，不采用 `__tests__` 目录。
- **集中配置**：`frontend/src/test-setup.ts`（`@testing-library/jest-dom/vitest`）。
- **公共 wrapper**：`frontend/src/test-utils.tsx` 提供 `renderWithProviders`（`MemoryRouter` + `AppProvider` 包裹），页面/组件测试复用。

## Mock 分层

| 目标 | 方式 |
|------|------|
| API 调用（`apiGet`/`apiPost`/`apiSSE`） | `vi.mock('../api/client')` — 隔离网络，控制返回值 |
| 浏览器 API（`crypto.randomUUID`、`setInterval`） | `vi.stubGlobal()` / `vi.useFakeTimers()` |
| Context（AppProvider） | 真实渲染，手动 dispatch — 测试集成行为 |
| react-router（`useNavigate`） | `MemoryRouter` 包裹 |
| Tailwind 样式 | 不验证类名 — 样式属于视觉范畴 |

不用 MSW：前端 API 层很薄（`api/agent.ts`、`api/memory.ts` 等只是包装 `apiGet`/`apiPost`/`apiSSE` 的透传），`vi.mock()` 覆盖所有调用场景足够；MSW 的 Service Worker 架构对此规模是过度设计。

## 当前覆盖范围

按类别（18 个文件）：

| 类别 | 已测文件 |
|------|---------|
| api | `client.ts`（SSE 解析）、`capabilities.ts` |
| components | `ApprovalCard`、`ChatInput`、`CitationText`、`ConflictCard`、`MessageBubble`、`NavBar`、`PatrolBrief`、`RichText`、`StatsDashboard`、`ToolCallPanel` |
| context | `AppContext`（reducer 全 action） |
| hooks | `useChat`（SSE 流式、interrupt、resume）、`useMemories` |
| pages | `ConflictsPage`、`ConnectorsPage`、`PatrolPage` |

## 已知缺口

- **页面**：`ChatPage`、`MemoriesPage`、`EntityGraphPage` 无测试。这三个涉及多 hook + 多 API 调用协调，复杂度最高，被有意放在组件/hook 级覆盖之后（当初的 Tier 4）。
- **组件**：`ChatArea`、`IngestSection`、`MemoryCard`、`MemorySearch`、`Sidebar`、`SourcesPanel` 无测试。
- **API 模块**：`agent.ts`、`conflicts.ts`、`connectors.ts`、`entities.ts`、`memory.ts`、`patrol.ts`、`scenarios.ts` 未直接测试（仅经 `client.ts` 间接覆盖）。
