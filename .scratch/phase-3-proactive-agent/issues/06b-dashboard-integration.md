# 06b — 仪表盘 + 侧边栏集成

**What to build:** 将巡检功能集成到现有导航和仪表盘中。侧边栏新增"巡检日志"导航项（图标 + 路由 `/patrol`），放在现有四个导航项之后。首页/仪表盘新增"今日简报"卡片——调用 `GET /api/patrol/logs?limit=1&patrol_type=daily` 获取最近一次每日巡检，展示前 3 条 pattern match findings 的摘要（标题 + 描述摘要），如果今日尚无巡检则显示"今日巡检尚未执行，预计 8:00 完成"提示。卡片提供"查看全部 →"链接跳转至 `/patrol` 页面。

**Blocked by:** 06a（共用 Patrol types 和 API client）

**Status:** ready-for-agent

- [ ] `frontend/src/components/Sidebar.tsx` 新增"巡检日志"导航按钮：图标（可用 eye/scan 类 SVG icon），文字"巡检日志"，路由 `/patrol`，放在现有导航项之后（新聊天、记忆库、实体图谱、连接器、巡检日志）
- [ ] `frontend/src/components/PatrolBrief.tsx`（或扩展 `StatsDashboard.tsx`）：调用 `listPatrolLogs({limit: 1, patrol_type: 'daily'})` 获取最新 daily patrol，展示卡片标题"今日简报"，列出前 3 条 pattern match findings（标题 + 一行描述截断），底部"查看全部 →"链接
- [ ] 今日简报状态处理：有 findings 时渲染列表；无 findings 时显示"今日巡检未发现需关注事项 ✅"；尚无巡检记录时显示"今日巡检尚未执行，预计 8:00 完成"；API 加载失败时静默降级（不显示卡片，不阻断页面）
- [ ] 今日简报卡片放在首页仪表盘区域（`ChatPage` 顶部或 `StatsDashboard` 区域），视觉风格与现有 StatsDashboard 卡片一致
- [ ] 前端测试：`frontend/src/components/PatrolBrief.test.tsx` — 有 findings 时渲染列表、无 findings 时渲染"未发现"文案、无数据时渲染"尚未执行"文案、API 错误时组件不崩溃且不渲染卡片
