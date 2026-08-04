# 04 — 连接器管理 API + 前端页面 + 来源徽章

**What to build:** 用户可以在 Web UI 中看到所有连接器的状态和投递日志，记忆卡片上显示来源图标。

- `GET /api/connectors` — 返回所有已注册连接器列表，含状态（active/pending/error）和 batch_mode 支持级别
- `GET /api/connectors/{source}/logs` — 分页返回该 source 最近 50 条 webhook 投递日志
- 新增 `/connectors` 前端页面：连接器卡片（名称、状态徽章、batch_mode 标识），点击可展开最近投递日志
- `MemoryCard` 组件增加来源徽章：按 source_type 显示对应颜色图标（Jira=蓝、CI=绿、Slack=紫、Git=灰）
- Sidebar 增加"连接器"导航入口
- 记忆搜索支持按 source_type 过滤

**Blocked by:** 01 — 连接器基础设施 + Webhook 端点 + Jira 连接器

**Status:** ready-for-agent

- [ ] `GET /api/connectors` 端点：返回连接器列表 + status + batch_mode
- [ ] `GET /api/connectors/{source}/logs` 端点：分页返回 webhook_logs（最近 50 条，支持 ?limit=&offset=）
- [ ] `/connectors` 前端页面：连接器卡片列表 + 状态徽章 + batch_mode 标签 + 最近日志弹窗
- [ ] `MemoryCard` 来源徽章：按 source_type 显示颜色编码图标（jira_issue/jira_bug→蓝, ci_build/ci_regression→绿, slack_thread→紫, git_commit→灰, 其他→默认）
- [ ] Sidebar 增加"🔌 连接器"按钮，导航到 `/connectors`
- [ ] 记忆搜索/记忆库页面支持按 source_type 过滤
- [ ] API 测试：`test_connector_routes.py`（列出连接器、查看日志、分页）
- [ ] 前端测试：`Connectors.test.tsx`（连接器列表渲染、状态展示）
