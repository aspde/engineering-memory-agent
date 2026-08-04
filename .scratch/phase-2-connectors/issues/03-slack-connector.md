# 03 — Slack 连接器

**What to build:** `POST /api/webhook/slack` 接收 Slack Events API payload → SlackConnector 校验事件类型，提取 thread 消息列表，过滤 bot/系统消息，保留对话顺序 → 记忆以 `source_type="slack_thread"` 存储。包含 `:memo:` emoji 或 `/remember` 关键字的消息触发自动摄入并标记高优先级。输出的记忆是简洁的技术讨论摘要，而非原始聊天记录。

**Blocked by:** 01 — 连接器基础设施 + Webhook 端点 + Jira 连接器

**Status:** ready-for-agent

- [ ] SlackConnector 实现：validate 校验事件类型（message, app_mention），normalize 提取 thread 消息 → 过滤 bot/subtype 消息 → 格式化为带发言者标签的对话文本
- [ ] `:memo:` 或 `/remember` 标记检测：含标记的消息在 metadata 中设置 `auto_ingest: true` 和高优先级标记
- [ ] 在 registry 中注册 SlackConnector（`WEBHOOK_SLACK_SECRET` 配置即激活）
- [ ] 单元测试：`test_connector_slack.py`（validate 拒绝非 message 事件、normalize 过滤 bot 消息、保留对话顺序、检测 :memo: 标记、检测 /remember 标记）
