# 01 — 连接器基础设施 + Webhook 端点 + Jira 连接器

**What to build:** 第一条完整的 webhook→记忆 管线。用户（或外部系统）向 `POST /api/webhook/jira` 发送 Jira issue webhook → HMAC-SHA256 签名验证 → JiraConnector 校验并归一化 payload → 调用 `write_memory()` → 记忆以 `source_type="jira_issue"`（bug/incident 则为 `"jira_bug"`）存入数据库。每次投递记录写入 `webhook_logs` 表，可在前端查看投递状态。连接器注册机制（`register_connector` / `get_connector`）就位，后续添加 CI、Slack 只需新建一个文件 + 一行注册。

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] Connector ABC（`validate` / `normalize` / `normalize_batch` / `process`）定义完成，`process()` 有默认实现调用 `write_memory()`
- [ ] Connector registry（`register_connector` / `get_connector` / `list_connectors`）就位
- [ ] `webhook_logs` 表创建（id, source, event_type, status, payload_summary, memory_id, error, created_at）
- [ ] `WEBHOOK_JIRA_SECRET` 等环境变量在 config 中可配置
- [ ] `POST /api/webhook/{source}` 端点：HMAC-SHA256 签名验证 → 查 registry → validate → normalize → process → 写 webhook_log
- [ ] 签名验证失败返回 401，未知 source 返回 404，校验失败返回 400，处理异常返回 500
- [ ] JiraConnector：validate 校验必要字段（issue.key, fields.summary），normalize 输出结构化文本（key + summary + description + resolution），bug/incident 类型标记为 `jira_bug`
- [ ] 记忆的 metadata 中携带原始 Jira URL（可溯源）
- [ ] 单元测试：`test_connector_base.py`（ABC 接口、registry 注册/获取）、`test_connector_jira.py`（validate/normalize 纯数据测试）
- [ ] API 测试：`test_webhook_routes.py`（签名校验 401、合法 payload 200、未知 source 404、缺失字段 400）
