# 04 — 飞书通知集成

**What to build:** 新增 `notify_feishu` Agent Tool 和飞书 Webhook 配置，使巡检结果和事件告警能推送到团队飞书群。Tool 通过 HTTP POST 调用飞书机器人 Webhook，支持两种消息格式：`text`（纯文本）和 `interactive`（富文本卡片，含标题和 Markdown 正文）。通知级别按 finding 类型映射：pattern_match → critical（🔴）、knowledge_gap → warning（🟡）、decay_cleanup → info（🔵）、new_entity → info。Agent 在巡检完成后判断是否需要推送——有 actionable findings 时调用 notify_feishu，无发现时跳过。Webhook 调用失败时降级：记录错误日志，不影响巡检结果存储。

**Blocked by:** 01（需要巡检执行产生 findings 才能推送）

**Status:** ready-for-agent

- [ ] `agent/tools.py` 新增 `notify_feishu_tool`：LangChain `@tool` 装饰，参数 `message: str`、`msg_type: str = "text"`、`title: str | None = None`，内部通过 `httpx.AsyncClient` POST 调用飞书 Webhook URL
- [ ] `msg_type="text"` 时构建 `{"msg_type": "text", "content": {"text": message}}` payload
- [ ] `msg_type="interactive"` 时构建飞书卡片格式 `{"msg_type": "interactive", "card": {"header": {...}, "elements": [...]}}`
- [ ] `backend/shared/config.py` 新增 `FEISHU_WEBHOOK_URL`（str，无默认值，为空时 tool 返回错误提示而非崩溃）
- [ ] `.env.example` 新增飞书 Webhook 配置项
- [ ] `notify_feishu_tool` 注册到 `agent/tools.py` 的 `ALL_TOOLS` 列表
- [ ] CI_FAILURE_PATROL_PROMPT 和 JIRA_RESOLVED_PATROL_PROMPT 中 instructions：有 actionable findings 时调用 `notify_feishu_tool` 推送（msg_type="interactive"），无发现时跳过
- [ ] Webhook HTTP 调用设置 timeout（默认 10s），超时或网络错误时 log error 并返回 `{"ok": false, "error": "..."}`，不抛出异常、不重试
- [ ] 单元测试：`tests/unit/test_notify_feishu.py` — `test_notify_feishu_formats_text_message`、`test_notify_feishu_formats_interactive_message`、`test_notify_feishu_returns_error_when_not_configured`、`test_notify_feishu_handles_timeout`、`test_tool_schema_has_required_params`
- [ ] 单元测试：`tests/unit/test_agent_tools.py` — 扩展已有测试，验证 `notify_feishu_tool` 的 tool schema 正确（参数名、类型、required fields）
