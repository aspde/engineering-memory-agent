# 05 — Agent 连接器感知

**What to build:** Agent 的工具描述更新，使其意识到连接器作为数据来源的存在。用户问"上周的事故我们学到了什么？"时，Agent 自动跨手动记忆和连接器来源记忆搜索，无需用户指定。用户询问特定 Jira issue（如"EMA-42 什么状态？"）时，Agent 先搜索已有记忆，若未找到则回复"该 issue 尚未被摄入"而不是泛泛的"不知道"。

**Blocked by:** 01 — 连接器基础设施 + Webhook 端点 + Jira 连接器

**Status:** ready-for-agent

- [ ] `search_memories_tool` 描述更新：说明可搜索的来源类型（手动、Jira、CI、Slack、Git），默认搜索全部
- [ ] `query_entity_tool` 描述更新：提示可查询连接器摄入的实体
- [ ] Agent system prompt 更新：当用户询问特定外部系统 issue/事件时，先搜索记忆，若未找到则友好提示"该 issue/事件 尚未被摄入"而非"不知道"
- [ ] 单元测试：`test_agent_tools.py` 中增加连接器感知相关测试（工具描述包含连接器关键词、搜索行为覆盖连接器来源）
