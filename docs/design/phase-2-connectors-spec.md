# Phase 2: 多源连接器 — 功能规格

## Problem Statement

EMA 当前只有两个数据来源——用户手动上传文档和手动触发 Git 历史摄取。团队的真实知识大部分不在这些手动入口中，而是分散在 Jira issue、CI/CD 日志、Slack/飞书讨论、Confluence 文档等外部系统中。

每次需要 EMA 记住什么，用户都要手动复制粘贴——这就像搜索引擎需要你自己把网页内容贴进去才能索引。EMA 需要从"等你喂"变成"自己吃"。

## Solution

新增**连接器系统**——一个统一的 Connector 抽象接口 + Webhook 接收端点 + 首批三个适配器（Jira、CI、Slack）。每个连接器只做一件事：把外部系统的原始数据转换为 EMA 的标准化内容格式，然后走现有记忆写入管线。

## User Stories

### 连接器基础设施

1. As a developer, I want a common Connector interface (validate → normalize → process) so that adding a new data source is a matter of implementing three methods, with no changes to the ingestion pipeline.
2. As an operator, I want to configure webhook secrets per source, so that each external system authenticates independently and a leaked Slack secret doesn't compromise Jira.
3. As an operator, I want a webhook delivery log (success/failure + timestamp + source), so that I can debug when an external system sends malformed data.
4. As a developer, I want connector failures to log and return 4xx/5xx without crashing the API server, so that one misconfigured connector doesn't bring down other endpoints.

### Jira 连接器

5. As a team member, I want EMA to automatically ingest resolved Jira issues as memories, so that bug root causes and fixes are searchable without anyone manually pasting them.
6. As a team member, I want the Jira connector to extract the issue key, summary, description, resolution, and linked commits into a single structured memory, so that the memory contains enough context for future retrieval.
7. As a team member, I want Jira issues marked as "bug" or "incident" to be tagged with source_type "jira_bug", distinct from "jira_story", so that fault-related memories can be filtered and cross-referenced with other fault data.

### CI 连接器

8. As a team member, I want EMA to automatically ingest CI build failures as memories, so that recurring build issues are tracked and can be correlated across time.
9. As a team member, I want the CI connector to capture the failed job name, error summary, commit SHA, and branch, so that the memory can be linked to specific code changes (via Phase 1 entity normalization).
10. As a team member, I want CI pipeline duration regressions (e.g., test suite went from 3min to 12min) to be detected and stored as a distinct memory type "ci_regression", so that performance degradations are tracked separately from outright failures.

### Slack 连接器

11. As a team member, I want to paste a message link in Slack and have EMA ingest the surrounding thread as a conversation memory, so that technical discussions in chat become searchable knowledge.
12. As a team member, I want the Slack connector to preserve the thread structure (who said what, in what order) but extract only the technical substance, so that the memory is concise rather than a raw chat log.
13. As a team member, I want Slack messages tagged with a decision emoji or keyword (e.g., `:memo:` or `/remember`) to be automatically ingested, so that teams can signal "this is worth keeping" without leaving Slack.

### 记忆可溯源

14. As a user, I want every memory from a connector to carry a link back to its original source (Jira URL, CI build URL, Slack thread permalink), so that I can verify context or read the full original discussion.
15. As a user, I want to filter memories by source in the Web UI — "show me only CI-derived memories" or "show me everything from Jira" — so that I can focus on a specific knowledge category.

### Agent 行为

16. As a user, I want the Agent to be aware of connectors as a data source — when I ask "what did we learn from last week's incidents?", the Agent should search across both manually-written memories and Jira/CI-derived ones without me specifying where to look.
17. As a user asking about a specific Jira issue (e.g., "EMA-42"), I want the Agent to first search for a memory derived from that issue, and if not found, be able to tell me "that issue hasn't been ingested yet."

### 前端

18. As a user, I want a "Connectors" settings page in the Web UI where I can see which connectors are configured, their status (active/inactive/error), and recent delivery logs.
19. As a user, I want to see a source badge on each memory card in the memory library — e.g., a Jira icon for jira-derived memories, a Git icon for git-derived memories — so that I can visually distinguish knowledge origins.

## Implementation Decisions

### Connector 抽象接口

所有连接器实现同一个接口，四个方法：

```
Connector ABC:
  validate(payload: dict) → bool                    # 校验 webhook payload 格式合法
  normalize(payload: dict) → str                    # 外部格式 → EMA 标准 content 文本
  normalize_batch(payloads: list[dict]) → list[str] # 批量归一化（batch_mode）
  process(content: str, metadata: dict)             # → write_memory() / write_chunks()
```

`process()` 有默认实现，直接调用 `write_memory()`。特殊连接器可以 override（例如 CI 连接器可能想同时写 chunk 和 memory）。

### batch_mode 设计（为 Phase 3 铺路）

**问题**：Phase 3 主动 Agent 上线后，事件驱动响应会大规模增加连接器吞吐量——例如一次 CI 构建包含 200 个 job，每个 job 独立触发一次 webhook → 一次 `write_memory()` → 一次 embedding 调用。BGE-M3 本地模式下每次 embedding 调用是 GPU/CPU 密集型操作，200 次串行调用会将一次事件响应的延迟从毫秒级拉到分钟级。更严重的是，如果多个事件同时到达，embedding 队列会迅速堆积——**嵌入成本在自动化后爆炸**。

**方案**：Connector 接口预留 `batch_mode`。当 webhook 接收到的 payload 是数组（`Content-Type: application/json` + 顶层 JSON 数组），自动切换到批量处理路径：

```
Webhook 接收层:
  if isinstance(payload, list) and connector.supports_batch:
      → connector.normalize_batch(payload)
      → 合并为一次 write_memory() 调用（多段内容，一次 embedding 批量计算）
  else:
      → connector.normalize(payload)
      → write_memory()（单条，现有路径）
```

**设计约束**：
- `batch_mode` 是接口层的预留，Phase 2 首批三个连接器只需实现 `normalize_batch()` 的默认实现（循环调用 `normalize()`——逐条处理，无性能优化，但接口已就绪）
- 真正的 embedding 批量优化在 Phase 3 触发——当性能监控显示单条处理成为瓶颈时，各连接器按需覆盖 `normalize_batch()` 实现真正的批量化
- `write_memory()` 本身已接受单条内容——批量写入的接口变更不在此 spec 范围内，Phase 3 时评估是否需要 `write_memories_batch()`
- 前端 Connectors 设置页展示每个连接器的 `batch_mode` 支持状态：`supported` / `pending` / `not_applicable`

### Connector 注册机制

通过 `source_type` 到 connector 实例的映射注册：

```python
# registry 不是 class，是一个 dict + 一个 factory 函数
# 这决定了 Connector 的发现方式——不是插件扫描，是显式注册
CONNECTOR_REGISTRY: dict[str, Connector] = {}

def register_connector(source_type: str, connector: Connector) -> None
def get_connector(source_type: str) -> Connector
```

应用启动时注册所有可用连接器。如果某个连接器缺少配置（如 API key），注册但不激活，前端显示为 "pending"。

### Webhook 端点

统一端点：`POST /api/webhook/{source}`

每个 source 使用独立的 secret 验证（配置在 `.env` 中）：

```
WEBHOOK_JIRA_SECRET=...
WEBHOOK_CI_SECRET=...
WEBHOOK_SLACK_SECRET=...
```

请求签名验证通过 HMAC-SHA256 header（标准 webhook 模式），验证失败返回 401，成功后将 payload 交给对应 connector 处理。

### 新增 webhook_logs 表

用于调试和状态展示：

```sql
CREATE TABLE webhook_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    event_type TEXT,              -- e.g., "issue.resolved", "build.failed"
    status TEXT NOT DEFAULT 'received',  -- received | processed | failed
    payload_summary TEXT,         -- first 200 chars of payload
    memory_id UUID,               -- resulting memory, if successfully written
    error TEXT,                   -- error message, if failed
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### 首批三个连接器

| 连接器 | source_type | normalize 做的事 |
|--------|-------------|-----------------|
| `JiraConnector` | `jira_issue` | issue.key + fields.summary + fields.description + fields.resolution → 结构化文本 |
| `CIConnector` | `ci_build` | job_name + status + error_summary + commit_sha + duration → 结构化文本 |
| `SlackConnector` | `slack_thread` | 提取 thread 消息列表 → 过滤 bot/系统消息 → 格式化为对话文本 |

每个连接器是一个独立的 Python 文件，不相互依赖。新增一个连接器只需要新建一个文件 + 在 registry 中注册——不改任何已有代码。

### API 新端点

- `POST /api/webhook/{source}` — 接收外部 webhook。验证签名 → 交给 connector → 返回处理结果
- `GET /api/connectors` — 列出所有注册的连接器及其状态（active/pending/error）
- `GET /api/connectors/{source}/logs` — 返回该 source 最近的 webhook 投递日志（分页，最近 50 条）

### 前端新增

- `src/pages/Connectors.tsx`（路由 `/connectors`）— 连接器列表 + 状态 + 最近投递日志
- 现有 `MemoryCard` 组件增加 source badge（小图标 + source_type 标签）
- `IngestSection` 中 source_type 枚举扩展

### Schema 迁移

- 新增 `webhook_logs` 表
- `memories` 表不改——`source_type` 字段是 free text，新连接器直接写新值即可
- 无破坏性变更

## Testing Decisions

### 测试原则

- Connector 的 `normalize()` 是纯数据转换，**不需要 mock**——输入已知 payload、断言已知输出
- Webhook 端点的签名验证逻辑需要 mock HMAC
- Connector 的 `process()` 方法中调用 `write_memory()` 需要 mock（验证参数传递正确即可）
- 已有管线（`write_memory` → `normalize_entities`）不重复测试

### 测试模块

| 测试文件 | 内容 | 参考模式 |
|---------|------|---------|
| `tests/unit/test_connector_base.py` | Connector ABC、registry 注册/获取 | test_llm_service.py（接口 + 实现模式） |
| `tests/unit/test_connector_jira.py` | Jira payload → normalize → content 文本 | 新——纯数据测试，无 mock |
| `tests/unit/test_connector_ci.py` | CI payload → normalize → content 文本 | 同上 |
| `tests/unit/test_connector_slack.py` | Slack payload → normalize → content 文本 | 同上 |
| `tests/api/test_webhook_routes.py` | 签名校验、format 校验、正常流程 | test_memory_routes.py |
| `src/pages/Connectors.test.tsx` | 连接器列表渲染、状态展示 | StatsDashboard.test.tsx |

### 测试用例速览

- `test_validate_rejects_missing_required_fields` — 校验拒绝不合规 payload
- `test_validate_accepts_valid_payload` — 校验接受合法 payload
- `test_normalize_jira_bug_produces_structured_text` — 输出 EMA 标准格式
- `test_normalize_ci_failure_includes_commit_sha` — 关键字段不丢失
- `test_normalize_slack_thread_strips_bot_messages` — 噪音过滤
- `test_process_calls_write_memory_with_correct_params` — 管线调用正确
- `test_webhook_invalid_signature_returns_401` — 签名校验
- `test_webhook_valid_payload_returns_200_with_memory_id` — 正常流程
- `test_webhook_unknown_source_returns_404` — 未注册 source
- `test_registry_get_connector_returns_correct_instance` — registry 正确性

## Out of Scope

- 双向同步（EMA → Jira/Slack）——连接器只做摄入，不反向写回
- 连接器热加载/插件系统——新增连接器需要改代码 + 重新部署
- 消息队列 / 异步处理——webhook 同步处理 + 写数据库，负载上来后再说
- Webhook URL 自动配置——用户在外部系统中手动设置 webhook URL，EMA 不主动注册
- OAuth 集成——用 shared secret（HMAC-SHA256），不做 OAuth flow
- Confluence / Notion / Linear 连接器——首批只做 Jira + CI + Slack，其他按需添加
- 新依赖引入——`hmac` 和 `hashlib` 是标准库，不引入新的 Python 包

## Further Notes

- 本 spec 对应 [ADR-006](./decisions/ADR-006-extension-roadmap.md) 的 Phase 2
- 连接器的价值依赖 Phase 1（实体归一化）——外部数据摄入后需要能链接到已有实体才有意义
- CI 连接器设计为通用接口——支持 GitHub Actions、GitLab CI、Jenkins 等，只需各自实现 `normalize()` 的差异化部分
- 连接器自己的配置（API keys、webhook secrets）存储在 `.env` 中，不存数据库——保持现有配置管理模式一致
