# ADR-006: EMA 进阶扩展路线图

**日期**: 2026-08-04

**状态**: 已接受

## 背景

EMA 基础功能已就绪：ReAct Agent 循环、三阶段记忆提取、四级相似度去重、召回统计、HITL 审批。项目进入进阶扩展阶段，需要在多个可能方向中做出优先级决策。

经过结构化 grilling 讨论（[领域模型文档](../design/domain-model.md)），确定了四个扩展方向及其依赖关系。

## 决策

**扩展按以下阶段推进，每个阶段是下一阶段的必要前提：**

```
Phase 1: 知识图谱化（实体归一化 + 一度关系查询 + 可视化）
    ↓
Phase 2: 多源连接器（Webhook 接收 + 连接器接口 + Jira/CI/Slack 适配）
    ↓
Phase 3: 主动 Agent（任务调度 + 自主巡检 + 洞察推送）
    ↓
Phase 4: 垂直场景（故障复盘 / 代码审查 / 新人 Onboarding，按需孵化）
```

| Phase | 关键词 | 依赖 Phase | 预计周期 | 核心可交付物 |
|-------|--------|-----------|---------|-------------|
| 1 | 知识图谱 | — | 1-2 周 | 实体归一化、一度关系 API、关系图谱可视化 |
| 2 | 连接器 | 1（归一化实体让新数据能关联到已有知识） | 1-2 周 | Webhook 端点、连接器接口、首批适配器 |
| 3 | 主动 Agent | 1+2（需要丰富知识和多源事件触发） | 2-3 周 | 任务调度器、自主巡检、洞察通知 |
| 4 | 垂直场景 | 1+2+3 | 按需 | 场景专用 System Prompt + Tool 集合 |

## 理由

### Phase 1 最先做的原因

1. **杠杆率最高**：实体和关系数据已存在于 JSONB 中，提取管线成熟。缺少的只是归一化层和查询接口——工作量小，但让所有记忆之间产生关联。
2. **零新依赖**：实体归一化用 LLM + pgvector 向量相似度，一度关系查询用纯 SQL，不需要引入图数据库（延续 [ADR-004](./ADR-004-no-neo4j.md) 决策）。
3. **解锁后续 Phase**：Phase 2 的新数据源想关联到已有知识，Phase 3 的主动 Agent 想做洞察，都需要实体归一化作为基础。

### Phase 2 在 Phase 1 之后的原因

- 连接器引入的外部数据（Jira issue、CI 事件、Slack 消息）需要能关联到已有实体才有意义
- 连接器接口设计依赖 Phase 1 确定的数据模型

### Phase 2 预留：连接器批量归一化（batch 脚手架）

`Connector` 基类预留了一套批量归一化接口（`backend/connectors/base.py`）：`supports_batch`（默认 `False`）、`batch_mode`（`supported` / `pending` / `not_applicable`）、`normalize_batch()`（默认逐条循环 `normalize`）。它预想的是"一次收到多条同源 payload"的场景——批量转换可共享上下文（同一批 CI job 的 repo/branch、同一批 PingCode issue 的项目），并潜在合并嵌入/提取调用。

**当前状态：预留未激活。** 三个连接器（feishu / pingcode / ci）都没有 override `supports_batch`，`batch_mode` 恒为 `"pending"`；`normalize_batch` 除定义处外无调用方（webhook 路径逐条处理，`backend/api/routes/webhook_routes.py`）；唯一消费是 `GET /api/connectors` 把 `batch_mode` 透传给前端连接器页的状态徽章（`frontend/src/pages/ConnectorsPage.tsx`，恒定显示"逐条处理"）。

**为何保留而不删除**：webhook 路径每条事件独立 `extract_memory`（3 次 LLM 提取），一旦出现批量回放/批量导入场景，批量归一化能在嵌入与提取层合并调用，接口预留可避免届时改动 `Connector` 契约。保留成本仅是三个未使用的属性 + 一个静态 UI 徽章，符合"每个 Phase 只做解锁下一阶段所需的最小能力"。

**启用触发信号**：出现真实的批量消费场景（连接器批量回放历史事件、批量导入既有数据）时，先为第一个连接器实现 `normalize_batch` 并设 `supports_batch=True`；`not_applicable` 仅在某个源永远逐条时由该连接器 override 返回。无触发信号前不实现任何连接器的批量路径。

### Phase 3 在 Phase 1+2 之后的原因

- 主动 Agent 的价值取决于知识的丰富程度和事件源的覆盖范围
- 没有图谱，Agent "想不清楚"；没有连接器，Agent "看不见东西"
- LangGraph 的 `interrupt()` 和 `PostgresSaver` 已为自主工作流预留了入口

### Phase 4 不做提前设计的原因

- 垂直场景会在前三个 Phase 完成后自然涌现
- 每个垂直场景本质上是不同的 System Prompt + Tool 集合，不需要架构变更

## 不做的事

| 事项 | 决策 | 依据 |
|------|------|------|
| 多租户/团队隔离 | **不做** | EMA 的核心价值是打破知识孤岛；用 metadata tag 替代硬隔离；等 SaaS 化时再评估 |
| 图数据库（Neo4j 等） | **不做** | 延续 ADR-004；一度关系纯 SQL 即可；多跳遍历可先用递归 CTE |
| 复杂图分析（中心性、社区发现） | **不做** | 工程记忆的查询模式是"关于 X 我们有什么"，不是"遍历网络结构" |
| IDE 插件 | **暂不做** | 等知识库足够丰富、团队养成使用习惯后再评估 |
| ChatOps Bot | **Phase 3 附近做** | 成本低（API 统一出口已就绪），但不是当前瓶颈 |
| Embedding provider 扩展 | **Phase 2 附带做** | OpenAI Embedding 等已在 `EmbeddingConfig` 中预留，工作量很小 |

## 交互形态演进

```
现在：API 补强（Webhook + Batch） + Web 做厚（图谱可视化）
    ↓
Phase 3 附近：ChatOps Bot（Slack/飞书），开发成本约一个周末
    ↓
远期：IDE 插件（等知识库和用户习惯成熟）
```

API 是所有界面的统一后端——每个新界面只是 API 的另一个消费者。

## 后果

- 每次 Phase 切换需重新评估下一步的优先级和范围
- Phase 1 的实体归一化方案是所有后续工作的基础——如果归一化效果不理想，需要及时调整
- 坚持简单优先：每个 Phase 只做"解锁下一阶段所需的最小能力"，不做过度设计
- 所有决策保持可逆——如果某个方向被验证不需要，可以砍掉而不影响已完成的部分
