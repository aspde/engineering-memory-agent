# EMA Task-Level E2E Evaluation Report

- Generated: 2026-08-24 01:34:52 UTC
- Suites: 1
- Tasks: 8
- Execution errors: 0
- Judge degradations: 0

## Overall

### 任务级端到端

| metric | completed | tool_recall | unexpected_rate | within_budget | n_steps | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.375 | 0.750 | 0.000 | 0.875 | 1.625 | 0.375 | 1.000 | 0.000 | 0.625 |


## 任务级端到端 by category

| category | completed | tool_recall | unexpected_rate | within_budget | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|
| factual | 0.500 | 1.000 | 0.000 | 1.000 | 0.500 | 1.000 | 0.000 | 0.500 |
| multi_retrieve | 0.500 | 0.750 | 0.000 | 1.000 | 0.500 | 1.000 | 0.000 | 0.500 |
| write | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| conceptual | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| notify | 0.000 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| no_tool | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

<details><summary>Per-task detail (任务级端到端)</summary>

**task-001** (factual)
- query: 向量检索后端选了什么方案，为什么不用 Elasticsearch？
- trajectory: search_memories_tool (n=1, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=340)
- answer: `向量检索后端选的是 **PostgreSQL 的 pgvector 扩展**，而不是 Elasticsearch。选型理由有三点（memory: 6e0b7685、1052118b）：

1. **与业务库同库**：向量和业务数据在同一个 …`

**task-002** (multi_retrieve)
- query: 线上 502 事故的根因是什么？另外长代码文件分块时会怎么处理？两个都回答。
- trajectory: search_memories_tool, retrieve_chunks_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=270)
- answer: `**1. 线上 502 事故根因**（memory: 202f0cc7）
- 根因是**数据库连接池被占满**。
- 触发链路：流量上涨 → 连接池无空闲连接 → 新请求排队超时 → 网关返回 502。
- 修复措施：重启连接池，并增加连接…`

**task-003** (write)
- query: 为什么嵌入模型选 BGE-M3？然后把『嵌入模型用 BGE-M3，1024 维，本地推理』记成一条记忆。
- trajectory: - (n=0, steps=0, completed=0.000, within_budget=0.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=0)

**task-004** (conceptual)
- query: 之前出过什么问题，会不会陷入死循环？怎么避免？
- trajectory: search_memories_tool, search_memories_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-005** (notify)
- query: 怎么把本地仓库的提交历史导入 EMA？然后把方法飞书通知给团队。
- trajectory: notify_feishu_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=467)
- answer: `**飞书通知发送失败**：环境变量 `FEISHU_WEBHOOK_URL` 未配置，无法推送。请先在部署环境中配置好飞书机器人的 webhook 地址（例如在 `.env` 或部署配置中设置 `FEISHU_WEBHOOK_URL=htt…`

**task-006** (factual)
- query: Windows 开发环境下 Agent 的对话状态为什么重启后会丢？
- trajectory: search_memories_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-007** (no_tool)
- query: 好的，谢谢！
- trajectory: - (n=0, steps=1, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=34)
- answer: `不客气！如果还有其他关于项目历史、代码或团队知识的问题，随时找我。😊…`

**task-008** (multi_retrieve)
- query: 怎么把本地仓库的提交历史导入 EMA？另外普通文档分块是怎么做的？
- trajectory: retrieve_chunks_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

</details>
