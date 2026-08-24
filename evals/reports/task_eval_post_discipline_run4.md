# EMA Task-Level E2E Evaluation Report

- Generated: 2026-08-24 02:18:10 UTC
- Suites: 1
- Tasks: 8
- Execution errors: 0
- Judge degradations: 0

## Overall

### 任务级端到端

| metric | completed | tool_recall | unexpected_rate | within_budget | n_steps | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.625 | 0.688 | 0.000 | 0.875 | 1.500 | 0.562 | 1.000 | 0.000 | 1.000 |


## 任务级端到端 by category

| category | completed | tool_recall | unexpected_rate | within_budget | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|
| factual | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 | 1.000 | 0.000 | 1.000 |
| multi_retrieve | 0.500 | 0.500 | 0.000 | 1.000 | 0.750 | 1.000 | 0.000 | 1.000 |
| write | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| conceptual | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| notify | 0.000 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| no_tool | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

<details><summary>Per-task detail (任务级端到端)</summary>

**task-001** (factual)
- query: 向量检索后端选了什么方案，为什么不用 Elasticsearch？
- trajectory: - (n=0, steps=0, completed=0.000, within_budget=0.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=0)

**task-002** (multi_retrieve)
- query: 线上 502 事故的根因是什么？另外长代码文件分块时会怎么处理？两个都回答。
- trajectory: search_memories_tool, retrieve_chunks_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=253)
- answer: `**1. 线上 502 事故根因**（memory: 29d781f3）
- 根因：**数据库连接池被占满**
- 触发链路：流量上涨 → 连接池无空闲连接 → 新请求排队超时 → 网关返回 502
- 修复：重启连接池，并增加连接泄漏监控…`

**task-003** (write)
- query: 为什么嵌入模型选 BGE-M3？然后把『嵌入模型用 BGE-M3，1024 维，本地推理』记成一条记忆。
- trajectory: search_memories_tool, write_memory_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=492)
- answer: `**为什么选 BGE-M3？**

根据知识库中的记录，选型理由是（memory: a9870a39）：

- **中英双语支持好**，契合团队的中文使用场景；
- **本地推理零 API 成本**，不依赖外部服务；
- **准确率与 Op…`

**task-004** (conceptual)
- query: 之前出过什么问题，会不会陷入死循环？怎么避免？
- trajectory: search_memories_tool, search_memories_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=707)
- answer: `结合知识库记录，分三部分回答：

## 之前出过什么问题

- **工具调用死循环（与你的问题最相关）**：Agent 早期未限制步数时陷入过死循环——`search_memory` → 无结果 → 再 `search_memory`，循环…`

**task-005** (notify)
- query: 怎么把本地仓库的提交历史导入 EMA？然后把方法飞书通知给团队。
- trajectory: notify_feishu_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=379)
- answer: `收到，推送失败的原因已确认：当前运行环境没有配置 `FEISHU_WEBHOOK_URL`，所以机器人无法发出消息。

接下来两件事可以并行推进：

**1. 飞书通知（待你配置后重试）**
- 在飞书目标群里添加自定义机器人（群设置 → …`

**task-006** (factual)
- query: Windows 开发环境下 Agent 的对话状态为什么重启后会丢？
- trajectory: search_memories_tool (n=1, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=431)
- answer: `Windows 开发环境下对话状态重启后丢失的原因（memory: eb17dac2）：

- **根因**：Windows 下使用 `AsyncPostgresSaver` 时，psycopg3 的异步模式与 Windows 的 Proa…`

**task-007** (no_tool)
- query: 好的，谢谢！
- trajectory: - (n=0, steps=1, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=37)
- answer: `不客气！如果还有其他关于项目历史、技术决策或代码文档的问题，随时告诉我 😊…`

**task-008** (multi_retrieve)
- query: 怎么把本地仓库的提交历史导入 EMA？另外普通文档分块是怎么做的？
- trajectory: - (n=0, steps=1, completed=0.000, within_budget=1.000)
- coverage=0.500 grounded=1.000 citation=1.000 (answer len=508)
- answer: `这两个问题我可以直接说明，不需要检索知识库：

## 1. 导入本地仓库的提交历史

使用 Git 历史摄入功能，把仓库路径告诉我即可，例如：

> 「帮我导入 `/home/user/projects/my-app` 这个仓库的提交历史」…`

</details>
