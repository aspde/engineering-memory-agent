# EMA Task-Level E2E Evaluation Report

- Generated: 2026-08-24 00:06:15 UTC
- Suites: 1
- Tasks: 8
- Execution errors: 0
- Judge degradations: 0

## Overall

### 任务级端到端

| metric | completed | tool_recall | unexpected_rate | within_budget | n_steps | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.250 | 0.938 | 0.375 | 1.000 | 1.875 | 0.375 | 1.000 | 0.000 | 0.500 |


## 任务级端到端 by category

| category | completed | tool_recall | unexpected_rate | within_budget | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|
| factual | 0.000 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 | 0.000 | 0.500 |
| multi_retrieve | 0.500 | 1.000 | 0.000 | 1.000 | 0.500 | 1.000 | 0.000 | 0.500 |
| write | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| conceptual | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| notify | 0.000 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| no_tool | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

<details><summary>Per-task detail (任务级端到端)</summary>

**task-001** (factual)
- query: 向量检索后端选了什么方案，为什么不用 Elasticsearch？
- trajectory: search_memories_tool, query_entity_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=333)
- answer: `向量检索后端选用了 **PostgreSQL 的 pgvector 扩展**，而不是 Elasticsearch（memory: 762623dc、1052118b）。

**不用 Elasticsearch 的原因：**

1. **与业…`

**task-002** (multi_retrieve)
- query: 线上 502 事故的根因是什么？另外长代码文件分块时会怎么处理？两个都回答。
- trajectory: search_memories_tool, retrieve_chunks_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-003** (write)
- query: 为什么嵌入模型选 BGE-M3？然后把『嵌入模型用 BGE-M3，1024 维，本地推理』记成一条记忆。
- trajectory: search_memories_tool, query_entity_tool, write_memory_tool (n=3, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-004** (conceptual)
- query: 之前出过什么问题，会不会陷入死循环？怎么避免？
- trajectory: query_rewrite_and_search_tool, search_memories_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-005** (notify)
- query: 怎么把本地仓库的提交历史导入 EMA？然后把方法飞书通知给团队。
- trajectory: notify_feishu_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-006** (factual)
- query: Windows 开发环境下 Agent 的对话状态为什么重启后会丢？
- trajectory: search_memories_tool, query_rewrite_and_search_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-007** (no_tool)
- query: 好的，谢谢！
- trajectory: - (n=0, steps=1, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=105)
- answer: `不客气！😊

如果你之后有需要，随时可以找我：
- 查询项目历史、技术决策、过往 bug 根因
- 搜索代码库或文档内容
- 摄入新的 Git 仓库、文档到知识库
- 推送巡检结果或告警到飞书群

祝工作顺利！…`

**task-008** (multi_retrieve)
- query: 怎么把本地仓库的提交历史导入 EMA？另外普通文档分块是怎么做的？
- trajectory: search_memories_tool, retrieve_chunks_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=575)
- answer: `两个问题分别回答：

## 1. 导入本地 Git 仓库提交历史

使用 `ingest_git_repo_tool` 即可，每条 commit 会变成一条结构化记忆（含作者、提交信息、diff 上下文）（memory: ce225d27）…`

</details>
