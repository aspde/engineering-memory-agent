# EMA Task-Level E2E Evaluation Report

- Generated: 2026-08-24 00:56:32 UTC
- Suites: 1
- Tasks: 8
- Execution errors: 0
- Judge degradations: 0

## Overall

### 任务级端到端

| metric | completed | tool_recall | unexpected_rate | within_budget | n_steps | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.375 | 0.688 | 0.000 | 1.000 | 1.875 | 0.438 | 1.000 | 0.000 | 0.750 |


## 任务级端到端 by category

| category | completed | tool_recall | unexpected_rate | within_budget | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|
| factual | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| multi_retrieve | 0.500 | 0.500 | 0.000 | 1.000 | 0.750 | 1.000 | 0.000 | 1.000 |
| write | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| conceptual | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| notify | 0.000 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| no_tool | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

<details><summary>Per-task detail (任务级端到端)</summary>

**task-001** (factual)
- query: 向量检索后端选了什么方案，为什么不用 Elasticsearch？
- trajectory: search_memories_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-002** (multi_retrieve)
- query: 线上 502 事故的根因是什么？另外长代码文件分块时会怎么处理？两个都回答。
- trajectory: search_memories_tool, retrieve_chunks_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=323)
- answer: `两个问题的答案如下：

## 1. 线上 502 事故根因

**根因是数据库连接池被占满**（memory: 2ad57bf5）。触发链路：

流量上涨 → 连接池无空闲连接 → 新请求排队超时 → 网关返回 502

当时的修复措施是重…`

**task-003** (write)
- query: 为什么嵌入模型选 BGE-M3？然后把『嵌入模型用 BGE-M3，1024 维，本地推理』记成一条记忆。
- trajectory: search_memories_tool, write_memory_tool (n=2, steps=3, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=484)
- answer: `**为什么选 BGE-M3？**

根据已有记忆（memory: a9870a39），选型理由主要有三点：

1. **中英双语支持** —— 适合团队的中文为主的工程语境；
2. **本地推理零 API 成本** —— 不依赖外部嵌入服务…`

**task-004** (conceptual)
- query: 之前出过什么问题，会不会陷入死循环？怎么避免？
- trajectory: query_rewrite_and_search_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=556)
- answer: `出过。根据知识库里的故障复盘（document: ema-eval-seed）：

**当时的问题**
- 早期 Agent 未限制步数，当 `search_memory` 返回空结果时，会反复调用同一个 tool
- 单次对话最多消耗 2…`

**task-005** (notify)
- query: 怎么把本地仓库的提交历史导入 EMA？然后把方法飞书通知给团队。
- trajectory: notify_feishu_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-006** (factual)
- query: Windows 开发环境下 Agent 的对话状态为什么重启后会丢？
- trajectory: search_memories_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=19)
- answer: `抱歉，生成回复时出现错误，请稍后重试。…`

**task-007** (no_tool)
- query: 好的，谢谢！
- trajectory: - (n=0, steps=1, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=36)
- answer: `不客气！如果还有其他关于项目历史、技术决策或代码库的问题，随时告诉我。😊…`

**task-008** (multi_retrieve)
- query: 怎么把本地仓库的提交历史导入 EMA？另外普通文档分块是怎么做的？
- trajectory: - (n=0, steps=1, completed=0.000, within_budget=1.000)
- coverage=0.500 grounded=1.000 citation=1.000 (answer len=531)
- answer: `这两个问题我可以直接回答，不需要检索团队记忆：

**1. 导入本地仓库的提交历史**

使用 Git 历史摄入功能，提供仓库的绝对路径即可。参数：

- **repo_path**（必填）：本地 Git 仓库的绝对路径
- **max_c…`

</details>
