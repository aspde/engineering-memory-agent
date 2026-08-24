# EMA Task-Level E2E Evaluation Report

- Generated: 2026-08-23 23:51:24 UTC
- Suites: 1
- Tasks: 8
- Execution errors: 0
- Judge degradations: 0

## Overall

### 任务级端到端

| metric | completed | tool_recall | unexpected_rate | within_budget | n_steps | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.375 | 0.875 | 0.375 | 1.000 | 2.000 | 0.812 | 1.000 | 0.000 | 1.000 |


## 任务级端到端 by category

| category | completed | tool_recall | unexpected_rate | within_budget | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|
| factual | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| multi_retrieve | 0.500 | 0.750 | 0.000 | 1.000 | 0.750 | 1.000 | 0.000 | 1.000 |
| write | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| conceptual | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| notify | 0.000 | 0.500 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 |
| no_tool | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

<details><summary>Per-task detail (任务级端到端)</summary>

**task-001** (factual)
- query: 向量检索后端选了什么方案，为什么不用 Elasticsearch？
- trajectory: search_memories_tool, query_entity_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=382)
- answer: `向量检索后端选了 **PostgreSQL 的 pgvector 扩展**，而不是 Elasticsearch（memory: 762623dc、1052118b）。

选 pgvector 的原因主要有三点：

1. **与业务库同库**…`

**task-002** (multi_retrieve)
- query: 线上 502 事故的根因是什么？另外长代码文件分块时会怎么处理？两个都回答。
- trajectory: search_memories_tool, retrieve_chunks_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=324)
- answer: `**1. 线上 502 事故根因**

根因是数据库连接池被占满。触发链路：流量上涨 → 连接池无空闲连接 → 新请求排队超时 → 网关返回 502。修复措施是重启连接池，并增加了连接泄漏监控（memory: 362654b2）。

**2…`

**task-003** (write)
- query: 为什么嵌入模型选 BGE-M3？然后把『嵌入模型用 BGE-M3，1024 维，本地推理』记成一条记忆。
- trajectory: search_memories_tool, retrieve_chunks_tool, write_memory_tool (n=3, steps=3, completed=0.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=570)
- answer: `**为什么选 BGE-M3？**

根据知识库记录，决策依据主要有三点：

1. **零 API 成本**：本地推理不产生 API 费用，适合 EMA 高频写入场景（document: ema-eval-seed）。
2. **效果接近闭源…`

**task-004** (conceptual)
- query: 之前出过什么问题，会不会陷入死循环？怎么避免？
- trajectory: query_rewrite_and_search_tool, search_memories_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=741)
- answer: `根据知识库记录，整理如下：

## 曾经出过的问题

1. **Agent 工具调用死循环**（memory: 165de545）
   早期未限制步数时，Agent 陷入 `search_memory → 无结果 → 再 search_m…`

**task-005** (notify)
- query: 怎么把本地仓库的提交历史导入 EMA？然后把方法飞书通知给团队。
- trajectory: notify_feishu_tool (n=1, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=1.000 (answer len=337)
- answer: `收到，情况确认如下：

**当前状态**
- 飞书通知发送失败，原因：`FEISHU_WEBHOOK_URL` 未配置
- 导入方法说明已完成，内容本身没有问题

**接下来需要做的**

1. **配置飞书 Webhook（需要管理员操作…`

**task-006** (factual)
- query: Windows 开发环境下 Agent 的对话状态为什么重启后会丢？
- trajectory: search_memories_tool, retrieve_chunks_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=716)
- answer: `原因是一条清晰的因果链：

**根因**：Windows 默认使用 ProactorEventLoop，而 psycopg3 的异步模式与之不兼容——连接池初始化时 `_conn_pool.setup` 抛 RuntimeError，导致 …`

**task-007** (no_tool)
- query: 好的，谢谢！
- trajectory: - (n=0, steps=1, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=94)
- answer: `不客气！😊 

不过这似乎是我们对话的开始——如果您有任何关于项目历史、技术决策、代码库、CI 构建记录或文档检索方面的问题，随时告诉我，我可以帮您搜索工程记忆库。

有什么需要帮忙的吗？…`

**task-008** (multi_retrieve)
- query: 怎么把本地仓库的提交历史导入 EMA？另外普通文档分块是怎么做的？
- trajectory: retrieve_chunks_tool, retrieve_chunks_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=0.500 grounded=1.000 citation=1.000 (answer len=652)
- answer: `## 1. 导入本地仓库的提交历史

直接把**仓库路径**发给我即可（例如 `/home/user/projects/my-repo`），我会调用 Git 历史摄入功能，把提交记录转成结构化记忆写入知识库，之后就能按来源检索。

补充一点…`

</details>
