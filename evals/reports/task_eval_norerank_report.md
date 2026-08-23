# EMA Task-Level E2E Evaluation Report

- Generated: 2026-08-11 03:00:19 UTC
- Suites: 1
- Tasks: 8
- Execution errors: 0
- Judge degradations: 0

## Overall

### 任务级端到端

| metric | completed | tool_recall | unexpected_rate | within_budget | n_steps | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.500 | 0.812 | 0.375 | 1.000 | 2.500 | 0.750 | 1.000 | 0.000 | 0.750 |


## 任务级端到端 by category

| category | completed | tool_recall | unexpected_rate | within_budget | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|---|---|---|
| factual | 0.000 | 0.500 | 1.000 | 1.000 | 0.500 | 1.000 | 0.000 | 0.500 |
| multi_retrieve | 0.500 | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| write | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| conceptual | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| notify | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| no_tool | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

<details><summary>Per-task detail (任务级端到端)</summary>

**task-001** (factual)
- query: 向量检索后端选了什么方案，为什么不用 Elasticsearch？
- trajectory: query_rewrite_and_search_tool, retrieve_chunks_tool (n=2, steps=2, completed=0.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=404)
- answer: `向量检索后端选用了 **PostgreSQL 的 pgvector 扩展**，而非 Elasticsearch（document: ema-eval-seed）。

选型核心考量（document: ema-eval-seed）：
1. *…`

**task-002** (multi_retrieve)
- query: 线上 502 事故的根因是什么？另外长代码文件分块时会怎么处理？两个都回答。
- trajectory: search_memories_tool, search_memories_tool, retrieve_chunks_tool, search_memories_tool (n=4, steps=3, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=376)
- answer: `两个问题分别回答如下：

**1. 线上 502 事故根因**
根因是**数据库连接池被占满**。触发链路：流量上涨 → 连接池无空闲连接 → 新请求排队超时 → 网关返回 502。修复方式为重启连接池并增加连接泄漏监控（memory: 6…`

**task-003** (write)
- query: 为什么嵌入模型选 BGE-M3？然后把『嵌入模型用 BGE-M3，1024 维，本地推理』记成一条记忆。
- trajectory: search_memories_tool, write_memory_tool (n=2, steps=2, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=425)
- answer: `关于嵌入模型选型 BGE-M3，知识库里有明确决策记录，理由很清晰：

**核心理由**（memory: 9e66436e）：选用 BGE-M3，1024 维，支持中英双语，本地推理零 API 成本。对比 OpenAI text-embed…`

**task-004** (conceptual)
- query: 之前出过什么问题，会不会陷入死循环？怎么避免？
- trajectory: query_rewrite_and_search_tool, search_memories_tool, query_rewrite_and_search_tool, search_memories_tool (n=4, steps=3, completed=1.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=48)
- answer: `以上就是对这个问题的完整回答——问题确实发生过、防护已就位、后续新增功能时按同样的模式规避即可。…`

**task-005** (notify)
- query: 怎么把本地仓库的提交历史导入 EMA？然后把方法飞书通知给团队。
- trajectory: search_memories_tool, retrieve_chunks_tool, notify_feishu_tool (n=3, steps=3, completed=0.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=308)
- answer: `导入方法（来自记忆 memory: 181860f8）：

**步骤**
1. 对 EMA 说「导入 Git 仓库」，我会调用 `ingest_git_repo_tool` 读取本地仓库提交历史并写入记忆
2. 参数说明：
   - `re…`

**task-006** (factual)
- query: Windows 开发环境下 Agent 的对话状态为什么重启后会丢？
- trajectory: query_rewrite_and_search_tool, search_memories_tool, search_memories_tool (n=3, steps=3, completed=0.000, within_budget=1.000)
- coverage=0.000 grounded=1.000 citation=0.000 (answer len=421)
- answer: `通过信息收集确认了目标环境使用了PHP语言。接下来基于已知PHP通用框架结构进行静态分析和动态分析验证。

已知:composer.json在项目根目录,里面包含项目使用的依赖包信息、PHP版本约束、命名空间映射等。PHP框架的路由规则通常…`

**task-007** (no_tool)
- query: 好的，谢谢！
- trajectory: - (n=0, steps=1, completed=1.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=50)
- answer: `不客气！如果之后有任何关于项目历史、技术决策、代码问题或者需要我帮忙检索记忆库的内容，随时告诉我 😊…`

**task-008** (multi_retrieve)
- query: 怎么把本地仓库的提交历史导入 EMA？另外普通文档分块是怎么做的？
- trajectory: search_memories_tool, search_memories_tool, search_memories_tool (n=3, steps=3, completed=0.000, within_budget=1.000)
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=723)
- answer: `两个问题分别回答如下：

## 1. 导入本地 Git 提交历史

用 **`ingest_git_repo_tool`** 即可，参数：
- `repo_path`：仓库绝对路径（必填）
- `max_commits`：处理最近多少条提交…`

</details>
