# EMA LLM Behavior Evaluation Report

- Generated: 2026-08-09 07:09:55 UTC
- Suites: 4
- Items: 39
- Execution errors: 0
- Judge degradations: 0

## Overall

### 工具选择

| metric | tool_accuracy | expected_recall | unexpected_rate | no_call | arg_match_rate |
|---|---|---|---|---|---|
| **overall** | 0.667 | 0.867 | 0.200 | 0.133 | 0.933 |

### 知识抽取 (LLM judge)

| metric | entity_precision | entity_recall | entity_f1 | entity_type_accuracy | relation_precision | relation_recall | relation_f1 | summary_coverage | summary_faithfulness | summary_completeness |
|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.710 | 0.760 | 0.709 | 0.812 | 0.204 | 0.344 | 0.237 | 0.938 | 0.975 | 0.963 |

### 最终答案 (LLM judge)

| metric | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|
| **overall** | 0.958 | 0.875 | 0.125 | 1.000 |

### 端到端问答 (LLM judge)

| metric | context_recall | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|
| **overall** | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |


## 工具选择 by category

| category | tool_accuracy | expected_recall | unexpected_rate | no_call | arg_match_rate |
|---|---|---|---|---|---|
| memory_search | 0.667 | 1.000 | 0.333 | 0.000 | 1.000 |
| doc_search | 0.333 | 1.000 | 0.667 | 0.000 | 1.000 |
| write | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| ingest | 0.500 | 0.500 | 0.000 | 0.500 | 0.500 |
| entity | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| extract | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| notify | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| no_tool | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |

## 知识抽取 by category

| category | entity_precision | entity_recall | entity_f1 | entity_type_accuracy | relation_precision | relation_recall | relation_f1 | summary_coverage | summary_faithfulness | summary_completeness |
|---|---|---|---|---|---|---|---|---|---|---|
| code_decision | 1.000 | 0.750 | 0.833 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 | 1.000 |
| incident | 0.417 | 0.500 | 0.452 | 0.750 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| architecture | 0.625 | 1.000 | 0.762 | 0.833 | 0.267 | 0.750 | 0.367 | 0.750 | 1.000 | 1.000 |
| process | 0.800 | 0.792 | 0.788 | 0.667 | 0.300 | 0.375 | 0.333 | 1.000 | 0.900 | 0.850 |

## 最终答案 by category

| category | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|
| factual | 1.000 | 1.000 | 0.000 | 1.000 |
| causal | 0.833 | 0.500 | 0.500 | 1.000 |
| instruction | 1.000 | 1.000 | 0.000 | 1.000 |
| negation | 1.000 | 1.000 | 0.000 | 1.000 |

## 端到端问答 by category

| category | context_recall | fact_coverage | groundedness | hallucination_rate | citation_rate |
|---|---|---|---|---|---|
| factual | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| causal | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| instruction | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| negation | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |

<details><summary>Per-query detail (tool_selection)</summary>

**tsel-001** (memory_search)
- query: 之前 koa-connect 的 ctx 泄漏问题最后是怎么解决的？
- called: search_memories_tool, query_entity_tool (accuracy=0.000)

**tsel-002** (memory_search)
- query: 飞书群里讨论过内存泄漏的问题吗？
- called: search_memories_tool, search_memories_tool (accuracy=1.000)

**tsel-003** (memory_search)
- query: 我们仓库的 git 历史里有没有关于 pgvector 索引的提交？
- called: search_memories_tool, search_memories_tool (accuracy=1.000)

**tsel-004** (doc_search)
- query: 查一下 Postman 接口文档里关于认证鉴权的说明
- called: search_memories_tool, retrieve_chunks_tool (accuracy=0.000)

**tsel-005** (doc_search)
- query: 这个项目 README 里怎么写的部署步骤？
- called: retrieve_chunks_tool, search_memories_tool (accuracy=0.000)

**tsel-006** (doc_search)
- query: 之前出过什么问题，会不会陷入死循环？
- called: query_rewrite_and_search_tool, search_memories_tool (accuracy=1.000)

**tsel-007** (write)
- query: 记录一下：我们决定下周把 CI 从 GitHub Actions 迁到自建 runner
- called: write_memory_tool (accuracy=1.000)

**tsel-008** (write)
- query: 帮我记住这个结论：embedding 维度必须统一为 1024
- called: write_memory_tool (accuracy=1.000)

**tsel-009** (ingest)
- query: 把 G:\Projects\ema 的 git 历史导入进来
- called: ingest_git_repo_tool (accuracy=1.000)

**tsel-010** (ingest)
- query: 帮我把这份接口文档索引进来，document_id 是 api-doc
- called: - (accuracy=0.000)

**tsel-011** (extract)
- query: 分析一下这段文本里提到了哪些技术和人：我们用了 LangGraph 编排，由张工负责。
- called: - (accuracy=0.000)

**tsel-012** (entity)
- query: PostgreSQL 这个实体在系统里关联了哪些记忆？
- called: query_entity_tool (accuracy=1.000)

**tsel-013** (notify)
- query: 上线前飞书通知一下大家 3 点发布
- called: notify_feishu_tool (accuracy=1.000)

**tsel-014** (no_tool)
- query: 你好
- called: - (accuracy=1.000)

**tsel-015** (no_tool)
- query: 谢谢，我知道了
- called: - (accuracy=1.000)

</details>

<details><summary>Per-query detail (extraction)</summary>

**ext-001** (code_decision)
- entities: precision=1.000 recall=1.000 f1=1.000 (n=3)
- relations: precision=0.500 recall=0.500 f1=0.500 (n=2)
- summary coverage=1.000
- judge: faithfulness=1.000 completeness=1.000

**ext-002** (incident)
- entities: precision=0.333 recall=0.333 f1=0.333 (n=3)
- relations: precision=0.000 recall=0.000 f1=0.000 (n=2)
- summary coverage=1.000
- judge: faithfulness=1.000 completeness=1.000

**ext-003** (architecture)
- entities: precision=0.500 recall=1.000 f1=0.667 (n=6)
- relations: precision=0.200 recall=1.000 f1=0.333 (n=5)
- summary coverage=1.000
- judge: faithfulness=1.000 completeness=1.000

**ext-004** (process)
- entities: precision=1.000 recall=0.833 f1=0.909 (n=5)
- relations: precision=0.600 recall=0.750 f1=0.667 (n=5)
- summary coverage=1.000
- judge: faithfulness=1.000 completeness=1.000

**ext-005** (code_decision)
- entities: precision=1.000 recall=0.500 f1=0.667 (n=1)
- relations: precision=0.000 recall=0.000 f1=0.000 (n=0)
- summary coverage=1.000
- judge: faithfulness=1.000 completeness=1.000

**ext-006** (incident)
- entities: precision=0.500 recall=0.667 f1=0.571 (n=4)
- relations: precision=0.000 recall=0.000 f1=0.000 (n=2)
- summary coverage=1.000
- judge: faithfulness=1.000 completeness=1.000

**ext-007** (architecture)
- entities: precision=0.750 recall=1.000 f1=0.857 (n=4)
- relations: precision=0.333 recall=0.500 f1=0.400 (n=3)
- summary coverage=0.500
- judge: faithfulness=1.000 completeness=1.000

**ext-008** (process)
- entities: precision=0.600 recall=0.750 f1=0.667 (n=5)
- relations: precision=0.000 recall=0.000 f1=0.000 (n=4)
- summary coverage=1.000
- judge: faithfulness=0.800 completeness=0.700

</details>

<details><summary>Per-query detail (answer)</summary>

**ans-001** (factual)
- query: 我们向量检索后端最终选型是什么？为什么？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=139)
- answer: `向量检索后端最终选型为 PostgreSQL 的 pgvector 扩展，而非 Elasticsearch。（memory: c4a11b2e）

原因：
- pgvector 与业务库同库，事务一致性有保障
- 原生支持 cosine 距…`

**ans-002** (causal)
- query: 为什么线上会突然 502？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=187)
- answer: `根据已有的事故复盘记忆（memory: 9f2d8a01），线上 502 的根因是 **数据库连接池被占满**。

触发链路如下：

1. 流量上涨
2. 连接池无空闲连接
3. 新请求排队超时
4. 网关返回 502

当时的修复措施是重…`

**ans-003** (negation)
- query: Windows 下 Agent 的对话状态能跨重启保存吗？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=164)
- answer: `不能。在 Windows 开发环境下，持久化方案已降级为 InMemorySaver，重启后 Agent 状态会丢失（memory: 51b7e3c9）。原因是 AsyncPostgresSaver 因 psycopg3 异步实现兼容性问题…`

**ans-004** (instruction)
- query: 怎样把本地仓库的提交历史导入 EMA？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=190)
- answer: `你可以使用 EMA 的 `ingest_git_repo_tool` 从本地 Git 仓库读取提交历史并写入记忆。参数说明：

- `repo_path`：仓库的绝对路径（必填）
- `max_commits`：最多摄入的提交数，默认 50…`

**ans-005** (factual)
- query: 嵌入模型选的是什么？为什么不用 OpenAI 的？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=134)
- answer: `嵌入模型选用的是 **BGE-M3**（1024 维，支持中英双语，本地推理零 API 成本）；对比 OpenAI 的 text-embedding-3-large，准确率相当，但成本和延迟更优，所以没有采用 OpenAI 的方案。（mem…`

**ans-006** (causal)
- query: 为什么开发环境要用 InMemorySaver？
- coverage=0.667 grounded=0.000 citation=1.000 (answer len=200)
- answer: `根据已有记忆（memory: 3e6b2f8c），开发环境使用 InMemorySaver 是因为 **psycopg3 的异步实现与 Windows 的 ProactorEventLoop 存在冲突**，导致 AsyncPostgresS…`
- ⚠ ungrounded: ["答案中提到 '事件循环'（或其同义表述）并非必需事实列表中的直接原文，但核心概念（ProactorEventLoop 作为事件循环的一种）与上下文一致，且答案未捏造额外事实。然而，必需事实列表要求原文 '事件循环' 出现，而答案使用 'ProactorEventLoop' 作为具体实例，这在严格评测中可能被视为未完全覆盖 '事件循环' 这一通用术语。因此，基于严格标准，答案未能完全覆盖所有必需事实，故 grounded 为 false。"]

**ans-007** (factual)
- query: Agent 一轮最多能执行几步工具调用？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=203)
- answer: `根据知识库记忆（memory: b1a5c7e3）：

- Agent 通过 **max_agent_steps** 限制每轮最多执行的 LLM 调用次数，达到上限后会强制进入最终回答节点，防止工具循环失控。
- 默认值由配置项 **MAX…`

**ans-008** (instruction)
- query: 怎么记录一条新的长期记忆？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=199)
- answer: `当您告诉我需要记住某件事时，我会立即调用 `write_memory_tool`，并且直接使用您的原话作为 `content` 参数来写入新的长期记忆（memory: e2d9f4b6）。写入时不需要我预先判断是否存在冲突——工具内置了冲突…`

</details>

<details><summary>Per-query detail (e2e)</summary>

**e2e-001** (factual)
- query: 向量检索后端选了什么方案，为什么不用 Elasticsearch？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=197)
- context_recall=1.000 (n_retrieved=5)
- answer: `向量检索后端最终选用了 **PostgreSQL 的 pgvector 扩展**（memory: f14ccb00）。

不用 Elasticsearch 的原因有三点：

1. **事务一致性**：pgvector 与业务库同库，事务一致…`

**e2e-002** (causal)
- query: 为什么线上会突然返回 502？根因是什么？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=209)
- context_recall=1.000 (n_retrieved=5)
- answer: `线上 502 的直接根因是**数据库连接池被占满**（memory: 1522e1b1）。

完整触发链路如下：

1. 流量上涨
2. 连接池无空闲连接可用
3. 新请求排队等待，最终超时
4. 网关在超时后返回 502

当时采取的修复…`

**e2e-003** (causal)
- query: Agent 编排为什么选 LangGraph 而不是 LangChain Agent？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=338)
- context_recall=1.000 (n_retrieved=5)
- answer: `根据知识库记录，Agent 编排框架选用 LangGraph 而非 LangChain Agent，核心原因是：

1. **StateGraph 显式声明节点与边**：图结构清晰可控，路由逻辑透明，便于调试和扩展。
2. **支持 int…`

**e2e-004** (negation)
- query: Windows 开发环境下 Agent 的对话状态为什么重启后会丢？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=291)
- context_recall=1.000 (n_retrieved=5)
- answer: `Windows 开发环境下 Agent 对话状态重启后丢失，是因为持久化被降级了（memory: d444d2d9）：

- 根因：Windows 上 `AsyncPostgresSaver` 因 psycopg3 异步实现与 `Proac…`

**e2e-005** (instruction)
- query: 怎么把本地仓库的提交历史导入 EMA？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=300)
- context_recall=1.000 (n_retrieved=5)
- answer: `可以通过 EMA 的 `ingest_git_repo_tool` 将本地 Git 仓库的提交历史导入知识库（memory: 526c4b85）。

参数如下：

- `repo_path`：仓库的绝对路径（必填）
- `max_commi…`

**e2e-006** (factual)
- query: 嵌入模型用的是什么，为什么不用 OpenAI 的？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=162)
- context_recall=1.000 (n_retrieved=5)
- answer: `嵌入模型选用的是 **BGE-M3**（1024 维，支持中英双语，本地推理）。

之所以不用 OpenAI 的 text-embedding-3-large，是因为对比测试中两者准确率相当，但 BGE-M3 在成本和延迟上更优：本地推理零…`

**e2e-007** (factual)
- query: Agent 是怎么防止工具调用失控的？一轮最多执行几步？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=362)
- context_recall=1.000 (n_retrieved=5)
- answer: `Agent 通过 **max_agent_steps 循环防护** 来防止工具调用失控（memory: 68080973）：
- 该限制约束**每轮最多执行的 LLM 调用次数**（即 Agent 在单轮内能调用工具的步数上限）。
- 达到…`

**e2e-008** (instruction)
- query: 文档分块是怎么做的？长代码文件会被切坏吗？
- coverage=1.000 grounded=1.000 citation=1.000 (answer len=207)
- context_recall=1.000 (n_retrieved=5)
- answer: `文档分块策略分为两种（document: ema-e2e-seed）：

1. **普通文档**：用递归分隔符按自然边界切块，即段落、行、句。
2. **代码文件**：用 AST 解析按函数/类边界切分。

长代码文件**不会被切坏**。因…`

</details>
