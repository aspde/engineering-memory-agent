# LLM 行为评测（工具选择 / 知识抽取 / 最终答案 / 端到端）

> 评测代码在 `tests/eval/` 下的 `llm_*` 模块，CLI 入口是 `python -m tests.eval.run_llm_eval`。

## 为什么需要它

检索评测（`run_eval.py`）只能回答"搜回来的记忆对不对"（Recall@5 / MRR / NDCG）。
它回答不了几个真正决定 Agent 行为质量的问题：

1. **工具选择**：模型该调 `search_memories_tool` 时有没有调？该保持沉默时有没有乱调？
2. **知识抽取**：`extract_memory` 抽出的实体、关系对不对？摘要是否忠实、完整？
3. **最终答案**：答案是否覆盖了检索上下文里的关键事实、有没有捏造上下文没有的内容？
4. **端到端**：用户真实提问时，EMA 是否检索到了对的上下文、并据此产出忠于上下文的答案？

`llm_*` 评测用真实 LLM 跑真实生产路径（真实 system prompt、真实 tool schema、
真实的 `extract_memory`、真实的检索链路），对这几个维度给出可量化、可回归的指标。

## 四个套件

### 工具选择（tool_selection）

标注集：`查询 → 必须调用的工具 + 禁止调用的工具 + 可接受的替代工具 + 可选参数约束`。
执行器 `make_tool_selector` 直接驱动 `backend.agent.nodes.call_llm_node`（完整工具表），
只测"决策"，不真正执行工具。

指标（每条查询）：

| 指标 | 含义 |
|------|------|
| `tool_accuracy` | 严格正确：必备工具全调、无意外调用、无禁用调用；`expected_tools` 为空时要求完全不动手 |
| `expected_recall` | 必备工具被调用的比例（部分给分） |
| `unexpected_rate` | 是否调了预期/允许之外的工具 |
| `no_call` | 该调工具却一个没调 |
| `arg_match_rate` | 参数约束子串在 args JSON 中的命中比例 |

### 知识抽取（extraction）

标注集：`源文本 → 期望实体(name+type) + 期望关系(from,to,type) + 摘要关键词`。
执行器 `make_extractor` 跑真实的 `extract_memory`（summary → entities → relations）。

指标（每条查询）：

| 指标 | 含义 |
|------|------|
| `entity_precision` / `entity_recall` / `entity_f1` | 实体匹配（名称归一化 + 容错包含匹配） |
| `entity_type_accuracy` | 匹配成功的实体中类型正确的比例 |
| `relation_precision` / `relation_recall` / `relation_f1` | 三元组精确匹配 |
| `summary_coverage` | 期望关键词在摘要中的命中比例（确定性代理指标） |
| `summary_faithfulness` / `summary_completeness` | `--judge=llm` 时由 LLM 裁判打分（0-1） |

### 最终答案（answer）

标注集：`查询 + 唯一允许使用的上下文 + 必须覆盖的事实 + 禁止出现的论断`。
执行器 `make_answer_generator` 用与 `generate_final_node` 完全一致的 prompt
（`agent.system` 模板 + `<memory>` 上下文块）生成答案，然后评判：

- **确定性通道**（`--judge=deterministic`）：子串匹配 `required_facts` 覆盖率；
  `prohibited_claims` 是否出现在答案中（出现即判 ungrounded）。
- **LLM 裁判通道**（`--judge=llm`，默认）：`llm_judge.py` 让第二个 LLM 用
  `chat_structured` 输出结构化判决（`covered_facts` / `grounded` /
  `ungrounded_claims`），语义更鲁棒，代价是每条多一次 LLM 调用。

指标：`fact_coverage`（必需事实覆盖率）、`groundedness`（是否忠实）、
`hallucination_rate`（是否出现无依据论断）、`citation_rate`（是否引用了上下文中的来源 ID）。

### 端到端问答（e2e）

answer 套件注入 golden context，回答不了"真实检索到底给了模型什么"。e2e 套件
把整条链串起来：**查询 → 真实检索 → 按生产 prompt 生成答案 → 评判**。

标注集：`查询 + 必须被检索到的 source_content + 必须覆盖的事实 + 禁止出现的论断 +
检索模式（memory / chunk）`。`source_content` 通过 `tests.eval.e2e_seed` 写入
memories/chunks 表（带 `eval_e2e` 标签，可 `--clear` 重建，不走 LLM 抽取，
与检索评测的 30 条语料互不污染）。

执行器 `make_e2e_runner` 跑真实生产读路径（memory 模式 → `query_memories`，
chunk 模式 → `retrieve_hybrid`），把检索结果按 `generate_final_node` 的
`<memory>/<doc>` 源标签框架组装成上下文（行内暴露记忆短 ID / document ID，
供答案引用），再用 `agent.system` 模板生成答案。

指标（每条查询）：

| 指标 | 含义 |
|------|------|
| `context_recall` | 检索到的上下文中覆盖了必需事实的比例——**检索侧边界**，确定性计算 |
| `fact_coverage` | 答案覆盖必需事实的比例（确定性子串 + LLM 裁判） |
| `groundedness` / `hallucination_rate` | 答案是否忠实于**实际检索到的上下文** |
| `citation_rate` | 答案是否引用了检索实际返回的来源 ID（非 golden ID） |

`context_recall` 与 answer 指标同表，能定位失败归属：`context_recall` 低 → 检索
没召回（生成层无法补齐）；`context_recall` 高但 `fact_coverage` 低 → 召回了但答案
没用对。8 条题目复用 EMA 自身工程史（pgvector 选型、502 复盘、InMemorySaver
降级等），其中 1 条 chunk 模式覆盖文档检索路径。

## 运行

```bash
# 只校验标注集一致性（零 LLM / 零 DB，CI 每 push 跑）
python -m tests.eval.run_llm_eval --validate-only

# 冒烟：每套件 3 条，确定性评判（最省 token）
python -m tests.eval.run_llm_eval --sample 3 --judge deterministic

# 三个无 DB 套件（CI llm-eval job 跑这个）
python -m tests.eval.run_llm_eval --suite tool_selection,extraction,answer

# 端到端套件（需要先 seeding + 本地 DB + embedding 模型）
python -m tests.eval.e2e_seed --clear
python -m tests.eval.run_llm_eval --suite e2e \
  --min-context-recall 0.90 --min-fact-coverage 0.70 \
  --min-groundedness 0.80 --min-citation-rate 0.80

# 全量 + LLM 裁判 + 报告
python -m tests.eval.run_llm_eval --suite all \
  --report-md tests/eval/reports/llm-eval-report.md

# 回归门禁（exit 2 = 指标跌破下限）
python -m tests.eval.run_llm_eval --suite all \
  --min-tool-accuracy 0.70 --min-entity-f1 0.60 --min-relation-f1 0.50 \
  --min-fact-coverage 0.60 --min-groundedness 0.80
```

成本：全套 15（工具选择）+ 8（抽取）+ 8（答案）+ 8（端到端）≈ 39 条，每条 1-5 次
LLM 调用，全量 + LLM 裁判约 80-120 次调用，适合每周定时任务。`--suite` 支持
逗号分隔（如 `tool_selection,answer`），`all` 含全部四个套件。

## 架构

```
tests/eval/
  llm_ground_truth.py   # 四套标注集的 item 类型 + validate_llm_dataset()（数据在 data/*.jsonl）
  llm_metrics.py        # 纯函数指标（无 I/O，单测覆盖）
  llm_executors.py      # 默认执行器：包装 call_llm_node / extract_memory / 答案 prompt / e2e 检索
  llm_judge.py          # LLM-as-judge：答案覆盖/忠实 + 摘要忠实/完整
  core.py               # 共用骨架（与检索/task 评测共享）：EvalResult / 聚合 / judge 失败零值 / JSON 序列化
  llm_runner.py         # 每套件一个 run_*：执行 + 聚合 + 错误行归零（结果类/聚合复用 core）
  llm_report.py         # Markdown + JSON 报告 + summarize 一行（序列化复用 core）
  e2e_seed.py           # e2e 语料 seeding CLI（--clear / --dry-run，独立标签）
  run_llm_eval.py       # CLI：--suite（逗号分隔）/ --judge / --sample / --min-*
  experiments/          # 一次性研究脚本归档（A/B、阈值标定、judge 校准、scale 探测…）：不复用主骨架、不进 CI，按需 `python -m tests.eval.experiments.<script>`
```

设计要点：

- **执行器可注入**：runner 只认 callable，单测用 fake 执行器，真实 LLM 只在
  默认执行器里触发（与检索评测的 `RetrieverAdapter` 同构）。
- **失败语义**：执行失败计入 `errors` 并作为全零行参与聚合（沿用检索 runner
  的"失败查询计入分母"策略）；`--judge=llm` 时裁判失败降级到确定性通道并计入
  `judge_errors`——评判降级不触发 CI 门禁，执行失败才触发。
- **seeding 显式**：e2e 的 `e2e_seed --clear` 是独立前置步骤，不藏在 runner 里
  （`--validate-only` 零成本不碰 DB，runner 纯编排可单测）。
- **数据维护**：标注集全部手写，`--validate-only` 校验每个 item 的内部一致性
  （工具名存在于 `ALL_TOOLS`、关系端点必须是金标准实体、上下文长度上限、
  e2e 的 `required_facts` 必须是 `source_content` 子串等），
  防止"编辑了标注却带病运行"。

## CI

- `ci.yml`：每次 push 跑 `--validate-only`（零成本门禁，与检索/LLM 数据集校验并列）。
- `eval.yml`：
  - `llm-eval` job：每周定时 + 手动触发，需要 `LLM_API_KEY` secret，跑
    `--suite tool_selection,extraction,answer`（三个无 DB 套件）并上传报告。
  - `e2e-eval` job：同样每周定时 + 手动触发，带 postgres service + BGE-M3 模型，
    `e2e_seed --clear` 后跑 `--suite e2e`。
  - `task-eval` job：同样每周定时 + 手动触发，前置与 e2e-eval 相同（postgres +
    BGE-M3 + LLM），`e2e_seed --clear` 后跑 `run_task_eval --judge deterministic`
    驱动真实 Agent 图。**门禁阈值暂未设置**——等首份
    `task-eval-report.md` 落地后按真实数字校准（参考 e2e-eval 的标定流程）。
  - 两个（llm/e2e）job 的**门禁都用 `--judge deterministic`**，阈值已按
    `tests/eval/reports/llm-eval-baseline.json`（2026-08-09，commit 4ae4848）校准，
    每个阈值低于基线 0.05-0.10 留噪声余量。门禁退出码：执行失败或指标跌破阈值
    为非零，CI 即红。

### 基线

- **基线文件**：`tests/eval/reports/llm-eval-baseline.json`——一次干净运行
  （0 执行错误、0 judge 降级）的逐指标结果，提交进仓库，不被每次 run 覆盖。
- **语义基线**：`tests/eval/reports/llm-eval-semantic-baseline.json`——judge 通道稳定后
  用 `--judge llm` 跑出的语义判定结果（groundedness / hallucination_rate /
  summary_faithfulness / summary_completeness），供手动分析；不进 CI 门禁。
- **对比**：`python -m tests.eval.experiments.compare_baseline` 把当前报告与基线做 diff，
  输出逐指标 delta，任何跌破 `--tolerance`（默认 0.01，吸收 ~±0.001 的
  运行间噪声）的下降都以非零退出码标红。**每次改 prompt 或模型后跑一次**，
  用 delta 判断该改动是提升还是回归。确定性报告与 LLM-judge 报告混比会被
  拒绝（judge 模式不匹配时脚本明确报错，避免把语义判定的更严当成回归）；
  语义对比用 `--baseline tests/eval/reports/llm-eval-semantic-baseline.json`。
- **重标定**：有意的行为变更（prompt 版本号 bump、模型切换、工具表调整）落地后，
  重新生成基线并同步 eval.yml 阈值。

### 为什么门禁不用 LLM judge

`--judge llm` 提供语义 groundedness 判定（子串匹配检测不到的捏造），但它依赖
judge LLM 的可用性。2026-08-09 标定过程中，配置的 judge provider
（glm-4.7-flash）在两次全量 run 中均被持续 429 限流（7/8、6/8+8/8 的
answer/e2e judge 调用失败），LLM 判定指标被归零、不可信。门禁若依赖这种通道，
judge 一抖动整个 job 就误红——标定也就失去意义。因此 **CI 门禁走确定性通道**
（可复现、judge 故障时不会拖垮 job）；`--judge llm` 保留为手动语义分析手段
（需配置完整的 `LLM_JUDGE_*` 块，`run_llm_eval` 在缺配置时会拒绝自判），
结果与语义基线对比。

> 注：judge 通道已稳定（2026-08-09 换用 `mimo-v2.5-free`，全量 0 judge 降级），
> 语义基线已固化，但门禁仍保持 deterministic——门禁的职责是稳定抓回归，语义评测
> 归手动。

---

## 任务级端到端评测（task_eval）

> 评测代码在 `tests/eval/` 下的 `task_*` 模块，CLI 入口是 `python -m tests.eval.run_task_eval`。

### 和上面四个套件的区别

上面的 `llm_*` 套件各自测一个维度，但**没有一个驱动 Agent 图本身**——`e2e` 套件
仍是"一次检索 + 一次回答"，没有 ReAct 循环、没有真实工具执行、没有 HITL 门。
`task_eval` 补的正是这个缺口：**让真实的 `build_agent_graph`（完整工具表、真实
ToolNode、真实审批/冲突中断）去完成多步任务**，测的不是单次决策，而是一整条轨迹。

关键设计：HITL 门**自动放行**（审批通过、冲突 keep_existing）——这样测的是
"如果人总是同意，Agent 能不能把任务做完"，把人的决策从 Agent 能力里隔离出来。
Auto-memory 在评测进程中关闭，避免后台抽取/写入污染语料、烧 token。

### 标注集

`task_ground_truth.py` 的 8 个任务，事实都指向 `e2e_seed` 语料（`e2e_seed --clear`
先灌库），所以每条事实都可检索。三类任务：

| 类型 | 任务 | 验证什么 |
|------|------|---------|
| 多源检索 | task-002/008（根因在 memories、AST 分块在 chunks，一次检索拿不全） | 必须**两个工具都调**才能答全，测多步轨迹 |
| 检索 + 写 / 通知 | task-003/005（`write_memory_tool` / `notify_feishu_tool`） | 检索后完成副作用动作 |
| 单检索 / 概念 / 拒绝 | task-001/004/006/007 | 循环基线、概念查询、no_tool 克制 |

校验器强制：非 no_tool 任务必须有 required_facts，且每条事实必须是某条 e2e seed
`source_content` 的子串——否则事实永远检索不到，`fact_coverage` 结构性 <1.0。

### 指标（task_metrics.py，纯函数）

| 指标 | 含义 |
|------|------|
| `completed` | 无错误 + 实质答案（非道歉 stub）+ 调齐 expected tools（no_tool 任务 = 一个不调） |
| `tool_recall` | expected tools 被调用的比例（部分给分） |
| `unexpected_rate` | 调了 expected ∪ allowed 之外的工具 |
| `within_budget` | 未撞 `max_steps` 强制终止（循环纪律） |
| `fact_coverage` / `groundedness` / `citation_rate` | 复用 answer 套件指标（judge 通道对 Agent 实际看到的工具上下文判定） |

### 运行

```bash
# 1. 灌 e2e 语料（记忆 + 分块）
python -m tests.eval.e2e_seed --clear
# 2. 全量跑（真实 LLM，默认 --judge llm）
python -m tests.eval.run_task_eval --report-md tests/eval/reports/task_eval_report.md
# 3. 免 judge 通道（CI 用，更便宜）
python -m tests.eval.run_task_eval --judge deterministic
# 4. 零成本校验（ci.yml 每 push 跑）
python -m tests.eval.run_task_eval --validate-only
```

### 实测结果

实测数字（2026-08-09 baseline 与 2026-08-11 锁死 LLM rerank 复测）见
[task_eval 评测报告](../../tests/eval/reports/task_eval_report.md)。

### 顺带修掉的一个生产 bug

为 task_eval 写拒绝路径单测时发现：**拒绝审批后写操作仍然执行**。根因是
LangGraph 1.2.10 在 resume 被 `interrupt()` 暂停的节点时，会同时走节点返回的
`Command(goto=...)` **和**该节点的静态/条件边——EMA 的 `check_approval` 同时有
条件边（`_route_after_approval → tools`）和返回 `Command`，批准/放行路径两者恰好
都指向 tools（无害），但**拒绝路径返回 `Command(goto="call_llm")` 时，静态边仍把
路由拉到 tools，ToolNode 执行了刚被拒绝的 tool_calls**——审批门对拒绝路径形同虚设。

修复（`backend/agent/graph.py`）：`check_approval` 每条路径都返回 `Command`，因此删掉它的
静态条件边，让 Command 成为唯一路由机制。回归测试
`test_agent_graph.test_rejected_approval_does_not_execute_tool` 用真实图驱动拒绝
路径并断言写操作未执行。这个 bug 暴露了 task_eval 的真正价值：**它能抓到组件级
评测发现不了的、跨节点编排层面的正确性问题**。
