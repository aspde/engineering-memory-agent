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
执行器 `make_tool_selector` 直接驱动 `agent.nodes.call_llm_node`（完整工具表），
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
  --report-md docs/interview/llm-eval-report.md

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
  llm_ground_truth.py   # 四套标注集 + validate_llm_dataset()
  llm_metrics.py        # 纯函数指标（无 I/O，单测覆盖）
  llm_executors.py      # 默认执行器：包装 call_llm_node / extract_memory / 答案 prompt / e2e 检索
  llm_judge.py          # LLM-as-judge：答案覆盖/忠实 + 摘要忠实/完整
  llm_runner.py         # 每套件一个 run_*：执行 + 聚合 + 错误行归零
  llm_report.py         # Markdown + JSON 报告 + summarize 一行
  e2e_seed.py           # e2e 语料 seeding CLI（--clear / --dry-run，独立标签）
  run_llm_eval.py       # CLI：--suite（逗号分隔）/ --judge / --sample / --min-*
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

- `ci.yml`：每次 push 跑 `--validate-only`（零成本门禁，与检索数据集校验并列）。
- `eval.yml`：
  - `llm-eval` job：每周定时 + 手动触发，需要 `LLM_API_KEY` secret，跑
    `--suite tool_selection,extraction,answer`（三个无 DB 套件）并上传报告。
  - `e2e-eval` job：同样每周定时 + 手动触发，带 postgres service + BGE-M3 模型，
    `e2e_seed --clear` 后跑 `--suite e2e`，门禁 `--min-context-recall 0.90
    --min-fact-coverage 0.70 --min-groundedness 0.80 --min-citation-rate 0.80`。
  - 门禁阈值均为保守起点，首次报告落地后按基线校准
    （见 workflow 注释与本节上方的 `--min-*` 示例）。
