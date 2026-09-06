# LLM 行为评测（工具选择 / 知识抽取 / 最终答案 / 端到端 / 写入链路）

> 评测代码在 `evals/` 下的 `llm_*` 与 `write_eval_*` 模块，CLI 入口是 `python -m evals.run_llm_eval`。

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
检索模式（memory / chunk）`。`source_content` 通过 `evals.e2e_seed` 写入
memories/chunks 表（带 `eval_e2e` 标签，可 `--clear` 重建，不走 LLM 抽取，
与检索评测的 70 条语料互不污染）。

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

### 写入链路（write_conflict / write_merge / auto_gate）

检索评测回答"搜回来的对不对"；写入链路评测回答"**存进去的对不对**"。记忆污染是
永久性的——一条捏造的合并、一次漏检的矛盾会持续恶化之后的每次检索。三个套件覆盖
写入路径上三个 correctness-critical 的 LLM 决策（gap-remediation §8 第一条）：

#### 冲突检测（write_conflict）

标注集：`已有摘要 + 新摘要 → 是否矛盾`。执行器 `make_conflict_detector` 跑真实的
`_detect_conflict`（`memory.conflict` prompt + 结构化输出），即生产写路径
0.72-0.85 相似度带里决定"进 HITL 仲裁还是当 supplement 放行"的那次调用。

指标：二分类的 confusion-cell 推导指标——`conflict_precision` / `recall` / `f1` /
`accuracy` + `false_positive_rate`（误把补充送仲裁）+ `false_negative_rate`
（漏掉真矛盾放进库）。硬负例刻意选同主题不矛盾对（细化/补充），因为"话题相同"
正是诱导模型误报矛盾的东西。确定性套件，无 LLM judge。

#### 合并质量（write_merge）

标注集：`已有摘要 + 新摘要 → 必须保留的关键事实`。执行器 `make_merge_summarizer`
跑 `merge_summaries()`——生产 merge 路径（`_merge_memory` 与冲突 resolve 的 merge
分支）共用的单一入口。

指标：`merge_fact_coverage`（双侧关键事实在合并摘要中的保留率，确定性子串）；
`--judge=llm` 时由 LLM 裁判打 `merge_faithfulness`（是否捏造）与
`merge_completeness`（是否丢事实）——正是"merge 可能捏造内容"这个风险的两个失败
方向。judge 失败沿用 extraction 策略：行标 `judge_error`、键留空，不写假 0 分。

#### 自动记忆门控（auto_gate）

标注集：`用户消息 → 是否持久知识`。执行器 `make_gate_checker` 跑
`_llm_gate_verdict`（`agent.auto_memory_gate` prompt）——**故意绕过 fail-open
包装**：生产包装层在 provider 故障时默认放行（对话路径正确行为），但评测里故障
必须记为执行错误而不是"全部 worthy"的完美分数。拆分出的 raising 核让两种语义各得其所。

指标：worthy 二分类的 precision / recall / f1 / accuracy + 双向错误率。
门控设计上偏保守（漏捕获可恢复，垃圾记忆永久污染），所以 `false_positive_rate`
（垃圾放行率）是最受关注的列。确定性套件。

#### 为什么二分类指标从混淆单元推导

真负例条目的逐条 precision 是 0/0 → 按约定记 0.0，直接平均会把含负例的完美分类
拖到 0.5。runner 记录逐条 tp/fp/fn/tn 指示器（均值 = count/N），聚合后由
`derive_binary_prf` 在每个桶（overall / category）推导出精确的 micro-P/R/F1。

### 写入链路基线（2026-09-05，ox-alpha-free 通道 + mimo-v2.5-free judge）

报告：`evals/reports/write-eval-baseline.md` / `.json`（conflict 20 条 + merge 8 条
judge 全量 0 降级 + gate 12 条）。通道来源已回填进基线 JSON 的 `run_provenance`
字段（2026-09-06）；早先文档误记为 DeepSeek 通道。**门禁可比性提醒**：该基线
与 CI 门禁通道（2026-09-06 实测为 openai/omen-alpha，见下）不同源——三 floors
已按 CI 首跑实测校准确认（全部通过、floors 维持），基线迁移到门禁通道按
「换模型 Runbook」步骤 2 执行。

| 套件 | 关键指标 | 数值 | 解读 |
|------|---------|------|------|
| write_conflict（20 条） | precision / recall / F1 | **1.000 / 1.000 / 1.000** | 扩集后仍全对：11 条真矛盾（含 5 条时间演化参数变更：Redis 上限、限流阈值、缓存 TTL、告警切换、生命周期规则）全抓住；9 条硬负例零误报，含"被否决的提案不算矛盾"（wc-019）和"不同环境不同参数"（wc-017） |
| write_merge | fact_coverage | **1.000** | 8/8 合并摘要保留了双侧全部关键事实 |
| write_merge (judge) | faithfulness / completeness | **1.000 / 1.000** | 8/8 judge 成功（0 降级）：合并既不捏造也不丢事实。早期读数 0.600/0.613 是免费 judge 额度 429 限流的降级伪影（降级行按分母政策记 0），不是质量回归 |
| auto_gate | precision / recall / F1 | 0.857 / 1.000 / 0.923 | 12 条 worthy 全放行（recall 1.000）；1 条误放行（ag-009"CI 重试两次就好了"——像观察结论的闲聊），fp_rate 0.083 |

已知问题与后续动作：

- **merge 的 LLM judge 通道受免费 judge 模型额度约束**（约 5-7 分钟窗口恢复，
  一次 merge 全量 8 次调用贴着额度跑）。judge 不稳时该列读数偏低是**分母政策**
  而非质量回归，看 `n_judge_errors` 区分；`judge_errors ≥ 50%` 时 CLI 已显式
  判 run 失败。
- 冲突检测扩集后（20 条，时间演化类已补）仍然全对——当前模型 + prompt 对这类
  边界是真实的强，短板假设被数据否定。后续扩集方向转向**语义等价但数值巧合**
  的负例（如"每分钟 60 次"出现在不同语义下）与跨记忆间接矛盾。
- 写入门禁首次真实 CI 校准完成（2026-09-06，run 34024657208，openai/omen-alpha
  通道 ×3 轮、0 执行错误）：conflict_f1 [1.000, 0.952, 0.952]（ci95_lo 0.900）、
  merge_fact_coverage 1.000、worthy_f1 0.923——三条门禁（0.90/0.90/0.85）全部
  通过，floors 维持原值不重推（能力底线，runbook 步骤 3）。conflict_f1 余量偏
  薄（2/3 轮各有 1 条假阴性），跌破 ≤0.87 ci95_lo 仍会触发门禁。provisional
  标注已移除；提交的写入基线仍为 ox-alpha-free 通道，门禁通道定型后按 runbook
  步骤 2 迁移。

## 运行

```bash
# 只校验标注集一致性（零 LLM / 零 DB，CI 每 push 跑）
python -m evals.run_llm_eval --validate-only

# 冒烟：每套件 3 条，确定性评判（最省 token）
python -m evals.run_llm_eval --sample 3 --judge deterministic

# 六个无 DB 套件（CI llm-eval job 跑这个，经 multi_run_gate 三次取均值）
python -m evals.run_llm_eval --suite tool_selection,extraction,answer,write_conflict,write_merge,auto_gate

# 写入链路三套件（同样无 DB，冲突/门控确定性、merge 可带 judge）
python -m evals.run_llm_eval --suite write_conflict,write_merge,auto_gate --judge llm

# 端到端套件（需要先 seeding + 本地 DB + embedding 模型）
python -m evals.e2e_seed --clear
python -m evals.run_llm_eval --suite e2e \
  --min-context-recall 0.90 --min-fact-coverage 0.70 \
  --min-groundedness 0.80 --min-citation-rate 0.80

# 全量 + LLM 裁判 + 报告
python -m evals.run_llm_eval --suite all \
  --report-md evals/reports/llm-eval-report.md

# 回归门禁（exit 2 = 指标跌破下限）
python -m evals.run_llm_eval --suite all \
  --min-tool-accuracy 0.70 --min-entity-f1 0.60 --min-relation-f1 0.50 \
  --min-fact-coverage 0.60 --min-groundedness 0.80
```

成本：全套 15（工具选择）+ 8（抽取）+ 8（答案）+ 8（端到端）+ 12+8+12（写入链路）
≈ 71 条，每条 1-5 次 LLM 调用，全量 + LLM 裁判约 100-150 次调用，适合每周定时任务。
`--suite` 支持逗号分隔（如 `tool_selection,answer`），`all` 含除 e2e 外的全部套件
（e2e 需要 seeding + DB，显式点名才跑）。

## 架构

```
evals/
  llm_ground_truth.py   # 七套标注集的 item 类型 + validate_llm_dataset()（数据在 data/*.jsonl）
  llm_metrics.py        # 纯函数指标（无 I/O，单测覆盖）
  llm_executors.py      # 默认执行器：包装 call_llm_node / extract_memory / 答案 prompt / e2e 检索
  llm_judge.py          # LLM-as-judge：答案覆盖/忠实 + 摘要忠实/完整 + 合并忠实/完整
  write_eval_metrics.py    # 写入链路纯函数指标（混淆单元推导二分类 P/R/F1 + merge 覆盖）
  write_eval_executors.py  # 写入链路执行器：包装 _detect_conflict / merge_summaries / _llm_gate_verdict
  write_eval_runner.py     # 写入链路三套件的 run_*（结果类/聚合复用 core）
  core.py               # 共用骨架（与检索/task 评测共享）：EvalResult / 聚合 / judge 失败零值 / JSON 序列化
  llm_runner.py         # 每套件一个 run_*：执行 + 聚合 + 错误行归零（结果类/聚合复用 core）
  llm_report.py         # Markdown + JSON 报告 + summarize 一行（序列化复用 core）
  e2e_seed.py           # e2e 语料 seeding CLI（--clear / --dry-run，独立标签）
  run_llm_eval.py       # CLI：--suite（逗号分隔）/ --judge / --sample / --min-*
  experiments/          # 一次性研究脚本归档（A/B、阈值标定、judge 校准、scale 探测…）：不复用主骨架、不进 CI，按需 `python -m evals.experiments.<script>`
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
    `--suite tool_selection,extraction,answer,write_conflict,write_merge,auto_gate`
    （六个无 DB 套件）经 `multi_run_gate` 三次取均值、按 95% CI 下界判门禁，
    并上传报告。写入链路三套件（2026-08-24 加入）的门禁已于 2026-09-06 按首次
    真实 CI 运行校准确认（run 34024657208，openai/omen-alpha 通道，三条门禁
    全部通过、floors 维持 0.90/0.90/0.85）。CI 通道由 secrets 决定
    （`LLM_MODEL` 已透传）——换通道时按「换模型 Runbook」走。
  - `e2e-eval` job：同样每周定时 + 手动触发，带 postgres service + BGE-M3 模型，
    `e2e_seed --clear` 后跑 `--suite e2e`。
  - `task-eval` job：同样每周定时 + 手动触发，前置与 e2e-eval 相同（postgres +
    BGE-M3 + LLM），`e2e_seed --clear` 后跑 `run_task_eval --judge deterministic`
    驱动真实 Agent 图。**门禁阈值暂未设置**——等首份
    `task-eval-report.md` 落地后按真实数字校准（参考 e2e-eval 的标定流程）。
  - 两个（llm/e2e）job 的**门禁都用 `--judge deterministic`**，阈值已按
    `evals/reports/llm-eval-baseline.json`（2026-08-09，commit 4ae4848）校准，
    每个阈值低于基线 0.05-0.10 留噪声余量。门禁退出码：执行失败或指标跌破阈值
    为非零，CI 即红。

### 基线

- **基线文件**：`evals/reports/llm-eval-baseline.json`——一次干净运行
  （0 执行错误、0 judge 降级）的逐指标结果，提交进仓库，不被每次 run 覆盖。
- **语义基线**：`evals/reports/llm-eval-semantic-baseline.json`——judge 通道稳定后
  用 `--judge llm` 跑出的语义判定结果（groundedness / hallucination_rate /
  summary_faithfulness / summary_completeness），供手动分析；不进 CI 门禁。
- **对比**：`python -m evals.experiments.compare_baseline` 把当前报告与基线做 diff，
  输出逐指标 delta，任何跌破 `--tolerance`（默认 0.01，吸收 ~±0.001 的
  运行间噪声）的下降都以非零退出码标红。**每次改 prompt 或模型后跑一次**，
  用 delta 判断该改动是提升还是回归。确定性报告与 LLM-judge 报告混比会被
  拒绝（judge 模式不匹配时脚本明确报错，避免把语义判定的更严当成回归）；
  语义对比用 `--baseline evals/reports/llm-eval-semantic-baseline.json`。
- **重标定**：有意的行为变更（prompt 版本号 bump、模型切换、工具表调整）落地后，
  重新生成基线并同步 eval.yml 阈值。

### 换模型 Runbook

每份报告/基线的 `run_provenance` 字段（provider / model / judge，2026-09-06 起
自动写入）说明它测于哪个通道。跨通道数字**不可比**——基线只在同通道上可比，
换模型后的动作按此收敛：

0. **新通道冒烟（先于一切数字）**：换模型先跑
   `python -m evals.run_llm_eval --suite write_conflict,write_merge,auto_gate --sample 3`
   和 task-eval 的 8 个任务，不等每周 CI。不同模型在"机器接口"上的差异比能力
   差异更容易立刻爆炸——结构化输出是否合规、JSON 契约能否通过、工具调用格式
   对不对。粗破损在冒烟层解决，别让全量 run 烧完 token 才发现模型不吐 JSON。
0b. **task-eval 核对 prompt 适配性**：ADR-012 的工具纪律结论（unexpected_rate
   多轮归零、task-008 零调用收口）是在特定模型上验证的**模型行为学结论**，
   跨模型不保证迁移——停手纪律这类"教模型听话"的 prompt，不同模型服从性不同。
   换模型后跑一轮 task-eval 确认 unexpected_rate 仍为 0、零调用收口未复发；
   首个新模型轮次同时给 task-eval 门禁（暂未设置）提供校准数据。
   **2026-09-06 首次跨模型验证（openai/omen-alpha，run 34024657208）**：
   unexpected_rate 0.0（纪律迁移成功）、completed 0.875（8/8 干净轮、0 执行
   错误）、groundedness 1.000——v7 纪律在新模型上保持，该基线数字可直接作
   task-eval 门禁校准参考。
1. **本地实验通道（不产基线）**：换模型试效果，用
   `compare_baseline --baseline <当前基线>` 看相对当前基线的 delta，或改动前后
   同通道各跑一次做 A/B。实验通道的数字不提交为基线、不进 eval.yml。
2. **成为门禁通道（CI 所用通道）**：重测基线并提交——
   `LLM_PROVIDER=<CI 通道> LLM_API_KEY=<key> python -m evals.run_llm_eval
   --suite <该 job 的套件> --judge llm --report-json evals/reports/<基线名>.json`，
   确认 0 执行错误 / 0 judge 降级后提交；基线 JSON 自带
   `run_provenance`，无需再文档考古。**基线匿名不可提交**——
   `evals/tests/test_baseline_provenance.py` 会拦下没有 model/judge 字段的
   基线（write 基线曾匿名入库、事后考古回填的事故不再重演）。
3. **门禁阈值何时动**：默认**不动**——写入门禁 floors（0.90/0.85 级别）是能力
   底线，换同量级模型通常仍贴近满分，门禁照常工作。只有 CI 连续误红（新模型
   压线）时，按新基线重推 floors 并去掉 YAML 里的 provisional 标注。
4. **judge 模型单独换**：等于换了测量仪器——语义指标（groundedness /
   faithfulness 等）跨 judge 不可比，语义基线需要重跑；确定性门禁不受影响。
5. **embedding 模型换**：另一个量级——全库重嵌入 + 检索基线全部作废重测，
   本 runbook 不覆盖。
6. **成本追踪同步**：`backend/service/usage.py` 的价格表对新模型回落保守默认
   价（不报 0 但不准，且无提示）。换模型时按 provider 定价把新模型加进
   `_PRICE_RULES`（一行正则匹配），否则成本仪表盘每次换模型静默降级为估算。

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

> 评测代码在 `evals/` 下的 `task_*` 模块，CLI 入口是 `python -m evals.run_task_eval`。

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
python -m evals.e2e_seed --clear
# 2. 全量跑（真实 LLM，默认 --judge llm）
python -m evals.run_task_eval --report-md evals/reports/task_eval_report.md
# 3. 免 judge 通道（CI 用，更便宜）
python -m evals.run_task_eval --judge deterministic
# 4. 零成本校验（ci.yml 每 push 跑）
python -m evals.run_task_eval --validate-only
```

### 实测结果

实测数字（2026-08-09 baseline 与 2026-08-11 锁死 LLM rerank 复测）见
[task_eval 评测报告](../../evals/reports/task_eval_report.md)。

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
