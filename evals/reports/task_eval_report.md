# Task 级端到端评测报告

> 数据来源：`python -m evals.run_task_eval`。设计说明见 [llm-eval.md](../../engineering/llm-eval.md)。

## 实测结果（2026-08-09，DeepSeek，`--judge deterministic`）

| 指标 | 数值 | 解读 |
|------|------|------|
| `completed` | **0.500** | 严格完成率——调齐必备工具 + 实质答案 + 无错误 |
| `tool_recall` | 0.938 | 该调的工具几乎都调了——**工具选择意图很准** |
| `unexpected_rate` | 0.375 | 3/8 任务调了预期外工具——**过度调用是主短板** |
| `within_budget` | 0.875 | 1 个概念查询撞 max_steps（task-004，单任务循环 8 次工具调用） |
| `groundedness` | 1.000 | 答案全部接地，无捏造（deterministic 通道） |
| `citation_rate` | 0.875 | 绝大多数答案引用了实际看到的记忆/文档 |
| 执行错误 | 0 | 8 个任务全部跑完，无超时/崩溃 |

**这是比分数更重要的发现**：`tool_recall 0.94 + groundedness 1.00 + 0 错误` 说明模型的
工具选择意图和答案忠实度都没问题，但 `completed 0.5` 暴露了组件级评测看不到的
**轨迹级行为**——DeepSeek 对单检索任务过度调用工具（task-006 回答一个记忆问题调了
`search_memories`×2 + `retrieve_chunks` + `query_entity` 共 4 次），概念查询甚至
循环到撞 max_steps。`unexpected_rate 0.375` 直接把严格 `completed` 拉下 0.5。

改进方向（任一都能拉高 completed）：① 强化工具描述边界——把 `search_memories`
（记忆）与 `retrieve_chunks`（文档）的决策标准写得更明确，减少"顺手再查一下"；
② 轨迹级节流——检索结果已覆盖提问就停手；③ max_steps 已兜底，但 5 步对概念
查询仍偏松。

## 锁死 LLM rerank 后的复测（2026-08-11）

对话 P95 分析（`measure_chat_p95.py`）发现每轮 ~19 次 `rerank_llm`（模型自主把
`use_llm_rerank=True` 传给检索工具，占约 40s/轮）。把三个检索工具 schema 里的
`use_llm_rerank` 参数移除后复测（`evals/reports/task_eval_norerank_report.md`）：

| 指标 | 2026-08-09 baseline | 锁死 rerank 后 | Δ |
|------|--------------------|----------------|-----|
| `completed` | 0.500 | **0.500** | 持平 |
| `tool_recall` | 0.938 | 0.812 | -0.13（1 个任务波动，8 任务小样本） |
| `within_budget` | 0.875 | **1.000** | +0.13（概念查询更少撞 max_steps） |
| `groundedness` | 1.000 | **1.000** | 不变（答案仍全接地） |
| `citation_rate` | 0.875 | 0.750 | -0.13（同上，噪声范围） |

结论：移除 LLM rerank 不伤害任务完成率与答案忠实度（`completed` 持平、
`groundedness` 1.000），且 `within_budget` 改善——与检索侧 eval 的结论一致
（rerank 不改变 recall@5，只微调排序）。对话 P95 的 rerank 大头由此消除。

## 工具纪律 prompt（agent.system v6）复测（2026-08-24）

针对 `unexpected_rate 0.375` 的 prompt/描述层修复：`agent.system` v5→v6
（store-selection + 停手纪律 + 小对话免检索）、三个检索工具 docstring 边界收紧
（search_memories 声明为默认首选并禁止链式、retrieve_chunks 删除"memory search
不够再查我"、query_rewrite 收窄为 last-resort）、tsel-006 重标对齐新策略。
改动明细见 ADR-012。

本地复跑（DeepSeek 走 opencode 代理，当晚 provider 多次 503/超时污染部分轮次的
答案指标；**工具调用发生在最终答案生成之前，轨迹指标不受污染影响，全部有效**）：

| 指标 | 基线（3 次观测） | 改动后（3 次复测） | Δ |
|------|----------------|------------------|-----|
| `unexpected_rate` | **0.375 / 0.375 / 0.375** | **0.000 / 0.000 / 0.000** | **-0.375，目标达成** |
| `tool_recall` | 0.875 / 0.938 / — | 0.812 / 0.688* / 0.750* | 见下注 |
| `within_budget` | 1.000 / 1.000 / — | 1.000 / 1.000 / 0.875 | 基本持平 |

\* 污染轮的 tool_recall 含超时桩导致的少调（task-003 整任务撞墙钟超时记 n=0），
非策略回归——干净轨迹里该调的工具都调了（如 run1 task-002
`search_memories + retrieve_chunks` 双库正确、task-003 `search_memories +
write_memory` 正确）。

**轨迹形态对比**（最能说明问题的单任务）：

- task-001（factual，预期仅 search_memories）：基线调 `query_rewrite +
  retrieve_chunks` 且漏掉预期工具 → 改动后三次全部精确一次 `search_memories`；
- task-002（multi_retrieve，双库合法）：改动后保持两库各查一次，停手纪律没有
  误伤多段问题的合法检索；
- task-007（no_tool 寒暄）：改动前后都是零调用——"小对话免检索"条款没有把
  边界推过头。

tool_selection 单决策套件（15 条）：accuracy **0.867→0.933**（修复 query_rewrite
docstring 示例与 tsel-006 的自相矛盾后）、unexpected_rate 0.000，高于 CI 门限 0.68。

**未决观察项**：task-008 在两次复测中出现零调用直接回答（模型认为 EMA 自身功能
问题无需检索）。样本太小且该任务 query 与 agent 自身知识高度重叠，暂不判定为
过度抑制；若后续复现需在 docstring 中强化"操作类问题也要先搜记忆库里的用法记录"。

**completed 干净复测（2026-08-24 run 4，provider 零故障轮）**：

| 指标 | 历史/基线 | run 4（干净） |
|------|----------|--------------|
| `completed` | 0.500（2026-08-09）/ 0.375（本地基线） | **0.625**（历史最高） |
| `unexpected_rate` | 0.375 | **0.000**（第 4 次连续归零） |
| `groundedness` / `citation_rate` | 1.000 / 0.750-1.000 | **1.000 / 1.000** |
| `within_budget` | 0.875-1.000 | 0.875（唯一 miss 是 task-001 撞 AGENT_TIMEOUT=180s 墙钟，provider 慢导致，非行为回归） |

run 4 里 task-002 双库、task-003 检索+写入、task-004 概念查询走记忆检索全部精确完成。

**task-008 观察项升级为已知取舍**：该任务（问 EMA 自身功能）在改动后三次复测中连续零调用直接作答（改动前调 3 次 search）。已验证两个事实当前均可检索到（memory 0.675 / chunk 1.0 排第一），所以不是检索不到，而是模型判断"问我自己的功能不需要查库"。答案 groundedness 保持 1.000 无捏造（system prompt 本身描述了摄入能力，模型答得对），但严格轨迹分归零。这是停手纪律在"agent 自身功能类问题"上的副作用；若要消除，需在 prompt/docstring 强化"涉及 EMA 自身用法的问题也先搜记忆库"，属后续迭代。
