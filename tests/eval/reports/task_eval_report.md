# Task 级端到端评测报告

> 数据来源：`python -m tests.eval.run_task_eval`。设计说明见 [llm-eval.md](../../engineering/llm-eval.md)。

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
`use_llm_rerank` 参数移除后复测（`tests/eval/reports/task_eval_norerank_report.md`）：

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
