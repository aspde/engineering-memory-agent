# 评测报告索引

本目录存放检索 / LLM 行为 / 任务级端到端评测的产物。报告分两类，按"是否支撑当前设计决策"划分——**决策证据**留在本目录，**实验记录**归档到 [archive/](./archive/)。

> 归档文件的引用路径均为 `tests/eval/reports/archive/<name>`；本索引是"哪份报告决定了哪个决策"的入口。

## 决策证据（本目录）

以下报告对应已落地的设计决策，改动相关代码时对照它们判断是提升还是回归：

| 报告 | 支撑的决策 |
|------|-----------|
| [decay_ab_report.md](./decay_ab_report.md) | 记忆排序移除衰减加权（[ADR-009](../decisions/ADR-009-decay-weighting-removed.md)）：衰减加权 recall@5 0.667 vs 无衰减 0.900 |
| [task_eval_report.md](./task_eval_report.md) / [task_eval_norerank_report.md](./task_eval_norerank_report.md) | 任务级端到端评测基线；锁死 LLM rerank 不伤害完成率（[ADR-010](../decisions/ADR-010-llm-rerank-locked-from-tools.md)） |
| [memory_llm_vs_ce_report.md](./memory_llm_vs_ce_report.md)（+ `.json`） | LLM rerank 不改变召回集合、只微调排序（recall@5 同 0.900）——rerank 锁死的依据 |
| [memory_path_report.md](./memory_path_report.md) / [memory_path_report_70.md](./memory_path_report_70.md)（+ `.json`） | 生产 `query_memories`（memory 路径）检索基线（当前数字来源） |
| [eval-report.md](./eval-report.md)（+ `.json`） | 检索基线历史报告；rerank scale-dependent 证据链（30 条语料 rerank 有害，0.967 vs 1.000） |
| [hybrid_report.json](./hybrid_report.json) / [hybrid_norerank_report.json](./hybrid_norerank_report.json) | 上述 rerank A/B 的原始数据 |
| [llm-eval-report.md](./llm-eval-report.md) / [llm-eval-baseline.json](./llm-eval-baseline.json) / [llm-eval-semantic-baseline.json](./llm-eval-semantic-baseline.json) | LLM 行为评测（工具选择 / 抽取 / 答案）基线与 CI 门禁基准 |

## 实验记录（[archive/](./archive/)）

一次性实验的输出：回答了当时的假设、产出了标定结果，但本身不是当前设计的一部分。复现实验时可重新生成，无需在主线引用。

| 报告 | 实验 |
|------|------|
| [archive/threshold_calibration_report.md](./archive/threshold_calibration_report.md) | 四级相似度阈值标定（0.92→0.85 / 0.75→0.72 的分布依据，8 对改写样本初步标定） |
| [archive/extraction_ab_report.md](./archive/extraction_ab_report.md) | 三阶段提取 few-shot + 函数调用通道 A/B |
| [archive/hard_negative_report.md](./archive/hard_negative_report.md)（+ `.json`） | 27 条 hard-negative 判别集；bounded cross-encoder 重排 59.3%→81.5% |
| [archive/judge_calibration_report.md](./archive/judge_calibration_report.md)（+ `.json`） | LLM judge 一致性小样本校准 |
| [archive/scale_1000_report.json](./archive/scale_1000_report.json) / [archive/scale_30_db_sparse.json](./archive/scale_30_db_sparse.json) / [archive/eval_30_db_sparse.md](./archive/eval_30_db_sparse.md) | 语料规模探测（probe_scale）与 sparse DB 侧迁移前的旧基线 |
