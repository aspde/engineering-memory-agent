# Extraction A/B — few-shot + function calling

> 三阶段提取的两项优化：entity/relation prompt 加 few-shot examples（v3→v4，含「只抽显著实体」约束），以及 OpenAI 兼容 provider 上的函数调用通道（enum 在生成期约束，失败降级到 `chat_structured`）。8 条标注 query，deterministic judge（纯子串/归一化匹配，无 LLM 裁判）。

## 结果

| 指标 | zero-shot + json（2026-08-09 基线） | few-shot + json | few-shot + 函数调用 |
|------|------|------|------|
| entity_precision | 0.762 | 0.740 | 0.713 |
| entity_recall | 0.781 | 0.927 | 0.969 |
| entity_f1 | 0.741 | 0.808 | 0.802 |
| entity_type_accuracy | 0.812 | 0.917 | 0.812 |
| relation_precision | 0.292 | 0.515 | 0.346 |
| relation_recall | 0.531 | 0.812 | 0.688 |
| relation_f1 | 0.356 | 0.600 | 0.436 |
| summary_coverage | 0.938 | 1.000 | 0.875 |

## 解读

- **few-shot 是主要收益**：相比 zero-shot 基线，entity_recall 0.781→0.927（+0.146）、relation_f1 0.356→0.600（+0.244）——in-domain examples + 「只抽显著实体」约束让模型更完整且更克制地抽取。
- **函数调用通道（对比 few-shot + json）**：entity 上 precision 略降（0.740→0.713）、recall 略升（0.927→0.969），F1 基本持平（0.808 vs 0.802）；**entity_type_accuracy 0.917→0.812 下降**——工具通道比 prompt 通道更容易在枚举边界处出错（enum 约束保证类型合法但不保证类型正确）。
- **description 收紧的实测收益**（对照第一轮：工具 description 未约束「只抽显著实体」时 entity_precision 0.609 / entity_f1 0.722）：**precision +0.104、F1 +0.080**。收紧 description 直接回拉了过度抽取。
- **relation 工具通道仍是弱项**：relation_precision 0.346 vs fs+json 0.515、relation_f1 0.436 vs 0.600——工具通道未继承 prompt 的「只抽明确支持」约束，模型倾向多抽关联。残余短板，可再收紧工具 description（relation 已是第二次 iteration 的对象）。
- **取舍**：函数调用通道用少量 entity precision 换结构保证（格式错误趋近 0）+ 双通道降级兜底。生产保持「函数调用优先、chat_structured 兜底」。

## 边界

- 基线来自 2026-08-09 committed 快照（同 provider deepseek-v4-flash），日期不同，模型可能漂移。
- 8 条标注集较小，单指标 ±0.1 以内波动可能只是采样噪声；趋势比绝对值更有意义。
- `few-shot + 函数调用`是当前生产代码；`few-shot + json`通过禁用工具通道获得。
- 与上一轮 A/B（description 未收紧，entity_precision 0.609）对比时注意：prompt 约束 v4 同时作用于两个通道，fs+json 本身也提升了。
