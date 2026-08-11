# EMA 工程文档

面向研发团队的长期记忆智能体。本文档目录汇总 EMA 的设计、评估与复盘材料。

## 项目理解

| 文档 | 内容 |
|------|------|
| [项目定位与背景](project-overview.md) | 项目解决的问题、技术栈、核心设计、量化成果、评估驱动的优化迭代 |
| [架构深潜](deep-dive.md) | 系统架构、分层职责、关键技术决策（4 个 ADR）、技术难点攻克、成果与反思 |

## 设计与权衡

| 文档 | 内容 |
|------|------|
| [技术设计笔记](design-notes.md) | 六个板块的设计与权衡：Agent 架构、RAG 与检索、向量检索与 pgvector、Prompt 工程、AI 工程化、系统设计 |
| [技术决策问答](decision-faq.md) | 项目演进中反复出现的技术决策及其依据，按主题整理 |
| [短板诊断与评估补全](gap-remediation.md) | 已知短板、补全方案、实测数据（压测、延迟、成本） |
| [代码审查记录](code-review-findings.md) | 基于对抗性技术审查的缺陷清单：证据、修复状态、处理方向 |
| [项目演进与决策复盘](lessons-learned.md) | 关键决策复盘：LangGraph 选型、阈值标定、Neo4j 反例、自动路由取舍 |

## 评估体系

| 文档 | 内容 |
|------|------|
| [LLM 行为评测](llm-eval.md) | 工具选择 / 知识抽取 / 最终答案 / 端到端四套件评测的设计说明 |
| [LLM 行为评测报告](../../tests/eval/reports/llm-eval-report.md) | LLM 行为评测报告（2026-08-09，39 条，LLM judge 通道） |
| [检索评测报告](../../tests/eval/reports/eval-report.md) | 2026-08-07 检索评测历史报告 |
| [检索评测（30 条库稀疏路径）](../../tests/eval/reports/eval_30_db_sparse.md) | 早期 30 条语料检索评测 |
| `tests/eval/reports/*` | 全部评测报告与基线（hard-negative、judge 校准、memory 路径、hybrid 对比、memory LLM rerank vs CE、decay A/B、extraction A/B、task 级端到端） |

## 相关目录

- 系统架构与技术决策：`../architecture.md`、`../agent-design.md`、`../memory-system.md`、`../deployment.md`
- 架构决策记录（ADR）：`../decisions/`
- 领域模型与扩展设计：`../design/`
