# EMA 工程文档

面向研发团队的长期记忆智能体。本文档目录汇总 EMA 的设计、评估与复盘材料。

## 项目理解

| 文档 | 内容 |
|------|------|
| [项目定位与背景](project-overview.md) | 项目解决的问题、技术栈、核心设计、量化成果——理解代码库的入口 |
| [架构深潜](deep-dive.md) | 系统架构、分层职责、关键决策（ADR-001/002/004/006 详解）、技术难点攻克、成果与反思 |

## 设计与权衡

| 文档 | 内容 |
|------|------|
| [技术设计笔记](design-notes.md) | 六个板块的设计问答：Agent 架构、RAG 与检索、向量检索与 pgvector、Prompt 工程、AI 工程化、系统设计 |
| [技术决策问答](decision-faq.md) | 被反复追问的技术决策及其依据（阈值标定、衰减移除、judge 可信度等），按主题整理 |
| [评估体系与优化记录](gap-remediation.md) | 短板诊断、评估体系设计、实测数据（压测/延迟/成本/万级语料）、检索召回优化（hybrid search、rerank scale-dependent） |
| [代码审查记录](code-review-findings.md) | 对抗性技术审查的缺陷清单：证据、修复状态、处理方向 |
| [项目演进与决策复盘](lessons-learned.md) | 五次关键决策复盘：LangGraph 选型、阈值标定、Neo4j 反例、自动路由取舍、推进方式 |

## 评估体系

评测代码在 `evals/`，五套件覆盖组件级到任务级；报告产物见 [evals/reports/](../../evals/reports/README.md)（决策证据在主目录，一次性实验归档在 `archive/`）。

| 文档 | 内容 |
|------|------|
| [LLM 行为与任务级评测](llm-eval.md) | 工具选择 / 抽取 / 最终答案 / 端到端四套件 + task_eval 任务级端到端的设计说明与 CLI 用法 |

## 相关目录

- 系统架构与技术决策：`../architecture.md`、`../agent-design.md`、`../memory-system.md`、`../deployment.md`
- 架构决策记录（ADR）：`../decisions/`（11 个，编号 001-011）
- 领域模型与扩展设计：`../design/`
