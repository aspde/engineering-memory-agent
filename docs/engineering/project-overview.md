# EMA 项目定位与背景

> 面向研发团队的长期记忆智能体。本文档描述项目解决的问题、技术栈、核心设计与量化成果，是理解整个代码库的入口。

## 项目一句话

EMA（Engineering Memory Agent）把研发过程中的代码、Git 历史、技术决策、故障经验，自动提取成结构化的长期记忆，让团队可以跨时间、跨人检索复用。

核心痛点：**团队的知识存在人脑和聊天记录里，人走了知识就没了**。EMA 不是写 wiki，而是从工程活动里自动沉淀。

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| 后端 | Python 3.12 + FastAPI | async 优先，所有 IO 用 async/await |
| Agent | LangGraph 手动 StateGraph | ReAct 循环 + 2 个 HITL 卡点（写前审批、冲突仲裁） |
| LLM | LLMProvider 抽象 | DeepSeek / Claude 切换，业务代码不依赖具体 SDK |
| Embedding | BGE-M3 本地部署 | 可替换，含故障转移 |
| 数据库 | PostgreSQL 16 + pgvector | 结构化数据 + 向量检索 + 对话 checkpoint 三合一 |
| 前端 | React + TypeScript + Vite + Tailwind | 6 页面，可独立全栈交付 |

## 核心设计

### 1. 四级相似度去重（阈值经标定）

- ≥0.85：LLM 合并摘要 + 实体关系
- 0.72–0.85：LLM 检测矛盾 → 矛盾走 HITL 仲裁，不矛盾补充关联
- 0.60–0.72：插入新记忆，建立 supplement 关联
- <0.60：全新插入

阈值不是拍脑袋：收集三类摘要对（同义改写 0.84-0.97 / 同类 ≤0.79 / 异类）算 BGE-M3 相似度分布，发现旧值 0.92 高到一半该 merge 的同义对被漏成冲突检测，0.85 是自然分离点。

### 2. 召回统计（替代艾宾浩斯衰减加权）

检索按纯相似度排序（单段 HNSW），命中记忆记录 `recall_count`/`recalled_at` 作元数据。原艾宾浩斯衰减加权（`相似度 × decay_factor`）因 **decay A/B 实测掉 recall 被移除**：合成老化分布上衰减加权 recall@5 0.667、无衰减 0.900——唯一测量表明衰减让检索变差，且其前提「近期/高频=相关」没有真实语料支撑；连续调参（S=2→8→12）本质是把曲线逼近 no-op。移出排序路径后，「记忆是否过期」由 patrol LLM 读原始 `recalls`/`last_recalled` 字段判断，不再靠机器算公式。

### 3. LangGraph 手动 StateGraph 而非预建 Agent

不用 `create_react_agent` 是因为它是黑盒，HITL 的插入点不好控制。手动建图让两个 HITL 卡点（写前审批、记忆冲突仲裁）精确可控，用 interrupt + PostgresSaver 实现跨重启的对话恢复。

## 量化成果

| 指标 | 数值 | 说明 |
|------|------|------|
| 数据源 | 4 个 | Git / PingCode / CI / 飞书 |
| 检索评估集 | 70 条标注 query | 5 类 × 14、hard 占 30%，生产 memory 路径默认确定性基线 Recall@5 0.886 / MRR 0.767（纯子串匹配无自证） |
| 检索判别力 | 27 条 hard-negative | 纯向量综合通过 59.3% → bounded cross-encoder top-3 重排后 81.5% |
| sparse 检索 | O(N) → O(log N) | jieba 分词落 chunks.tokens 列 + GIN 索引，1000 条语料延迟 -69% |
| rerank | 17.5s → 0.19s | A/B 验证小语料下 cross-encoder 有害（0.15 floor 误伤低分相关），默认跳过 |
| Agent 任务级 | completed 0.5 / tool_recall 0.94 / grounded 1.0 | 8 个多步任务驱动完整图，0 执行错误；unexpected_rate 0.375 暴露过度调用 |
| 压测 | 10 并发 QPS 4.8 / P95 110ms | 160 并发 QPS 63、0 失败，瓶颈在 BGE-M3 CPU 嵌入 |
| 规模 | 后端+agent 1.6 万行 / 1450+ 测试 | — |

## 评估驱动的优化迭代

评估集带出几轮假设检验，每一轮都修正了此前的认知：

1. **score 阈值不可行**：第一版评估 Recall@5 0.83、5 个概念查询 miss。假设"score 低的 query 召回不可靠，用 score 阈值触发 query 改写"——跑数据验证打脸：命中的 top-1 score 最低 0.54，没命中的最高 0.69，**完全重叠**。score 不能当置信度用。
2. **tsvector simple 切不了中文**：先用 Postgres 原生 tsvector 做 BM25，`simple` 分词器把整段中文当一个 token，对中文 query 返回 **0 行**。
3. **语料质量决定"瓶颈"真假**：把语料改成真实工程记录后，BGE-M3 稠密召回直接到 1.00——当时的 5 个 miss 一部分是语料/标注问题，不是纯检索算法缺陷。
4. **真瓶颈是中文 sparse O(N) 扫描**：jieba 分词落库 + GIN 索引解决，1000 条语料延迟 -69%。
5. **rerank 收益 scale-dependent**：cross-encoder + 0.15 floor 在小语料下误伤低分相关结果（q015 打分 0.142 被滤掉），跳过 rerank 延迟从 17.5s 降到 0.19s。别假设"更重的模型一定更好"。
6. **Agent 质量要靠整条轨迹测**：任务级端到端评测（8 个多步任务驱动完整图）抓出过度调用（unexpected 0.375）——组件级评测永远看不到的轨迹级问题。这套评测还抓出并修复了一个生产 HITL bug（拒绝审批后写操作仍被执行）。

## 工程实践

项目端到端主导、没有团队兜底，所以从第一天用团队工程的纪律约束自己：每个架构决策写 ADR、每次提交过 CI、每个检索改动跑评估集对比——用可追溯的流程替代缺失的 code review。

## 相关文档

- [架构深潜](deep-dive.md)：系统架构、分层职责、关键决策
- [技术决策问答](decision-faq.md)：被反复追问的设计决策及依据
- [短板诊断与评估补全](gap-remediation.md)：已知短板与补全方案
- [代码审查记录](code-review-findings.md)：对抗性审查证据与修复状态
- [技术设计笔记](design-notes.md)：各技术主题的设计问答
- [项目演进与决策复盘](lessons-learned.md)：演进过程中的决策复盘
- [LLM 行为评测](llm-eval.md)：工具选择/抽取/答案/端到端评测
- [检索评测报告](../../tests/eval/reports/eval-report.md)：检索 Recall@K / MRR / 阈值标定
