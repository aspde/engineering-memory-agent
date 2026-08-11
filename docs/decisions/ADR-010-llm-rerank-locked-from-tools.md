# ADR-010: 检索工具锁死 LLM rerank 暴露

**日期**: 2026-08-11

**状态**: 已接受

## 背景

`search_memories_tool` / `retrieve_chunks_tool` 曾把 `use_llm_rerank` 作为**工具参数暴露给 LLM**（默认 `False` 但模型可自选 `True`）。`measure_chat_p95.py` 实测对话端到端 P95 **73.6s**，其中约 **40s** 来自每轮对话 ~19 次 `rerank_llm` 调用（每个候选 memory 一次 LLM 调用，~2.5s/次）——模型在对话里自主开启了慢速重排。

`llm_usage` 表的量化证据（provider 层唯一咽喉点埋点）：

- **全历史**（964 行调用）：`rerank_llm` 842 次 / 291,131 tokens / 5,796s（96.6 分钟）/ 估算成本占 **56%**。
- **p95 对话线程**（10 轮）：`rerank_llm` 191 次 / 475s，占该轮延迟 **58.6%**、成本 **46%**。
- **单轮结构**：走 rerank 的轮 67-110s，没走的轮 12-40s——**是否走 rerank 是对话 P95 的决定性变量**。

## 决策

**把 `use_llm_rerank` 从三个检索工具（`search_memories_tool` / `retrieve_chunks_tool` / `query_rewrite_and_search_tool`）的 schema 中移除**，使模型在 agent 路径上无法再触发 LLM rerank：

- 工具调用固定为纯相似度排序（hybrid 用 RRF 融合，均为确定性、零额外 LLM 调用）。
- **LLM rerank 能力保留在服务层**：`rerank_llm()` 与各读函数的 `use_llm_rerank` 参数仍在，供**显式调用者**（eval 评测、未来的服务端调用方）使用。`backend/service/retrieval.py` 各签名已标注 `NOT exposed in agent tool schemas`，防止回归暴露。

## 理由

1. **量化证据明确**：rerank_llm 是对话路径延迟和成本的双第一（延迟 58.6%、成本 46%），是 agent_chat 本身（124s）的 3.8 倍。
2. **锁死不损失检索质量**：eval 已实测 LLM rerank 在小语料下**不改变召回集合**（`memory_llm_vs_ce_report.md`：recall@5 同为 0.900，MRR +0.014 / NDCG +0.011），只微调排序。锁死牺牲的是 MRR +0.014 的边际排序收益，换来延迟/成本各降近半。
3. **回归验证通过**：锁死后的 task_eval 复测（`task_eval_norerank_report.md`）——`completed` 0.500 持平、`groundedness` 1.000 不变、`within_budget` 0.875→1.000（概念查询更少撞 max_steps）。对话 P95 预期从 73.6s 降至 ~25s 量级。
4. **暴露点是设计缺陷而非功能**：慢速且成本高的可选路径不该出现在模型的自主决策面上——模型按"尽力检索"的意图选择 rerank，但无法评估其延迟/成本代价。

## 代价与保留

- **LLM rerank 作为能力保留**：服务层 `rerank_llm` 仍在，显式调用者可启用。这是能力降级为 opt-in 而非删除。
- **万级语料的 rerank 正收益仍然成立**：scale 实测（`gap-remediation.md` §11.5.3）显示 10k 语料下 +CE rerank Recall 0.933→0.967 / MRR 0.732→0.886。锁死的是 **agent 自主开启 LLM rerank** 这条路径，不是 rerank 本身——数据量增长到 rerank 正收益时，重新评估的是"如何低成本开启"，而不是回到模型自主选择。
- **cross-encoder 重排不受影响**：`query_memories(use_cross_encoder=True)` 的 bounded top-3 重排（硬负例判别力 59.3%→81.5%）保留，它是本地零 API 成本路径，不在本决策范围。

## 拐点

当以下条件**同时**出现时，重新评估 agent 端 LLM rerank：

1. 语料规模进入万级（rerank 正收益区间，见 `gap-remediation.md` §11.5.3）；
2. embedding/rerank 服务化上 GPU，`rerank_llm` 单次延迟降到交互可接受范围（当前 ~2.5s/候选，10k 实测 29.9s/query 全量）。

在此之前，agent 与 HTTP API 检索路径维持纯确定性排序；LLM rerank 仅由服务层显式调用者（eval）启用。

## 后果

- 三个检索工具 schema 移除 `use_llm_rerank` 参数；服务层签名保留并标注"NOT exposed in agent tool schemas"。
- `/api/memory/search` 与 `/api/memory/memories/search` 的请求体同样移除 `use_llm_rerank`（2026-08-11 补充，见下节）——开关只在服务层存活，外部入口无法触发。
- 对话 P95 预期从 73.6s 降至 ~25s 量级，对话成本估算省 ~46%。
- 完整量化证据见 `docs/engineering/gap-remediation.md` §3.1.1 与 `tests/eval/reports/task_eval_norerank_report.md`；检索侧 rerank 效果对比见 `tests/eval/reports/memory_llm_vs_ce_report.md`。

## 补充：API 层同样封死（2026-08-11）

初版决策只封了 agent 工具 schema，`/api/memory/search` 与 `/api/memory/memories/search` 的请求体仍暴露 `use_llm_rerank`（默认 `False`）。该开关在生产路径上无真实调用者——前端两个搜索端点都不传此字段（`frontend/src/api/memory.ts` 的载荷只有 `{query, top_k}`），eval 评测直调服务层不经 HTTP。但初版把"API 客户端"列为显式调用者之一，实际上这个入口只是遗留的可触发慢速重排的洞，与"模型自主决策面不暴露慢路径"的精神相悖。

**决策**：两个搜索请求体同样移除该字段。Pydantic 默认 `extra="ignore"`，旧客户端即使仍传该字段也会被静默忽略、返回 200，不构成 breaking change。`retrieve_hybrid` / `query_memories` 服务层签名不变，eval 的 `--compare` 对照、hard-negative、scale 实验全部不受影响。若未来出现需要经 HTTP 显式开启 LLM rerank 的服务端调用方，加回字段即可（显式 opt-in，不进 agent/API 默认路径）。
