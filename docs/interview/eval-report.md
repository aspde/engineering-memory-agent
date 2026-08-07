# EMA Retrieval Evaluation Report

- Generated: 2026-08-06 06:11:20 UTC (vector baseline) / 2026-08-06 07:50 UTC (hybrid+jieba+rerank) / 2026-08-06 08:20 UTC (hybrid+jieba no rerank)
- Queries: 30
- Configs: 3
- Errors: 0

## Overall

| config | recall@5 | precision@5 | hit_rate@5 | mrr | ndcg@5 | map@5 | latency_ms | n_queries |
|---|---|---|---|---|---|---|---|---|
| chunk:vector@k5 | 0.833 | 0.180 | 0.833 | 0.817 | 0.819 | 0.811 | 1046 | 30 |
| hybrid:ce@k5 (jieba, +rerank) | 0.967 | 0.193 | 0.967 | 0.917 | 0.930 | 0.917 | 20906 | 30 |
| hybrid:norank@k5 (jieba, no rerank) | **1.000** | 0.200 | **1.000** | **0.983** | **0.988** | **0.983** | 1085 | 30 |

> hybrid = BGE-M3 向量 + jieba 中文分词 BM25 (Jaccard)。
> `+rerank` = cross-encoder (BGE-reranker-v2-m3 568M) + 0.15 floor；`no rerank` = max(dense sim, sparse jaccard) 排序。
> **关键发现**：30 条语料下 rerank 有害——候选池覆盖 73%，dense sim 已接近完美排序，cross-encoder 打分噪声掉 MRR，0.15 floor 误伤 q015。跳过 rerank 后 Recall 1.00、延迟 235ms（稳态，去掉首 query warmup）。
> rerank 收益是 scale-dependent 的：万级语料候选池缩到 4%，dense sim 区分度下降，rerank 才有正收益——生产部署前需重新评估。

## Recall@5 by category

| config | 技术决策 | 故障复盘 | 架构设计 | 代码实现 | 历史背景 |
|---|---|---|---|---|---|
| chunk:vector@k5 | 1.000 | 0.667 | 0.833 | 1.000 | 0.667 |
| hybrid:ce@k5 (jieba, +rerank) | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 |
| hybrid:norank@k5 (jieba) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## MRR by category

| config | 技术决策 | 故障复盘 | 架构设计 | 代码实现 | 历史背景 |
|---|---|---|---|---|---|
| chunk:vector@k5 | 1.000 | 0.667 | 0.750 | 1.000 | 0.667 |
| hybrid:ce@k5 (jieba, +rerank) | 1.000 | 0.833 | 0.833 | 1.000 | 0.917 |
| hybrid:norank@k5 (jieba) | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 |

## Recall@5 by difficulty

| config | easy | medium | hard |
|---|---|---|---|
| chunk:vector@k5 | 0.714 | 0.929 | 0.778 |
| hybrid:ce@k5 (jieba, +rerank) | 1.000 | 0.929 | 1.000 |
| hybrid:norank@k5 (jieba) | 1.000 | 1.000 | 1.000 |

## hybrid:ce@k5 (+rerank) 唯一 miss

| id | query | n_ret | n_rel | recall@5 | 原因 |
|---|---|---|---|---|---|
| q015 | 人工介入 HITL 是怎么实现的 | 3 | 0 | 0.000 | seed-015 在候选池内，cross-encoder 打分 0.142 < `_RERANK_FLOOR` 0.15，被 floor 滤掉 |

> hybrid:norank@k5 无 miss：跳过 rerank 后 q015 的目标在 top-5 内（rank-2，MRR=0.5），Recall@5=1.000。

<details><summary>Per-query detail (chunk:vector@k5)</summary>

| id | category | difficulty | n_ret | n_rel | recall@5 | mrr | ndcg@5 | latency_ms |
|---|---|---|---|---|---|---|---|---|
| q001 | 技术决策 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 24920 |
| q002 | 技术决策 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 223 |
| q003 | 技术决策 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 204 |
| q004 | 技术决策 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 218 |
| q005 | 技术决策 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 223 |
| q006 | 技术决策 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 216 |
| q007 | 故障复盘 | easy | 5 | 0 | 0.000 | 0.000 | 0.000 | 260 |
| q008 | 故障复盘 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 204 |
| q009 | 故障复盘 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 197 |
| q010 | 故障复盘 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 209 |
| q011 | 故障复盘 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 203 |
| q012 | 故障复盘 | hard | 5 | 0 | 0.000 | 0.000 | 0.000 | 193 |
| q013 | 架构设计 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 198 |
| q014 | 架构设计 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 206 |
| q015 | 架构设计 | medium | 5 | 2 | 1.000 | 0.500 | 0.693 | 196 |
| q016 | 架构设计 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 218 |
| q017 | 架构设计 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 213 |
| q018 | 架构设计 | medium | 5 | 0 | 0.000 | 0.000 | 0.000 | 194 |
| q019 | 代码实现 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 198 |
| q020 | 代码实现 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 226 |
| q021 | 代码实现 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 280 |
| q022 | 代码实现 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 231 |
| q023 | 代码实现 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 205 |
| q024 | 代码实现 | easy | 5 | 2 | 1.000 | 1.000 | 0.877 | 198 |
| q025 | 历史背景 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 222 |
| q026 | 历史背景 | easy | 5 | 0 | 0.000 | 0.000 | 0.000 | 192 |
| q027 | 历史背景 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 257 |
| q028 | 历史背景 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 219 |
| q029 | 历史背景 | hard | 5 | 0 | 0.000 | 0.000 | 0.000 | 219 |
| q030 | 历史背景 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 426 |
</details>
