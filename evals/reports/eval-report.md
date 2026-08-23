# EMA Retrieval Evaluation Report

- Generated: 2026-08-07 02:56:49 UTC
- 基于当前语料同源重跑（chunks 于 2026-08-06 07:12 UTC 重新播种；旧版 0.833 baseline 生成于 06:11，早于重新播种，见下方叙事更正）
- Queries: 30
- Configs: 3
- Errors: 0

## Overall

| config | recall@5 | precision@5 | hit_rate@5 | mrr | ndcg@5 | map@5 | latency_ms | n_queries |
|---|---|---|---|---|---|---|---|---|
| chunk:vector@k5 | 1.000 | 0.200 | 1.000 | 0.983 | 0.988 | 0.983 | 748 | 30 |
| hybrid:ce@k5 | 0.967 | 0.193 | 0.967 | 0.917 | 0.930 | 0.917 | 17539 | 30 |
| hybrid:norank@k5 | 1.000 | 0.200 | 1.000 | 0.983 | 0.988 | 0.983 | 190 | 30 |

> **定义**：hybrid = BGE-M3 向量 + jieba 中文分词 BM25 (Jaccard)。`+rerank` = cross-encoder (BGE-reranker-v2-m3 568M) + 0.15 floor；`no rerank` = max(dense sim, sparse jaccard) 排序。
>
> **关键发现（当前语料）**：
> 1. **BGE-M3 稠密召回已达 Recall@5=1.000**——语料足够具体时，sparse/hybrid 不提供召回增益，三条路径 recall 相同。
> 2. **cross-encoder rerank 仍有害**：0.15 floor 误伤 q015（打分 0.142 < 0.15 被过滤），hybrid:ce Recall 0.967 < 跳过 rerank 的 1.000；延迟高 ~90 倍（17.5s vs 0.19s/query，CPU 推理）。
> 3. **叙事更正**：旧报告「dense 0.833 → sparse 救回 1.00」对比不一致——vector baseline（0.833）生成于 08-06 06:11，早于 07:12 的语料重新播种；且旧指纹与 seed 内容不匹配（如 q007 旧指纹「koa-connect wrapper 导致 ctx 泄漏」不在 seed-007 内容中）。当前语料 + 当前指纹下 dense 即 1.000，旧结论不可复现，属语料/标注修订前的历史结果。
> 4. rerank 收益 scale-dependent：万级语料候选池缩到 4%、dense sim 区分度下降时 rerank 才有正收益，生产部署前需重新评估。

## Recall@5 by category

| config | 技术决策 | 故障复盘 | 架构设计 | 代码实现 | 历史背景 |
|---|---|---|---|---|---|
| chunk:vector@k5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid:ce@k5 | 1.000 | 1.000 | 0.833 | 1.000 | 1.000 |
| hybrid:norank@k5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## MRR by category

| config | 技术决策 | 故障复盘 | 架构设计 | 代码实现 | 历史背景 |
|---|---|---|---|---|---|
| chunk:vector@k5 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 |
| hybrid:ce@k5 | 1.000 | 0.833 | 0.833 | 1.000 | 0.917 |
| hybrid:norank@k5 | 1.000 | 1.000 | 0.917 | 1.000 | 1.000 |

## Recall@5 by difficulty

| config | easy | medium | hard |
|---|---|---|---|
| chunk:vector@k5 | 1.000 | 1.000 | 1.000 |
| hybrid:ce@k5 | 1.000 | 0.929 | 1.000 |
| hybrid:norank@k5 | 1.000 | 1.000 | 1.000 |

## hybrid:ce@k5 唯一 miss

| id | query | n_ret | n_rel | recall@5 | 原因 |
|---|---|---|---|---|---|
| q015 | 人工介入 HITL 是怎么实现的 | 3 | 0 | 0.000 | 目标 chunk 在候选池内（dense rank-2、sparse 亦命中），但 cross-encoder 打分 0.142 < `_RERANK_FLOOR` 0.15，被 floor 滤掉 |

> q015 是「rerank 在小规模语料下有害」的直接证据：稠密与稀疏都召回了目标，唯独 cross-encoder + 0.15 floor 把低分相关结果滤掉了。跳过 rerank 后该查询在 top-5（MRR=0.500，见 chunk:vector 明细）。

## A/B comparison

### Δ hybrid:ce@k5 − chunk:vector@k5

| metric | chunk:vector@k5 | hybrid:ce@k5 | Δ |
|---|---|---|---|
| recall@5 | 1.000 | 0.967 | -0.033 |
| precision@5 | 0.200 | 0.193 | -0.007 |
| hit_rate@5 | 1.000 | 0.967 | -0.033 |
| mrr | 0.983 | 0.917 | -0.067 |
| ndcg@5 | 0.988 | 0.930 | -0.058 |
| map@5 | 0.983 | 0.917 | -0.067 |
| latency_ms | 748 | 17539 | +16791 |

### Δ hybrid:norank@k5 − hybrid:ce@k5

| metric | hybrid:ce@k5 | hybrid:norank@k5 | Δ |
|---|---|---|---|
| recall@5 | 0.967 | 1.000 | +0.033 |
| precision@5 | 0.193 | 0.200 | +0.007 |
| hit_rate@5 | 0.967 | 1.000 | +0.033 |
| mrr | 0.917 | 0.983 | +0.067 |
| ndcg@5 | 0.930 | 0.988 | +0.058 |
| map@5 | 0.917 | 0.983 | +0.067 |
| latency_ms | 17539 | 190 | -17349 |

<details><summary>Per-query detail (chunk:vector@k5)</summary>

| id | category | difficulty | n_ret | n_rel | recall@5 | mrr | ndcg@5 | latency_ms |
|---|---|---|---|---|---|---|---|---|
| q001 | 技术决策 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 17047 |
| q002 | 技术决策 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 189 |
| q003 | 技术决策 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 177 |
| q004 | 技术决策 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 199 |
| q005 | 技术决策 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 194 |
| q006 | 技术决策 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 199 |
| q007 | 故障复盘 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 178 |
| q008 | 故障复盘 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 186 |
| q009 | 故障复盘 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 182 |
| q010 | 故障复盘 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 194 |
| q011 | 故障复盘 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 190 |
| q012 | 故障复盘 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 178 |
| q013 | 架构设计 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 174 |
| q014 | 架构设计 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 198 |
| q015 | 架构设计 | medium | 5 | 1 | 1.000 | 0.500 | 0.631 | 182 |
| q016 | 架构设计 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 191 |
| q017 | 架构设计 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 190 |
| q018 | 架构设计 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 170 |
| q019 | 代码实现 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 186 |
| q020 | 代码实现 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 183 |
| q021 | 代码实现 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 190 |
| q022 | 代码实现 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 194 |
| q023 | 代码实现 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 186 |
| q024 | 代码实现 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 182 |
| q025 | 历史背景 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 182 |
| q026 | 历史背景 | easy | 5 | 1 | 1.000 | 1.000 | 1.000 | 178 |
| q027 | 历史背景 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 186 |
| q028 | 历史背景 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 195 |
| q029 | 历史背景 | hard | 5 | 1 | 1.000 | 1.000 | 1.000 | 174 |
| q030 | 历史背景 | medium | 5 | 1 | 1.000 | 1.000 | 1.000 | 186 |
</details>
