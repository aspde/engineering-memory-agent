# 检索基线存档 — 2026-08-12

## 这次存档是什么

语料修正（5 条过期 seed 重写、新增 15 条真实口吻 query）之后的**当前基线**，供后续 `compare_baseline.py` 回归对比。本目录所有 `2026-08-12_*.json` / `.md` 由 `python -m tests.eval.run_eval` 生成。

## 运行环境

- 数据库：`localhost:5432/ema_dev`（覆盖 `DATABASE_URL`；`.env` 默认的 `postgres` 是 docker 内网主机名，本机直连需覆盖）
- 语料：seed 70 条（`python -m tests.eval.seed --clear --memories` 重灌，反映修正后的当前内容）；labeled set 84 条 query（初始 85，删除了与 q018 目标重复的 q076）
- Embedding：本地 BGE-M3（模型缓存已存在）
- 默认路径 `skip_rerank=True`（生产默认，无 cross-encoder / LLM rerank）

## 数字汇总（全部 @k5, 0 errors；n=84，删除了与 q018 目标重复的 q076）

| 路径 | recall@5 | precision@5 | mrr | ndcg@5 | map@5 | avg_latency |
|------|----------|-------------|-----|--------|-------|-------------|
| memory:norank | 0.917 | 0.178 | 0.787 | 0.820 | 0.787 | 439ms |
| memory:ce | 0.917 | 0.178 | 0.811 | 0.837 | 0.811 | 1925ms |
| rewrite:norank | 0.635 | 0.165 | 0.579 | 0.593 | 0.579 | 14998ms |
| chunk:norank | 0.635 | 0.165 | 0.592 | 0.603 | 0.592 | 375ms |
| vector:norank | 0.635 | 0.165 | 0.592 | 0.603 | 0.592 | 360ms |
| hybrid:norank | 0.635 | 0.165 | 0.556 | 0.576 | 0.556 | 382ms |
| hybrid_norerank:norank | 0.635 | 0.165 | 0.556 | 0.576 | 0.556 | 399ms |

> **2026-08-12 修正后的第二版**：初版 memory:norank 是 0.800，重跑为 0.917。原因是一次 ground-truth 构造 bug 被修掉（见下节）——12 条 query 的指纹只存在于目标 seed 的 `content`、不在 `summary`，而 memory 检索路径匹配 `summary`（`dataset.make_memory_retriever` 的 `match_field`），这些 query 在 memory 路径上**永远不可命中**。修正后它们大多命中（检索本就能找到目标）。

观察：`memory:ce`（bounded top-3 cross-encoder）recall 持平、mrr 0.787→0.811 / ndcg 0.820→0.837，排序有小幅提升，代价是 4 倍延迟（439→1925ms）。`rewrite`（LLM 多路改写）recall 反而低且平均 15s/query——当前语料下成本收益不佳。chunk/vector/hybrid 的 0.635 低于 memory 的 0.917，因为 chunk 路径匹配 content（seed 的 `content` 是原始长文，指纹是 summary 精炼短语，content 内子串命中更难）——不是同一套指纹直接可比，见下方 miss 分析。

## 与旧基线的对比（重要）

历史基线（`tests/eval/reports/memory_path_report.json` 等）是 **30 条自问自答 query** 时代的数字（memory:norank recall@5=1.0, mrr=0.944）。**不是同一份语料，数字不可直接比**。下降的主因是语料本身变难了：

- 新增 15 条 query 全部是真实用户口吻、不含目标指纹原文（不再自问自答）
- 5 条 seed 内容修正后，对应 query 的 ground truth 从"过时描述"变成"当前代码描述"（如 q016 现在要求命中"衰减已移除"的演变，而不是"衰减整合"）

**不要把 0.917 当作"检索变好了"**——语料变难了（新增真实口吻 query + 5 条 seed 修正 + 12 条 content-only 指纹修复），0.917 衡量的是更难、更真实、且 ground truth 可满足的查询集。旧 1.0 是自问自答语料的假满分。

## 当前 miss 分析（memory:norank，84 条中 7 条 miss）

指纹修正后剩下的 7 条 miss 是**真实的检索短板**（指纹都在目标 seed 的 summary 里，检索却没召回）：

- **q052**（主模型挂了怎么兜底 / seed-056）：query 过短过泛，"主模型挂了"与 seed 的"FallbackLLMProvider failover"缺少表面词桥梁；而口语化的 q082（同一目标）已命中——说明 query 表述方式决定命中
- **q056**（重复查询为什么不重新算向量 / seed-062，easy）：query 直接但目标 seed-062 的 summary 措辞是"LRU 缓存/SSE 重连"——easy 也 miss，值得优先查
- **q068**（项目现在怎么一键跑起来 / seed-076，easy）：同上，easy 也 miss
- **q070**（对话变快经历了哪几步 / seed-078，hard）：概念问法，目标是 P95 三步走
- **q071**（搜索变慢 / BGE-M3 CPU 瓶颈）：query 与目标无共享实体词，隐式关联
- **q074**（JSON 被代码块包裹 / seed-011）：query 描述现象，目标 summary 是机制描述
- **q079**（聊天卡顿 / rerank 锁死）：同上，现象 vs 机制

其中 q056/q068 是 easy 级——按 difficulty 分布它们本应最容易命中，miss 说明纯向量在"口语化问法 → 代码实现记忆"上的固有盲区。这些是 rerank / query 重写 / hybrid 优化的回归对象。per-query 明细见 `2026-08-12_memory_norank.json`。

## 复现命令

```bash
# 重灌语料（seed 内容变更后）
DATABASE_URL="postgresql://ema:...@localhost:5432/ema_dev" \
  python -m tests.eval.seed --clear --memories

# 跑基线（每路径一条）
DATABASE_URL="postgresql://ema:...@localhost:5432/ema_dev" \
  python -m tests.eval.run_eval --retriever memory \
  --report-json tests/eval/reports/baseline/2026-08-12_memory_norank.json \
  --report-md   tests/eval/reports/baseline/2026-08-12_memory_norank.md
```

## 限制

- **未跑 `memory:llm`（LLM 点式 rerank）**。理由：① 非生产路径——`use_llm_rerank` 在生产对话里已被工具 schema 锁死（对话 P95 优化），仅 API 显式调用者与 eval 可用；② 项目已有稳定的 A/B 结论——LLM rerank 不改变 recall@5、比 bounded CE 慢 5.5 倍、成本占对话 46%（见 seed-034/041/043/051）；③ 全量成本 ~1700 次 LLM 调用（85 queries × 20 候选），为复现已知结论不划算。如需上限参考，可用 `--category` 跑单类目小样本。
- `ema_dev` 是开发库，memories 表含 8 条非 eval_seed 的真实记忆，作为天然噪声池（比纯 seed 更接近真实环境）。
