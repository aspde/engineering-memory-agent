# 检索基线存档 — 2026-08-12

## 这次存档是什么

语料修正（5 条过期 seed 重写、12 条 content-only 指纹修复、15 条真实口吻 query 追加）加 **git 历史提炼**（`ingest_repo` 摄入 EMA 30 条 commit → 提炼 8 条新 seed seed-079..086 + 8 条新 query q086..093）之后的**当前基线**，供后续 `compare_baseline.py` 回归对比。本目录所有 `2026-08-12_*.json` / `.md` 由 `python -m tests.eval.run_eval` 生成。

## 运行环境

- 数据库：`localhost:5432/ema_dev`（覆盖 `DATABASE_URL`；`.env` 默认的 `postgres` 是 docker 内网主机名，本机直连需覆盖）；`ema_eval_git`（摄入 30 条真实 commit 记忆的隔离库，见下节）
- 语料：seed **78 条**（70 手写 + 8 git 提炼，`python -m tests.eval.seed --clear --memories` 重灌）；labeled set **92 条 query**
- Embedding：本地 BGE-M3（模型缓存已存在）
- 默认路径 `skip_rerank=True`（生产默认，无 cross-encoder / LLM rerank）
- **环境注意**：`tests/conftest.py` 默认用占位密码 `ema123` 连 `127.0.0.1:5432`，与本机 PostgreSQL 的 ema 密码（`.env` 的 `koxJG…`）不匹配——本机跑 pytest 需 `TEST_DATABASE_URL="postgresql://ema:…@127.0.0.1:5432/ema_test"` 覆盖

## 数字汇总（第三版：92 query / 78-seed 池；前两版数值见下）

| 路径 | 池 | recall@5 | mrr | ndcg@5 | avg_latency |
|------|------|----------|-----|--------|-------------|
| memory:norank | 78 seed | 0.891 | 0.782 | 0.810 | 330ms |
| memory:norank | 78 seed + 30 git | 0.891 | 0.781 | 0.809 | 333ms |
| memory:ce (84q 历史) | 70 seed | 0.917 | 0.811 | 0.837 | 1925ms |
| rewrite:norank (84q) | 70 seed | 0.635 | 0.579 | 0.593 | 14998ms |
| chunk:norank (84q) | 70 seed | 0.635 | 0.592 | 0.603 | 375ms |
| hybrid:norank (84q) | 70 seed | 0.635 | 0.556 | 0.576 | 382ms |

> **第三版（92 query）**：`git ingest_repo` 摄入后，78-seed 池 recall@5=0.891 / mrr=0.782。与上一版（84 query / 0.917）不可直接比——加了 8 条真实 commit 提炼的 query（q086..093，其中 6/8 命中），语料更真实。加入 30 条 git 记忆到检索池（108 池）后数字几乎不变（0.891/0.781），说明 git 记忆未显著稀释目标召回。
>
> **第二版（84 query）**：content-only 指纹 bug 修复后 memory:norank 从 0.800 升到 0.917——12 条 query 的指纹只在 seed `content`、而 memory 检索匹配 `summary`，它们在 memory 路径上永远不可命中（构造 bug，不是检索短板）。CE 排序有小幅提升（mrr 0.787→0.811）但代价 4 倍延迟；`rewrite` 在当前语料下成本收益不佳（recall 更低且 15s/query）。

## git 历史提炼（本轮新增）

用 `ingest_repo(G:/Projects/ema, max_commits=30)` 摄入 30 条真实 commit 到隔离库 `ema_eval_git`（走完整记忆管线：提取 + embed + 相似度分级，`source_type=git_commit`）。从 30 条中**抽检 8 条**有决策/复盘价值的提炼为 eval seed（seed-079..086），配真实口吻 query（q086..093）：

| seed | commit | 主题 | category |
|------|--------|------|----------|
| seed-079 | a1ff1d4b | eval 阈值标定的假 paraphrase bug 修复 | 历史背景 |
| seed-080 | 6732f719 | 真实 DB 凭据误提交事故 | 故障复盘 |
| seed-081 | ce20f282 | force-write 记忆改为 agent 工具注入 | 架构设计 |
| seed-082 | a500748c | 三处并发槽计数收敛为 SlotLimiter | 代码实现 |
| seed-083 | 729bfc94 | ADR-011 广度层按 flag 挂载 | 架构设计 |
| seed-084 | 410cda00 | 测试免 API key / 硬编码凭据 | 代码实现 |
| seed-085 | 8cab42b4 | 熔断器简化重构 | 代码实现 |
| seed-086 | aff7b018 | API 限流（per-key token-bucket） | 架构设计 |

剔除：纯 chore（.gitignore / untracked 测试 / type-cleanup / docs 同步）×7，及被现有 seed 充分覆盖的（衰减移除、use_llm_rerank 锁死、提取标定）×6。验证：`validate_dataset` 0 warnings，新增 8 query 指纹全部唯一且落在目标 seed summary（Check 5）。

## 与旧基线的对比（重要）

历史基线（`tests/eval/reports/memory_path_report.json` 等）是 **30 条自问自答 query** 时代的数字（memory:norank recall@5=1.0, mrr=0.944）。**不是同一份语料，数字不可直接比**。下降的主因是语料本身变难了：

- 新增 15 条 query 全部是真实用户口吻、不含目标指纹原文（不再自问自答）
- 5 条 seed 内容修正后，对应 query 的 ground truth 从"过时描述"变成"当前代码描述"（如 q016 现在要求命中"衰减已移除"的演变，而不是"衰减整合"）

**不要把 0.917 当作"检索变好了"**——语料变难了（新增真实口吻 query + 5 条 seed 修正 + 12 条 content-only 指纹修复），0.917 衡量的是更难、更真实、且 ground truth 可满足的查询集。旧 1.0 是自问自答语料的假满分。

## 当前 miss 分析（memory:norank，92 条中 10 条 miss）

10 条 miss 都是**真实的检索短板**（指纹都在目标 seed summary 里，检索却没召回）：

- **q052**（主模型挂了怎么兜底 / seed-056）：query 过短过泛，"主模型挂了"与 seed 的"FallbackLLMProvider failover"缺少表面词桥梁；而口语化的 q082（同一目标）已命中——说明 query 表述方式决定命中
- **q056**（重复查询为什么不重新算向量 / seed-062，easy）：query 直接但目标 seed-062 的 summary 措辞是"LRU 缓存/SSE 重连"——easy 也 miss，值得优先查
- **q068**（项目现在怎么一键跑起来 / seed-076，easy）：同上，easy 也 miss
- **q070**（对话变快经历了哪几步 / seed-078，hard）：概念问法，目标是 P95 三步走
- **q071**（搜索变慢 / BGE-M3 CPU 瓶颈）：query 与目标无共享实体词，隐式关联
- **q074**（JSON 被代码块包裹 / seed-011）：query 描述现象，目标 summary 是机制描述
- **q079**（聊天卡顿 / rerank 锁死）：同上，现象 vs 机制
- **q082**（主模型要是挂了会不会甩报错 / seed-056）：与 q052 同目标，两个问法都 miss——seed-056 的"重试耗尽或熔断打开"措辞与口语化问法缺表面桥，是同一盲区重复采样
- **q086**（评估报告的相似度数字可信吗 / seed-079，git 提炼）：抽象问"数据可信性"，目标 seed 讲"假 duplicate 污染 merge 带"——query 与目标无共享词
- **q093**（接口被人一直刷有保护吗 / seed-086，git 提炼）：抽象问"自我保护"，目标 seed 讲"token-bucket 限流"——同样无共享实体词

其中 q056/q068 是 easy 级、q086/q093 是 git 提炼 query——按 difficulty 它们本应最容易命中，miss 说明纯向量在"口语化抽象问法 → 代码实现/架构记忆"上的固有盲区（缺表面词桥梁）。这些是 rerank / query 重写 / hybrid 优化的回归对象。per-query 明细见 `2026-08-12_memory_norank_92.json`（78-seed 池）与 `2026-08-12_memory_norank_git.json`（108 池）。

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
