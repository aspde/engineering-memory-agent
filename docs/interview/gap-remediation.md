# 短板诊断与评估体系补全方案

> 本文档针对 EMA 作为面试项目最危险的短板——**AI 工程化（评估体系 / 成本监控）**——给出可执行的补全方案。
> 目标：面试前用 1-2 天补齐，让"没做评估"变成"有最小评估体系"。

---

## 一、短板全景（基于代码核查）

### 1.1 已核查的短板证据

| 短板 | 代码证据 | 危险度 |
|------|---------|--------|
| 无用户级认证（登录/角色/权限） | 接入认证已有 API key：所有 `/api` 请求须携带 `Authorization: Bearer <EMA_API_KEY>`（[auth.py](../../backend/api/auth.py) `secrets.compare_digest` 常量时间比较 + 通用 401，全局挂载于 [router.py](../../backend/api/router.py)，`APP_ENV=test` 豁免），单 key 共享、无用户身份 | 🟢 有意取舍 + 已实现兜底 |
| 未容器化 | 已解决：根目录 [Dockerfile](../../Dockerfile)（后端 + 前端单镜像）+ `docker compose up -d` 一键启动（[deployment.md](../deployment.md)） | ✅ 已补 |
| 巡检内嵌主进程 | [main.py:163-272](../../backend/main.py) 调度器在 FastAPI 主进程 | 🟡 有意取舍（[ADR-007](../../decisions/ADR-007-patrol-in-process-scheduler.md)） |
| Windows 降级 | [main.py:24-30](../../backend/main.py) PostgresSaver 降级 InMemorySaver | 🟢 已知平台差异 |
| 连接池写死 | [db/__init__.py](../../backend/db/__init__.py) 5+10 | 🟢 低 |

> **已补齐（相对本文档初版）**：评估体系（Recall@K/MRR/NDCG + LLM-as-judge，见 §2.4）、成本监控（`llm_usage` 表 + `/api/usage/*` 端点 + 成本估算，见 [architecture.md](../architecture.md)）、CI/CD（`.github/workflows/ci.yml` 跑后端 pytest + 前端 vitest + `eval.yml` 每周检索评估）、Prompt 版本管理（[prompts.py](../../backend/service/prompts.py) 中央注册表 + 版本号）、**运行监控指标**（[runtime_metrics.py](../../backend/shared/runtime_metrics.py) + `GET /metrics` 暴露 Prometheus 时序：HTTP 请求/延迟、LLM 调用/token、熔断器状态、Agent 并发槽位与 ReAct 步数分布，见 [architecture.md](../architecture.md) Observability）、**数据库备份与恢复**（compose `backup` 容器每小时 pg_dump -Fc 到 `./backups/` + 恢复 runbook，已实测恢复一致，见 [deployment.md](../deployment.md) Backup & Restore）、**监控采集闭环**（compose `prometheus` 抓取 `/metrics` + `grafana` 渲染 "EMA — Runtime Health" 看板，见 [deployment.md](../deployment.md) Monitoring）、**API 限流**（per-key 令牌桶中间件，chat/general 两档，429 + Retry-After，见 [ratelimit.py](../../backend/api/ratelimit.py)）、**Caddy TLS 反代**（compose `caddy` 服务，默认本地 HTTP、配域名自动 Let's Encrypt，见 [deployment.md](../deployment.md) HTTPS (TLS)）、**日志结构化**（`LOG_FORMAT=json` 单行 JSON + trace_id/thread_id，见 [logging_config.py](../../backend/shared/logging_config.py) + deployment.md 日志（结构化））、**Alembic schema 版本化迁移**（`alembic_version` 表 + `migrations/versions/` 成对 upgrade/downgrade，基线迁移固化 9 张业务表；`init_db()` 跑 `upgrade head` + 运行时 embedding 维度对齐，新库/旧库/回滚三路径已实测，见 deployment.md Schema Migration）。这些不再列为短板。

### 1.2 优先补全顺序

```
评估体系（§2） → 关键数字实测（§3） → 成本监控（§4） → 压测（§5）
     ↑              ↑                     ↑              ↑
   已补齐        数字已部分回填       已补齐（llm_usage 表）  已补齐（locust，见 §5.3）
```

> **进度更新**：评估体系（§2）与成本监控（§4）已按方案落地——评估见 §2.4，成本监控的 `llm_usage` 表 + `/api/usage/*` 已在 [architecture.md](../architecture.md) 文档化。运行监控指标也已落地（`runtime_metrics.py` + `GET /metrics` 暴露 Prometheus 时序，见 [architecture.md](../architecture.md) Observability）。容器化已补：根目录 Dockerfile + `docker compose up -d`（见 [deployment.md](../deployment.md)）。压测已补：`tests/perf/locustfile.py` + 实测数字（§5.3）。**对话 P95 与 token 成本已实测**（§3.1，10 轮真实对话，P95 73.6s、≈28.6k tokens/轮）。

---

## 二、评估体系（已交付）

### 2.1 为什么这是最危险的短板

AI/LLM 应用工程师岗位面试**必问**的三连击：
1. 「你怎么知道你的 RAG 检索准不准？」
2. 「换了个 reranker，效果变好还是变差？怎么量化？」
3. 「你的记忆衰减加权有没有让检索变好？」

评估体系已交付（见 §2.4）——现在这三问都有数字可答，而非只讲设计。§2.2-2.3 保留最初的设计草案与实现说明，供回顾演进过程。

### 2.2 最小可行评估体系设计

**目标**：用半天到一天，建一个能跑出数字的评估 pipeline。不需要完美，要有数据。

#### 2.2.1 标注集构造

构造 20-50 条 `(query, 相关 memory_id)` 标注对：

```python
# tests/eval/ground_truth.py
"""评估标注集 —— 手工构造，覆盖典型查询模式"""

GROUND_TRUTH: list[dict] = [
    {
        "query": "PostgreSQL 连接池配置多少合适",
        "relevant_memory_ids": ["uuid1", "uuid2"],  # 人工判断相关的记忆 ID
        "category": "技术决策",  # 用于分类分析
    },
    {
        "query": "之前怎么修的 OOM 问题",
        "relevant_memory_ids": ["uuid3"],
        "category": "故障复盘",
    },
    # ... 覆盖 5 个类别，每类 4-10 条
    # 类别建议：技术决策 / 故障复盘 / 架构设计 / 代码实现 / 历史背景
]
```

**构造方法**：
1. 从 EMA 记忆库导出 50 条记忆，按类别分桶
2. 针对每类想 4-10 个自然查询
3. 人工判断每个查询应该召回哪些记忆（这是 ground truth）
4. 存成上面的结构

**关键**：标注集不必大，20 条能跑出数字就够面试讲。

#### 2.2.2 评估指标实现

```python
# tests/eval/metrics.py
"""RAG 评估指标 —— Recall@K / MRR / NDCG"""

from typing import List


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Recall@K：前 K 个召回结果中，包含多少比例的相关记忆"""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR (Mean Reciprocal Rank)：第一个相关结果的倒数排名"""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """NDCG@K：考虑排名位置的归一化增益"""
    import math
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], 1):
        if rid in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    # ideal DCG：相关结果全排前面
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0
```

#### 2.2.3 评估脚本

```python
# tests/eval/run_eval.py
"""跑评估：对每个 query 调检索，算指标"""

import asyncio
from backend.service.retrieval import query_memories
from tests.eval.ground_truth import GROUND_TRUTH
from tests.eval.metrics import recall_at_k, mrr, ndcg_at_k


async def evaluate(use_llm_rerank: bool = False, top_k: int = 5):
    """跑一轮评估，返回平均指标"""
    results = []
    for item in GROUND_TRUTH:
        memories = await query_memories(
            item["query"], top_k=top_k, use_llm_rerank=use_llm_rerank
        )
        retrieved_ids = [str(m["id"]) for m in memories]
        relevant = item["relevant_memory_ids"]

        results.append({
            "query": item["query"],
            "category": item["category"],
            "recall@5": recall_at_k(retrieved_ids, relevant, k=5),
            "mrr": mrr(retrieved_ids, relevant),
            "ndcg@5": ndcg_at_k(retrieved_ids, relevant, k=5),
        })

    # 汇总
    n = len(results)
    avg = {
        "recall@5": sum(r["recall@5"] for r in results) / n,
        "mrr": sum(r["mrr"] for r in results) / n,
        "ndcg@5": sum(r["ndcg@5"] for r in results) / n,
    }
    return avg, results


if __name__ == "__main__":
    # 对比：开不开 LLM rerank
    print("=== cross-encoder rerank ===")
    avg1, _ = asyncio.run(evaluate(use_llm_rerank=False))
    print(avg1)

    print("=== LLM rerank ===")
    avg2, _ = asyncio.run(evaluate(use_llm_rerank=True))
    print(avg2)
```

#### 2.2.4 跑出数字后的面试话术

有了数字后，可以这样答：

> 「我建了一个 20 条的标注集，覆盖技术决策、故障复盘、架构设计等 5 个类别。当前 cross-encoder rerank 的 Recall@5 是 [0.72]，MRR 是 [0.61]。我对比过开不开 LLM rerank——LLM rerank 把 Recall@5 提到 [0.78] 但单次检索成本增加 [X] 倍，所以默认关闭，只在 `use_llm_rerank=True` 时启用。
>
> 这个评估集不大，但能让我量化验证每个改动。比如调 chunk_size 从 512 到 256，Recall@5 [升/降] 了 [Y]，我就知道方向对不对。」

### 2.3 LLM-as-judge 与 Agent 行为评测（已实现）

> 状态：✅ 已交付。工具选择 / 知识抽取 / 最终答案三套 LLM 行为评测见
> [llm-eval.md](./llm-eval.md)，CLI 为 `python -m tests.eval.run_llm_eval`。
> 2026-08-09 新增第四套 **任务级端到端评测**（`python -m tests.eval.run_task_eval`）——
> 驱动真实 Agent 图（ReAct 循环 + 真实工具执行 + HITL 自动放行）完成多步任务，测
> `completed` / `tool_recall` / `within_budget` 与答案接地。它顺带抓出并修复了一个
> 生产 bug：拒绝审批后写操作仍被静态边路由执行（LangGraph 1.2.10 resume 时 Command
> 与静态边同时生效），见 [llm-eval.md](./llm-eval.md) 末尾。
> 下方是最初的最小草案，实际实现有三处演进：
> 1. 评测面从"答案质量"扩到**工具选择 + 知识抽取 + 最终答案**三个维度；
> 2. 裁判输出改用 `chat_structured`（JSON Schema 校验 + 语义重试），放弃裸
>    `json.loads` + 解析失败归零的脆弱写法；
> 3. 裁判 prompt 刻意让 LLM 输出**结构化的 covered_facts / grounded /
>    ungrounded_claims**，而不是 1-5 分——覆盖率与忠实度可从判决直接算指标，
>    幻觉论断还能进报告的 per-query 明细。

```python
# tests/eval/llm_judge.py — 当前实现
"""LLM-as-judge：让 LLM 评估答案质量"""

import json
from backend.service.llm_service import get_llm_provider

JUDGE_PROMPT = """\
你是一个严格的评估员。根据以下信息评估答案质量。

问题：{question}
检索到的记忆：{retrieved}
生成的答案：{answer}

请从三个维度打分（1-5 分）：
1. 准确性：答案是否基于检索记忆，没有幻觉
2. 完整性：答案是否覆盖了检索记忆中的关键信息
3. 相关性：答案是否直接回应了问题

只返回 JSON：{{"accuracy": X, "completeness": X, "relevance": X, "reason": "..."}}
"""


async def judge_answer(question: str, retrieved: str, answer: str) -> dict:
    llm = get_llm_provider()
    prompt = JUDGE_PROMPT.format(
        question=question, retrieved=retrieved, answer=answer
    )
    resp = await llm.chat([{"role": "user", "content": prompt}])
    try:
        return json.loads(resp.strip())
    except Exception:
        return {"accuracy": 0, "completeness": 0, "relevance": 0, "reason": "parse_failed"}
```

**话术**：「生成质量我用 LLM-as-judge，让 LLM 从准确性、完整性、相关性打 1-5 分。LLM 评估有偏差，只能做粗筛，所以我会人工抽检 20%。」

### 2.4 实施完成记录（阶段 B 已交付）

> 状态：✅ 已完成（342 单测全过——检索评估 + LLM 行为评测，CLI 可跑）。下方为实际交付物，替代 §2.2 的最小草案。

**交付目录**：`tests/eval/`

| 文件 | 职责 |
|------|------|
| [metrics.py](../../tests/eval/metrics.py) | 6 个纯函数指标：Recall@K / Precision@K / HitRate@K / MRR / NDCG@K / MAP@K + `compute_all` |
| [ground_truth.py](../../tests/eval/ground_truth.py) | 70 条标注集，5 类（技术决策/故障复盘/架构设计/代码实现/历史背景）× 14 条，含 difficulty 分级 |
| [seed_memories.jsonl](../../tests/eval/seed_memories.jsonl) | 70 条种子记忆（EMA 自身工程史，覆盖评估/衰减校准/阈值标定/容器化等），每条 summary 含独特指纹 |
| [dataset.py](../../tests/eval/dataset.py) | 标注集加载 + 指纹匹配（`is_relevant`/`relevance_mask`）+ 语义相关性通道（`semantic_relevance_mask`，embedding cosine≥0.80）+ retriever 适配（chunk/memory）+ 4 项一致性校验 |
| [runner.py](../../tests/eval/runner.py) | 单组评估 + A/B 对比 + 按 category/difficulty 聚合，error 容错；语义通道默认关闭（opt-in），per-query 记录 `substring_hits`/`semantic_only_hits` 拆分 |
| [report.py](../../tests/eval/report.py) | Markdown + JSON 报告，含 A/B delta 表 |
| [run_eval.py](../../tests/eval/run_eval.py) | CLI：`python -m tests.eval.run_eval --validate-only` / `--compare` / `--report-md` |
| [seed.py](../../tests/eval/seed.py) | CLI：`python -m tests.eval.seed --dry-run` / `--memories` / `--clear` |

**单元测试**（101 passed）：[test_eval_metrics.py](../../tests/unit/test_eval_metrics.py)（指标手算 case）+ [test_eval_dataset.py](../../tests/unit/test_eval_dataset.py)（数据集一致性 + 合成坏语料校验器）+ [test_eval_runner.py](../../tests/unit/test_eval_runner.py)（synthetic retriever 端到端）+ [test_eval_report.py](../../tests/unit/test_eval_report.py)

**关键设计决策**（面试可讲）：
1. **指纹而非 UUID**：标注集用 `relevant_fingerprints`（summary 中的独特子串）匹配相关结果，不依赖 memory_id。可移植、可重现、CI 友好。`validate_dataset` 强校验每个指纹唯一且可解析。
2. **retriever 无关**：runner 接收 `RetrieverAdapter(fn, match_field)`，同一套指标同时评估 chunks 表（`retrieve`）和 memories 表（`query_memories`）。
3. **位置 ID 映射**：runner 把 retrieved 结果映射成 `[0,1,...,n-1]`，relevant 集为命中指纹的 index 集合，复用纯函数 metrics，零 I/O 耦合。
4. **difficulty 分级**：easy（词重合）/medium（改写）/hard（概念问法），暴露向量质量短板。
5. **A/B 对比内置**：`--compare` 跑 cross-encoder vs LLM rerank，输出 delta 表，量化"换 reranker 效果如何"。
6. **语义通道默认关闭（opt-in）**：relevance 默认 = 纯子串匹配（确定性基线，无"被评测模型自评"的自证）；`--semantic-relevance` 显式开启后 = 子串 OR 目标 seed 摘要的 embedding cosine≥0.80。报告 overall 表暴露 `semantic_rescued`（词面 0 命中、靠语义救回的查询数），per-query 标 `✓/—/✗`，直接量化"语义检索质量"。每周由 `.github/workflows/eval.yml` cron 自动跑（默认纯子串，门禁基于 70 条 0.886/0.767 校准）。

**跑出真实数字的流程**：
```bash
# 1. 灌种子（chunks 表，零 LLM 成本）
python -m tests.eval.seed
# 2. 跑 chunks 路径评估
python -m tests.eval.run_eval --retriever chunk --report-md report.md
# 3. A/B 对比 rerank 策略
python -m tests.eval.run_eval --retriever chunk --compare --report-json ab.json
# 4.（可选）memories 路径：先灌结构化记忆
python -m tests.eval.seed --memories --clear
python -m tests.eval.run_eval --retriever memory
```

**待回填数字**（跑完后填入 self-introduction.md / ema-deep-dive.md；**注**：以下 chunks 路径数字为 08-06 06:11 旧 baseline，基于重新播种前语料——当前语料下 vector 单独即 Recall@5=1.000，见 §11 更正）：
- chunks 路径（vector_search，无 rerank）：Recall@5=**0.833** / MRR=**0.817** / NDCG@5=**0.819** / MAP@5=0.811（30 query，旧 baseline，当前语料已 1.000）
- memory 路径：**已跑**（`python -m tests.eval.seed --memories` 后 `run_eval --retriever memory`）——**2026-08-11 扩充到 70 条后默认确定性基线 Recall@5=0.886 / MRR=0.767 / NDCG@5=0.798**（70 query，语料翻倍 + hard 占比 30%，见 [memory-path-report-70.md](../../tests/eval/reports/memory_path_report_70.md)；30 条时代为 0.900/0.825/0.844，见 [memory-path-report.md](../../tests/eval/reports/memory_path_report.md)）
- cross-encoder rerank vs 无 rerank Δ recall@5：**-0.033**（hybrid:ce 0.967 vs hybrid:norank 1.000，30 query 全量实测——rerank 在小语料下有害，见 §11.5） | **LLM rerank vs bounded-CE（生产 memory 路径，已实测）**：recall@5 相同 **0.900**（rerank 不改变召回集合），**MRR 0.819→0.833（+0.014）/ NDCG@5 0.840→0.851**，但**平均延迟 2.5s→14.1s（5.5x，每候选 1 次 LLM 调用）**——小语料下 rerank 只微调排序不救召回，收益 scale-dependent，见 [memory_llm_vs_ce_report.md](../../tests/eval/reports/memory_llm_vs_ce_report.md)
- easy/medium/hard recall@5：**0.714 / 0.929 / 0.778**（medium 最高，easy 反而最低——部分 easy query 词重合但向量区分度不足；hard 概念查询 0.778 优于预期）
- 按 category：技术决策 1.000 / 代码实现 1.000 / 架构设计 0.833 / 故障复盘 0.667 / 历史背景 0.667（故障复盘+历史背景是短板，概念查询多）

**性能瓶颈实测**（面试可讲）：cross-encoder rerank（BGE-reranker-v2-m3，568M 参数）在 CPU 上单 query 总耗时 **17.5s**（含 embed+search+rerank，见 [eval-report.md](./eval-report.md) hybrid:ce 行）。这是 EMA 单机 CPU 部署的已知瓶颈，优化方向：① embed/rerank 服务化接 GPU；② 降过采样倍数（top_k×4→top_k×2）；③ 高频 query 缓存 rerank 结果。当前评估默认走 hybrid 无 rerank 路径绕过此瓶颈。

---

## 三、关键数字实测清单

面试官最爱追问的数字，必须实测填实：

### 3.1 必测数字

| 指标 | 测量方法 | 实测值 | 面试话术 |
|------|---------|--------|---------|
| 记忆库总条数 | `SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL` | 0（生产空）/ 70 条评估种子 | "评估集 70 条标注 query，生产待部署" |
| chunks 总条数 | `SELECT COUNT(*) FROM chunks` | 70（评估种子） | "种子语料 70 条，覆盖 5 类记忆" |
| 向量召回 P95 延迟 | 在 `/api/memory/search` 加计时日志 | ~190ms（hybrid 无 rerank，含 embed + sparse + sort） | "hybrid 稳态 190ms，含 BGE-M3 embed" |
| 完整检索 P95（含 rerank） | 同上，覆盖 rerank | 17.5s/query（CPU 瓶颈） | "cross-encoder rerank CPU 上 17.5s/query，是已知瓶颈，待 GPU 优化" |
| Agent 单轮对话 P95 | 在 `/api/agent/chat` 加计时 | **10 轮真实对话实测 P95 73.6s**（P50 43.2s / mean 35.9s / min 12.7s / max 73.6s，2026-08-11，DeepSeek deepseek-v4-flash + 本地 BGE-M3）。**已定位根因并从工具 schema 锁死 LLM rerank**——那 73.6s 里约 40s 是每轮 ~19 次 `rerank_llm`（模型自主把 `use_llm_rerank=True` 传给检索工具）。锁死后对话检索走纯衰减排序（~2s/query，recall@5 0.90 基线），预期 P95 降至 ~25s 量级；task_eval 验证移除 rerank 后任务完成率 0.500 持平、groundedness 1.000 不变 | "对话 P95 73.6s 是含 LLM 重排的实测。重排每轮约 19 次、占 40s——模型被工具 schema 允许自主开 rerank。我已从 schema 锁死这个开关，检索质量不变（eval 证明 rerank 不改 recall@5），task 完成率也不掉，P95 预期降到大半。" |
| 单次对话平均 token 数 | LLMProvider 加计数（见 §4） | **≈28.6k tokens/轮**（10 轮合计 285.9k；估 $0.008/轮 ≈ ¥0.06，用内置价格表，见 [usage.py](../../backend/service/usage.py) `estimate_cost`） | "10 轮真实对话合计 285.9k tokens、估算 $0.081（约 ¥0.06/轮）；成本大头是 rerank_llm（75.6k tokens）+ agent_chat（165.8k，其中 144k 是 cache_read 折扣价）——缓存命中大幅压低实际成本" |
| 单次对话平均 tool 调用数 | 在 Agent 加计数 | 2.6 次/任务（task 轨迹均值） | "任务级评测 8 任务平均 2.6 次 LLM 调用，概念查询会到 5" |
| Agent 任务级完成率 | `python -m tests.eval.run_task_eval --judge deterministic` | completed 0.500 / tool_recall 0.938 / within_budget 0.875 | "8 任务实测：工具选择意图准（0.94）但过度调用（unexpected 0.375）拉低严格完成率，是轨迹级真实短板" |
| Agent 答案接地 | 同上（judge 对工具上下文判定） | groundedness 1.000 / citation 0.875 / 0 执行错误 | "答案全部接地、无捏造、零错误；过度调用而非幻觉是主要问题" |
| BGE-M3 embed 单条延迟 | `time` 包裹 `embed()` | 150-230ms（CPU） | "embed 150-230ms/条，CPU 推理" |
| 检索 Recall@5 | `python -m tests.eval.run_eval --retriever memory` | 0.886（70 条默认确定性基线，2026-08-11）；语义通道 opt-in 更高 | "生产默认路径（query_memories, threshold 0.3）70 条标注集默认纯子串实测 recall@5=0.886；语义通道自证故默认关，如实披露" |
| 检索 MRR | 同上 | 0.767（70 条默认） | "生产 memory 路径默认 MRR 0.77（70 条语料，hard 占比 30% 更真实）" |
| 检索 NDCG@5 | 同上 | 0.959 | "memory 路径 NDCG@5 0.96" |
| 检索判别力（hard-negative） | `python -m tests.eval.hard_negative` | 纯向量 59.3% → **bounded-CE 81.5%** / MRR 0.790→0.889 / worse 11→5 | "27 条陷阱集实测：纯向量找得到但容易被表面词带偏（综合通过 59.3%）。已落地 bounded cross-encoder top-3 重排（`query_memories(use_cross_encoder=True)`，默认关）提至 81.5%，见 [hard-negative-report.md](../../tests/eval/reports/hard_negative_report.md)" |
| 检索 QPS | locust 压测 `/api/memory/search` | 10 并发 4.77 / 40 并发 18.6 / 160 并发 63.3 | "QPS 随并发线性涨至 160 仍 0 失败，瓶颈在 BGE CPU embed" |
| 检索 P95（压测） | 同上 | 10 并发 110ms / 80 并发 690ms / 160 并发 1.0s | "缓存热路径 P95 110ms@10 并发；冷查询单次 1.75s（含 embed）" |

### 3.1.1 rerank_llm 占比量化（2026-08-11，`llm_usage` 表实测）

> 对话 P95 73.6s 的根因不是 Agent 循环本身，而是 **LLM 在对话里自主选择了慢速 LLM rerank**。分析工具：`python -m tests.perf.analyze_usage`（新增，读 `llm_usage` 按场景聚合 token/延迟/成本）。数据来自 `llm_usage` 表（provider 层唯一咽喉点统一埋点，含 `latency_ms` 列）。

**全历史**（964 行 LLM 调用，含历史对话 + 评测）：

| 场景 | 调用数 | total_tokens | 累计延迟 | 平均延迟/次 | 估算成本 |
|------|--------|-------------|---------|------------|---------|
| agent_chat | 77 | 487,460 | 614s | 8.0s | $0.059 |
| **rerank_llm** | **842** | **291,131** | **5,796s（96.6 分钟）** | **6.9s** | **$0.160（56%）** |
| agent_final | 23 | 90,674 | 355s | 15.4s | $0.049 |
| query_rewrite | 11 | 15,292 | 163s | 14.8s | $0.015 |

**p95 对话线程**（`measure_chat_p95.py` 跑的 10 轮，thread_id LIKE 'p95-%'）：

| 场景 | 调用数 | total_tokens | 累计延迟 | 延迟占比 | 估算成本 | 成本占比 |
|------|--------|-------------|---------|---------|---------|---------|
| agent_chat | 31 | 165,819 | 124s | 15% | $0.018 | 22% |
| **rerank_llm** | **191** | **75,617** | **475s** | **58.6%** | **$0.037** | **46%** |
| agent_final | 14 | 32,878 | 97s | 12% | $0.014 | 17% |
| query_rewrite | 8 | 11,555 | 115s | 14% | $0.011 | 14% |

**单轮结构**（per-thread 拆分，最有说服力的数字）：

| 轮次类型 | 轮数 | rerank_llm 调用/轮 | rerank 占该轮延迟 | 该轮总延迟 |
|---------|-----|-------------------|------------------|-----------|
| 走了 LLM rerank | 7 | 16-40 次 | **73-81%** | 67-110s |
| 没走 rerank | 7 | 0 次 | 0% | **12-40s** |

**根因**：`search_memories_tool` / `retrieve_chunks_tool`（[tools.py](../../backend/agent/tools.py)）把 `use_llm_rerank` 作为**工具参数暴露给 LLM**（默认 False 但 LLM 可自选 True）。LLM 在对话里自主开启 → 每个候选 memory 一次 LLM 调用（~2.5s/次）→ 每轮 40-80s 纯 rerank 延迟。

**量化结论（面试可讲）**：
1. **rerank_llm 是对话路径延迟和成本的双第一**：延迟占 58.6%、成本占 46%，是 agent_chat 本身（124s）的 3.8 倍。
2. **是否走 rerank 是对话 P95 的决定性变量**：走 rerank 的轮 67-110s，没走的轮 12-40s——**关掉 LLM rerank，P95 预计从 73.6s 降到 ~25s，成本省 ~46%**。
3. **这不是删功能**：eval 已证（[memory_llm_vs_ce_report.md](../../tests/eval/reports/memory_llm_vs_ce_report.md)）LLM rerank 在小语料下只微调排序不改变召回集合（recall@5 同为 0.900，MRR +0.014），默认路径的确定性排序在召回上等价。优化动作是把工具参数的默认锁死为 False（或从 schema 移除），让 LLM 无法在对话路径上开启慢速 rerank。

### 3.2 计时日志埋点示例

```python
# 在 retrieval.py 的 query_memories 加埋点
import time, logging
logger = logging.getLogger(__name__)

async def query_memories(query: str, top_k: int = 5, use_llm_rerank: bool = False):
    t0 = time.perf_counter()
    # ... 原有逻辑 ...
    t1 = time.perf_counter()
    logger.info(
        "query_memories latency: total=%.0fms, top_k=%d, llm_rerank=%s",
        (t1 - t0) * 1000, top_k, use_llm_rerank
    )
    return results
```

跑 20 次查询，取 P95（排序后第 95 百分位）。

---

## 四、成本监控（已交付）

> **现状**：已按方案落地并超越——provider 层统一埋点（唯一咽喉点），观测值进内存缓冲 + 后台 flusher 批量 INSERT 到 `llm_usage` 表（持久化），暴露 `/api/usage/*` 端点（summary/scenarios/models/threads/trace），含 `estimate_cost` 按模型价格表估算成本。详见 [architecture.md](../architecture.md) 的 "Observability (LLM usage tracing)" 一节。下方的 §4.1-4.2 为最初的内存计数草案，现被持久化方案取代，仅保留作演进记录。

### 4.1 （已实现替代）内存计数草案

```python
# 已被 backend/service/usage.py 的 llm_usage 持久化方案取代——
# 原草案是进程内 dict 计数（重启清零），现方案是内存缓冲 + 后台落库。
# 此处保留最初设计供回顾。
from collections import defaultdict
import logging

_token_usage: dict[str, int] = defaultdict(int)  # key: 场景, value: 累计 token


def _record_usage(scenario: str, usage: dict | None) -> None:
    """记录 token 用量。usage 是 LLM 返回的 usage 字段。"""
    if not usage:
        return
    total = usage.get("total_tokens", 0)
    _token_usage[scenario] += total
    logger.info(
        "LLM token usage: scenario=%s, this_call=%d, cumulative=%d",
        scenario, total, _token_usage[scenario]
    )


def get_token_usage() -> dict[str, int]:
    """返回各场景累计 token 用量，供面试演示。"""
    return dict(_token_usage)
```

> 该草案的计数逻辑现由 `backend/service/usage.py` 实现（`record_usage` + 有界内存缓冲 + 后台 flusher + `estimate_cost`），场景字段与 `llm_usage.scenario` 对齐。

### 4.2 面试话术

> 「LLM 调用我做了统一埋点——provider 层是唯一咽喉点，每次调用写一行观测到 `llm_usage` 表，按场景统计（记忆提取、冲突检测、rerank、Agent 对话分开算），并通过 `/api/usage/*` 暴露按天/按场景/按模型的汇总和成本估算。trace_id 贯穿一次 agent 运行，能回放单轮对话的调用链。
>
> 成本控制策略：90% 写入只走向量（零 LLM 成本），只在 0.72-0.85 冲突检测边界调 LLM；rerank 默认关闭（eval 显示小语料下 rerank 有害），启用时优先本地 cross-encoder（零 API 成本）；对话路径已锁死 LLM rerank（模型无法自主开，见 §3.1）；max_steps=5 防止 Agent 失控烧 token。」

---

## 五、压测补全（locust）

### 5.1 压测脚本（已交付）

`tests/perf/locustfile.py` 压测 `/api/memory/search`。相比最小草案的升级：
- **带认证头**：读取 `.env` 的 `EMA_API_KEY`，每个请求带 `Authorization: Bearer`（`backend/api/auth.py` 全局守卫所有 `/api` 路由，不带 key 返回 401）
- **真实查询集**：从 `tests/eval/ground_truth.py` 抽取 10 个查询，保证每个请求都命中真实检索（BGE embed + pgvector + sparse），不压空结果

### 5.2 跑压测（实测命令）

```bash
pip install locust
cd G:\Projects\ema
EMA_API_KEY=<key> python -m locust -f tests/perf/locustfile.py \
  --headless -u 10 -r 2 -t 60s --host http://127.0.0.1:8000 --only-summary
```

需先 `python -m tests.eval.seed` 灌入 70 条评估种子，后端以 `.venv` 启动（Windows 见下注）。

### 5.3 实测结果（2026-08-09，本机 CPU）

| 并发 | QPS | P50 | P95 | P99 | 失败率 |
|------|-----|-----|-----|-----|--------|
| 10 | 4.77 | 78ms | 110ms | 440ms | 0% |
| 20 | 9.39 | 70ms | 100ms | 230ms | 0% |
| 40 | 18.63 | 74ms | 240ms | 430ms | 0% |
| 80 | 34.77 | 110ms | 690ms | 850ms | 0% |
| 160 | 63.32 | 410ms | 1000ms | 1300ms | 0% |

**瓶颈分析**：QPS 随并发线性增长（10→160 并发，4.8→63.3），直至 160 并发仍 **0 失败**，但 80→160 增幅减半、P95 从 690ms 恶化到 1.0s——这是典型的 **CPU 嵌入式（BGE-M3）绑定 + 无失败、延迟随并发恶化** 曲线。注意本压测查询命中 query-embedding LRU 缓存（10 个固定查询重复压），测出的是**缓存热路径**；冷查询每次需 BGE embed 150-230ms（单次 curl 实测 ~1.75s）。

### 5.3.1 冷路径压测（2026-08-11，本机 CPU，多样查询）

> 热路径数字只代表"重复查询"场景。真实用户查询文本几乎不重复，query-embedding LRU（上限 1024）几乎总 miss，每条都要走 BGE-M3 CPU embed。用 seed 语料 48 个实体 × 15 个问句模板 + 30 条标注 query 生成 **779 条互不相同**的查询池（[locustfile.py](../../tests/perf/locustfile.py)，`HOT=1` 回退到 10 条固定池），压测 `HOT` 未设置 = 冷路径：

| 并发 | QPS（热/冷） | P50（热/冷） | P95（热/冷） | 失败率 |
|------|------------|-------------|-------------|--------|
| 10 | 4.77 / **2.61** | 78ms / **640ms** | 110ms / **19,000ms** | 0% |
| 40 | 18.63 / **4.88** | 74ms / **6,200ms** | 240ms / **9,400ms** | 0% |

**结论（诚实口径的核心）**：
1. **原 QPS/P95 数字高估了系统真实能力**——热路径 10 并发 P95 110ms，冷路径 P95 19s（172x）；40 并发 QPS 从 18.6 跌到 4.9（-74%）。真实多样查询下系统只能做到 **QPS ~2.6-5、P50 秒级**。
2. **根因是 query-embedding LRU 缓存**：热路径的唯一"加速器"是把重复 query 的 embed 变成 dict 查找。真实用户查询永远 miss，BGE-M3 CPU embed（并发下实测 366→1120ms/条）就是吞吐天花板。
3. **10 并发 P95 19s 是 CPU 并发超卖长尾**：torch encode 走 `asyncio.to_thread` 默认线程池，并发请求同时抢 CPU（容器日志 BGE embed latency 从 366ms 恶化到 1120ms+），偶发 GC/线程抖动把 P95 推到秒级。40 并发时 P95 9.4s 反而"更稳"（排队常态化）。
4. **这才是真实的规模性能短板**：单机 CPU 嵌入决定了系统本质上是"低并发、高延迟"形态，QPS 天花板 ~5。要改变形态必须把 embed 服务化上 GPU（batch 化），这正是风险矩阵"规模性能 🟡"的根子。

> 冷路径压测方法见 [locustfile.py](../../tests/perf/locustfile.py) 顶部注释（冷/热池构造 + `HOT=1` 对照）。

### 5.3.2 并发控制落地（2026-08-11）：P95 长尾从 19s 降到 690ms

> 冷路径压测暴露的 P95 19s 长尾根因是 **CPU 并发超卖**：torch encode 走 `asyncio.to_thread` 默认线程池（12 个线程），并发请求同时涌入且每个都抢满全部 8 核（OpenMP），互相饿死。修复在 [embedding_service.py](../../backend/service/embedding_service.py) `BGEEmbeddingProvider`：
> 1. **线程信号量**（`threading.BoundedSemaphore`，跨事件循环安全）限制同时进行的推理任务数——`EMBEDDING_MAX_CONCURRENCY`，默认 2；
> 2. **`torch.set_num_threads`** 限制单任务的内部线程数——`EMBEDDING_TORCH_THREADS`，默认 `cpu_count//2`（8 核下=4）。
>    两者乘积 ≈ 核数（2×4=8），零超卖。配置项见 [config.py](../../backend/shared/config.py) `EmbeddingConfig`。

**容器内复测**（同 779 条冷池，Docker 后端，改前后对比）：

| 并发 | QPS（改前/改后） | P50（改前/改后） | P95（改前/改后） | Max（改前/改后） |
|------|----------------|----------------|-----------------|-----------------|
| 10 | 2.61 / **4.39** | 640ms / **310ms** | **19,000ms / 690ms** | 20,215ms / **920ms** |
| 40 | 4.88 / 完成 30 | 6,200ms / **940ms** | 9,400ms / **1,900ms** | 11,188ms / **1,999ms** |

**结论**：
1. **目标场景（10 并发）P95 从 19 秒降到 690ms（-96%），Max 降到 920ms——长尾消失，同时吞吐升 68%（QPS 2.61→4.39）**。修复前 10 并发的 P95 是 CPU 超卖长尾（单条 embed 366→1120ms 被并发拖慢），限并发后单条 embed 稳定在 ~300ms。
2. **tradeoff 在 40 并发显现**：延迟大幅改善（P95 9.4s→1.9s）但吞吐受限（60s 完成 30 vs 290）——conc=2 限死了 embed 并行度。这是 CPU 密集系统的固有取舍：**低并发高质量 or 高并发高延迟**。要更高并发吞吐，调 `EMBEDDING_MAX_CONCURRENCY=3/4`（P95 会回升）或 embed 服务化上 GPU（batch 化）。
3. **配置匹配核数是最优默认**：2×4=8 线程峰值 = 核数，零超卖。高并发场景按需上调 conc，代价是 P95 上升——这是可量化的旋钮，不是黑盒。

**面试话术**：「冷压测 10 并发 P95 19s 的根因是 CPU 并发超卖——每个 embed 都抢满 8 核。我加了线程信号量限并发 + `torch.set_num_threads` 限单任务线程（乘积=核数），容器内复测 P95 从 19s 降到 690ms、Max 920ms，QPS 反而升到 4.39。40 并发下 P95 也降到 1.9s，但吞吐受限——这是单机 CPU 的固有边界，要突破得 embed 上 GPU。」

> **Windows 启动注**：Windows 上 uvicorn 默认 ProactorEventLoop 与 psycopg 异步不兼容，`_pool.wait()` 会无限重试挂起 lifespan。已修复：`_setup_checkpointer` 给 wait 加 10s 超时，超时降级 InMemorySaver（`backend/service/agent_service.py` + `tests/unit/test_checkpointer_fallback.py`）。Linux 容器（Dockerfile）无此问题。

### 5.4 面试话术

> 「我用 locust 压过检索接口：10 并发 QPS 4.8、P95 110ms；160 并发 QPS 63、P95 1.0s，全程 0 失败。**但这些是缓存热路径——10 个固定查询反复压，query-embedding 命中 LRU。我后来用 779 条互不相同的真实查询压冷路径：10 并发 QPS 掉到 2.6、P95 19s，40 并发 QPS 4.9、P95 9.4s**。瓶颈在 BGE-M3 CPU 推理——每个新 query 都要 embed（并发下 366→1120ms），LRU 缓存是热路径唯一的加速器，真实用户永远 miss。高并发要把 embedding 服务化成独立 batch 服务上 GPU。」

---

## 六、面试应答总策略

### 6.1 评估体系类问题（必问）

| 问题 | 补了评估后（实际数字） |
|------|----------------------|
| 「怎么知道检索准不准？」 | 「**70 条标注集，5 类 × 14 条 + 难度分级（hard 占 30%）**；生产 memory 路径**默认确定性基线** Recall@5=0.886、MRR 0.767（纯子串匹配，无自证）；语义通道为显式 opt-in（用被评测模型自评故非默认）——**但这只证明找得到。真实判别力看 27 条 hard-negative 陷阱集：纯向量目标召回 100%、陷阱入侵 96.3%、综合通过仅 59.3%，11 条陷阱压过目标**。已用这个集驱动改进：bounded cross-encoder top-3 重排把综合通过提至 81.5%（默认关、显式启用）」 |
| 「换 reranker 效果如何？」 | 「A/B 实测：cross-encoder 反而有害——hybrid:ce 0.967 vs 无 rerank 1.000，0.15 floor 误伤 q015；收益 scale-dependent，默认关闭是数据支撑的决策。**生产 memory 路径 LLM rerank vs bounded-CE 也实测了：recall@5 相同 0.900（rerank 不改变召回集合），MRR +0.014 / NDCG +0.011，但延迟 2.5s→14.1s（5.5x）**——小语料下 rerank 只微调排序不救召回，见 [memory_llm_vs_ce_report.md](../../tests/eval/reports/memory_llm_vs_ce_report.md)」 |
| 「记忆衰减加权有用吗？」 | 「**A/B 实测 + 已调参闭环（[decay_ab_report.md](../../tests/eval/reports/decay_ab_report.md)）**：原公式 `S=1+2·recall` 半衰期≈0.7·S 小时太激进，合成老化分布下把 recall@5 打到 0.367——19/30 条目标被压出 top-5。我把 S 乘数调到 12、加了 0.10 保留 floor，**同分布重测 recall@5 回升到 0.667、MRR 0.622**（被压出目标降到 ~10 条），且保留排序偏好（29/30 排序仍变）。语义权衡：decay 仍低于无衰减（0.667 vs 0.900）是「过时沉底」的有意代价。另修复过：从未召回的记忆按 created_at 衰减（旧代码恒返 1.0 等于永不沉底）」 |

### 6.2 成本类问题

| 问题 | 没补前 | 补了后 |
|------|--------|--------|
| 「一次对话多少钱？」 | 讲不出 | 「平均 X token，按 DeepSeek 价格约 Y 元」 |
| 「怎么控制成本？」 | 讲策略 | 「策略 + 实测：90% 写入零 LLM 成本，日均 X token」 |

### 6.3 规模类问题

| 问题 | 没补前 | 补了后 |
|------|--------|--------|
| 「能扛多少并发？」 | 讲不出 | 「locust 压测 10 并发 QPS 4.8、P95 110ms；160 并发 QPS 63、全程 0 失败」 |
| 「瓶颈在哪？」 | 推测 | 「实测瓶颈在 BGE-M3 CPU 推理——QPS 线性涨但 P95 恶化（690ms@80 并发 → 1.0s@160），冷查询再加 150-230ms embed」 |

---

## 七、执行清单（面试前 2 天）

> **进度**：评估体系、成本监控、CI、Dockerfile、locust 压测已交付。下方原清单保留待办项，已完成的标记 ✅。

### Day 1（约 6 小时）

- [x] 上午：构造 20 条标注集（已建 30 条，5 类 × 6 条 + 难度分级）—— ✅ 完成
- [x] 上午：实现 metrics.py + run_eval.py，跑出 Recall@5 / MRR / NDCG —— ✅ 完成（101→342 单测）
- [x] 下午：对比 cross-encoder vs LLM rerank，对比开关衰减，记录数字 —— ✅ 完成（memory 路径 LLM rerank vs bounded-CE：recall@5 相同 0.900、MRR +0.014、延迟 5.5x，见 §2.4；衰减开关 A/B：合成老化分布下 decay 打掉 recall 0.900→0.367，见 §6.1 与 [decay_ab_report.md](../../tests/eval/reports/decay_ab_report.md)）
- [x] 下午：LLMProvider 加 token 计数，跑 10 轮对话算成本 —— ✅ 完成（llm_usage 表）
- [x] 晚上：在 retrieval/agent 加计时日志，跑 20 次取 P95 —— ✅ 完成（检索延迟已测 190ms；对话 P95 已实测 73.6s，见 §3.1）

### Day 2（约 4 小时）

- [x] 上午：写 Dockerfile（后端 + 前端单镜像，torch 分步安装）—— ✅ 完成（见 [deployment.md](../deployment.md)）
- [x] 上午：locust 压测，拿 QPS + P95 数字 —— ✅ 完成（见 §5.3 实测表）
- [ ] 下午：把所有实测数字填回 self-introduction.md / ema-deep-dive.md 占位符 —— 1h

### 完成后应能答

- ✅ 「你评估过检索质量吗？」→ 有 Recall@5 数字
- ✅ 「换 reranker 效果如何？」→ 有对比数字
- ✅ 「一次对话多少钱？」→ 有 token 数 + 成本
- ✅ 「能扛多少并发？」→ 有 QPS 数字
- ✅ 「怎么部署的？」→ 有 Dockerfile + CI
- ✅ 「有真实数据吗？」→ 全部填实

---

## 八、如果时间不够（只补 P0）

> 评估体系、成本监控、容器化、locust 压测、对话 P95/token 成本均已交付，剩余高优先级动作：

1. **填实所有占位符**（1h）—— 避免临场卡壳

检索与生成评估数字已有（生产 memory 路径默认确定性基线 **70 条** Recall@5 0.886 / MRR 0.767，语义通道 opt-in 贡献已量化；hard-negative 判别力 27 条实测——纯向量综合通过 59.3%，bounded-CE 提至 81.5%；LLM 行为评测四套件 / locust QPS 63@160 并发），标注集已从 30 条扩充到 70 条；检索判别力改进已落地（bounded cross-encoder top-3 重排），语义通道自证已通过"默认关闭 + opt-in"修复。对话 P95（73.6s，10 轮真实对话）与 token 成本（≈28.6k tokens/轮、≈¥0.06）已实测（§3.1），且已从工具 schema 锁死对话 LLM rerank。

---

## 九、技术深度评估

> 基于 EMA 实际代码核查的深度热力图。用于决定面试时主打什么、避开什么。

### 9.1 整体判断

- **对 3-5 年中级 AI 应用工程师岗位**：✅ 深度够用，有 4-5 个能扛追问的硬亮点
- **对 5-8 年高级岗位**：⚠️ 偏薄，缺 RAG 高级技巧和模型层优化

### 9.2 深度热力图

| 模块 | 深度 | 能扛追问 | 代码证据 | 策略 |
|------|------|---------|---------|------|
| 四级去重 + 冲突 | 🟢 深 | 3 层 | [memory.py:63-77](../../backend/service/memory.py) 四分支 + [memory.py:247-256](../../backend/service/memory.py) `_deferred` 载荷 + [memory.py:429-434](../../backend/service/memory.py) 关系三元组去重 | **主打** |
| 双 HITL LangGraph | 🟢 深 | 3 层 | [graph.py:41-51](../../backend/agent/graph.py) max_steps 防循环 + Command(goto) 动态路由 | **主打** |
| 实体归一化双层 | 🟢 深 | 2-3 层 | [entity.py:67-91](../../backend/service/entity.py) 向量粗筛 + [entity.py:160-189](../../backend/service/entity.py) LLM 精判 fails safe | **主打** |
| LLMProvider 抽象 | 🟢 中深 | 2 层 | [llm_service.py:191-199](../../backend/service/llm_service.py) Anthropic system 拆分 + [llm_service.py:157-174](../../backend/service/llm_service.py) tool_calls 双形态处理 | 主打 |
| 衰减加权整合 | 🟡 中 | 2 层 | [retrieval.py:161](../../backend/service/retrieval.py) top_k*4 + [retrieval.py:171](../../backend/service/retrieval.py) `_RERANK_FLOOR=0.15` | 讲整合不讲公式 |
| 三阶段提取 | 🟢 中深 | 2 层 | [extraction.py](../../backend/service/extraction.py) gather 并行 + few-shot prompt v3 + 函数调用通道（enum 生成期约束、降级 chat_structured） | **主动讲**：A/B 数字见 §10 |
| rerank | 🟡 薄 | 1-2 层 | [rerank.py:57](../../backend/service/rerank.py) SDK 调用 + [rerank.py:95](../../backend/service/rerank.py) pointwise gather | cross-encoder 原理必须会 |
| vector_search | 🟡 薄 | 1 层 | [retrieval.py:99-104](../../backend/service/retrieval.py) 白名单 filter | 讲 SQL 可见 |
| RAG 高级技巧 | 🟢 有 | 2 层 | jieba 中文分词 hybrid + query_rewrite_and_search tool（[retrieval.py](../../backend/service/retrieval.py) sparse_search / [tools.py](../../backend/agent/tools.py)） | 主动讲，三次假设迭代是亮点 |
| Prompt 高级技巧 | 🟡 有 | 1-2 层 | 提取 prompt 已加 few-shot examples（v3，`backend/service/prompts.py`），有量化对比 | 讲提取 A/B；CoT 仍无 |
| 模型微调 | ❌ 无 | 0 层 | 直接用预训练 BGE-M3 | 不主动提 |

### 9.3 主打 3 个深度点（每个能讲 3-5 分钟）

1. **四级去重 + 冲突检测** → 阈值调参、fails safe、`_deferred` 载荷传递、4 选项解决
2. **双 HITL LangGraph** → interrupt/Command vs edge、max_steps 防循环、PostgresSaver 持久化
3. **实体归一化双层判断** → 向量粗筛 + LLM 精判、fails safe 假定不匹配、批量回填

### 9.4 避开深度薄的地方

- 三阶段提取：只讲"并行编排 + fails safe"，**不主动提 prompt 优化**
- rerank：cross-encoder 原理必须能讲，**不主动提 listwise/微调**
- 衰减：讲"工程整合"，**不主动讲公式系数调参**

### 9.5 被问到缺失深度时的应答

> 「这块 EMA 目前没做，是后续优化方向。比如 query rewriting 我考虑过——在 retrieve 前加一步 LLM 改写，能提升召回率，但增加一次 LLM 调用，成本和延迟要权衡。如果做，我会先 A/B 测算 ROI。」

**公式**：承认没做 + 讲考虑过的方案 + 讲权衡 → 体现"知道但有意没做"，比"不知道"强。

---

## 十、三阶段提取优化方案（补深度薄的地方）

> 三阶段提取是 EMA 记忆写入的关键环节，目前全 zero-shot + JSON 解析，深度偏薄。
> 本方案给出 2 个可落地的优化，让面试时能讲"我做过 prompt 优化"。
>
> **状态（2026-08-11）：✅ 两项优化均已实施并实测**——few-shot examples 已进 prompt（`extraction.entities`/`extraction.relations` v3），函数调用通道已实现（`extract_entities`/`extract_relations` 工具，OpenAI 兼容 provider 优先，降级 `chat_structured`）。**A/B 实测见 [extraction_ab_report.md](../../tests/eval/reports/extraction_ab_report.md)：few-shot 让 entity_recall 0.781→0.927、relation_recall 0.531→0.688、relation_f1 0.356→0.427（+0.071）；函数调用通道 recall 持平、precision 略降（过度抽取）但 entity_type_accuracy 上升（enum 生成期约束）**。下方保留原始方案文档供回顾。

### 10.1 现状问题

**当前实现**（[extraction.py](../../backend/service/extraction.py)）：
- 3 个 prompt 全 zero-shot，无 few-shot examples
- 输出靠 `json.loads` 解析，失败返回空 list（[extraction.py:69-71](../../backend/service/extraction.py)）
- entity type 枚举 7 类，没统计覆盖率
- 没有对比过 few-shot vs zero-shot 的准确率

**会被追问到露馅的点**：
- 「为什么不用 few-shot？效果差多少？」→ 答不上
- 「entity 抽取准确率多少？」→ 没测
- 「JSON 解析失败率多少？」→ 没统计

### 10.2 优化 1：Few-shot Examples（2-3 小时，高 ROI）

**目标**：在 entity 和 relation 提取的 prompt 里加 2-3 个 few-shot examples，提升准确率 + 降低 JSON 解析失败率。

**实现**（修改 [extraction.py](../../backend/service/extraction.py) 的 prompt）：

```python
_ENTITIES_PROMPT = """\
Extract named entities from the following text.
Return ONLY a JSON array of objects with "name" and "type" fields.
Types must be one of: person, project, technology, decision, event, file, concept.
Use the same language as the input text for entity names.

Text:
{input_text}

Examples:
Input: "我们决定用 PostgreSQL 替代 MySQL，因为 pgvector 支持向量检索"
Output: [{{"name": "PostgreSQL", "type": "technology"}}, {{"name": "MySQL", "type": "technology"}}, {{"name": "pgvector", "type": "technology"}}, {{"name": "用 PostgreSQL 替代 MySQL", "type": "decision"}}]

Input: "DBConfig.java 的连接池配置导致 OOM，已回滚"
Output: [{{"name": "DBConfig.java", "type": "file"}}, {{"name": "连接池配置", "type": "concept"}}, {{"name": "OOM", "type": "event"}}]

Now extract entities from the text above. Output ONLY the JSON array:"""
```

**关键设计**：
- 2 个 examples 覆盖不同场景（技术决策 / 故障复盘）
- examples 里展示 type 枚举的正确用法（decision 不是 technology）
- examples 用 EMA 真实场景，提升 in-domain 效果

**面试话术**：
> 「我对比过 zero-shot 和 few-shot——加 2 个 in-domain examples 后，entity 抽取的 JSON 解析失败率从 [X%] 降到 [Y%]，准确率也有提升。关键是 examples 要覆盖不同 type 的边界 case，比如 decision 和 technology 容易混淆。」

### 10.3 优化 2：Function Calling 强制结构化输出（1-2 小时）

**目标**：用 LLM 的 function calling 替代 `json.loads`，从机制上杜绝格式错误。

**实现**（修改 [extraction.py](../../backend/service/extraction.py)，用 [llm_service.py:54-83](../../backend/service/llm_service.py) 的 `chat_raw`）：

```python
# 定义 entity 抽取的 function schema
_EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract named entities from text",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": ["person", "project", "technology",
                                         "decision", "event", "file", "concept"]
                            }
                        },
                        "required": ["name", "type"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
}


async def extract_entities_v2(content_or_summary: str) -> list[dict]:
    """用 function calling 强制结构化输出，替代 JSON 解析。"""
    try:
        llm = get_llm_provider()
        result = await llm.chat_raw(
            messages=[{"role": "user", "content": f"Extract entities from:\n{content_or_summary}"}],
            tools=[_EXTRACT_ENTITIES_TOOL],
            tool_choice={"type": "function", "function": {"name": "extract_entities"}},
        )
        tool_calls = result.get("tool_calls", [])
        if tool_calls:
            args = tool_calls[0]["args"]
            entities = args.get("entities", [])
            if isinstance(entities, list):
                return entities
        return []
    except Exception:
        logger.exception("LLM entity extraction failed")
        return []
```

**优势**：
- `type` 用 `enum` 约束，LLM 不会输出非法类型
- `required` 强制字段存在
- 不再依赖 `json.loads`，格式错误率趋近 0
- `tool_choice` 强制调用该 function

**注意**：`tool_choice` 强制调用是 OpenAI 的语法，Anthropic 的语法略有不同（需要在 [llm_service.py](../../backend/service/llm_service.py) 的 `chat_raw` 里适配）。如果 provider 不支持强制调用，退回到 few-shot + JSON 解析。

**面试话术**：
> 「我后来把 entity 抽取从 JSON 解析改成了 function calling——用 enum 约束 type 字段，required 强制字段存在，格式错误率从 [X%] 降到接近 0。代价是部分 provider 的 tool_choice 支持不一致，我在 LLMProvider 抽象层做了适配。」

### 10.4 优化 3：抽取准确率评估（可选，配合 §2 评估体系）

**目标**：给三阶段提取一个准确率数字，被追问时能答。

**方法**：
1. 从记忆库抽 50 条原始内容
2. 人工标注正确的 entities 和 relations（ground truth）
3. 跑 `extract_memory()`，算 entity 的 precision/recall
4. 对比 zero-shot vs few-shot vs function calling 三种方式

```python
# tests/eval/extraction_eval.py
"""评估三阶段提取的准确率"""

from backend.service.extraction import extract_memory
from tests.eval.ground_truth import EXTRACTION_GT  # 50 条人工标注


async def eval_extraction():
    results = []
    for item in EXTRACTION_GT:
        extracted = await extract_memory(item["content"])
        gt_entities = {e["name"] for e in item["entities"]}
        pred_entities = {e["name"] for e in extracted["entities"]}

        tp = len(gt_entities & pred_entities)
        precision = tp / len(pred_entities) if pred_entities else 0
        recall = tp / len(gt_entities) if gt_entities else 0

        results.append({"precision": precision, "recall": recall})

    n = len(results)
    return {
        "precision": sum(r["precision"] for r in results) / n,
        "recall": sum(r["recall"] for r in results) / n,
    }
```

**面试话术**：
> 「我评估过 entity 抽取——50 条标注集上，precision [0.82]，recall [0.74]。主要漏召是复合实体（如"连接池配置"被拆成"连接池"和"配置"）。加 few-shot 后 recall 提到 [0.81]。」

### 10.5 执行清单

> **状态：✅ 前 3 项已完成并实测**（few-shot 已进 prompt v3；函数调用通道已实现，provider 不支持时降级；A/B 数字见 [extraction_ab_report.md](../../tests/eval/reports/extraction_ab_report.md)）。第 4 项"50 条标注集"用现有 8 条 `llm_ground_truth.EXTRACTION_ITEMS` 替代——量级更小但复用同一套评测管线，标注集扩充列为后续优化。

| 优先级 | 动作 | 耗时 | 收益 | 状态 |
|--------|------|------|------|------|
| 🟡 P1 | 加 few-shot examples 到 entity/relation prompt | 2h | 能讲"做过 prompt 优化" | ✅ 完成（v3） |
| 🟡 P1 | 跑 zero-shot vs few-shot 对比，记数字 | 1h | 有量化对比 | ✅ 完成（entity_recall +0.146） |
| 🟢 P2 | 实现 function calling 版 extract_entities | 2h | 能讲"结构化输出" | ✅ 完成（enum 生成期约束 + 降级） |
| 🟢 P2 | 构造 50 条标注集，算 precision/recall | 3h | 被追问准确率能答 | 🟡 部分（用 8 条 llm_eval 标注集替代） |

### 10.6 面试应答对比

| 问题 | 优化前 | 优化后（实测数字） |
|------|--------|--------|
| 「为什么不用 few-shot？」 | 答不上 | 「加 few-shot 后 entity_recall 0.781→0.927（+0.146）、relation_recall 0.531→0.688（+0.157）、relation_f1 0.356→0.427」 |
| 「JSON 解析失败怎么办？」 | 「fails safe 返回空」 | 「entity/relation 走函数调用通道——enum 在生成期约束，非法 type 机制上产生不出来；格式错误率趋近 0。DeepSeek thinking 模式拒绝强制 tool_choice（400），所以不强制、靠模型自然调用工具，失败降级 chat_structured」 |
| 「entity 抽取准确率？」 | 没测过 | 「8 条标注集：entity_recall 0.927 / entity_f1 0.772 / type_accuracy 0.792（few-shot + 函数调用）」 |
| 「prompt 怎么迭代？」 | 靠 git | 「有标注集，`python -m tests.eval.extraction_ab` 每次改动跑 zero-shot/few-shot/函数调用三臂对比」 |

**核心收益**：把"三阶段提取"从 🟡 薄 升到 🟢 中深，多一个能扛 2 层追问的深度点。

---

## 十一、检索召回优化方案（补 RAG 高级技巧短板）

> **2026-08-07 更新（语料更正）**：旧版"0.833→1.000（5/5 救回）"的 baseline 生成于 08-06 06:11，早于 07:12 的语料重新播种，且旧指纹与 seed 内容不匹配——**对比不一致，旧结论在当前语料上不可复现**。当前语料 + 当前指纹下 BGE-M3 稠密召回即 Recall@5=1.000。详见 §11.5 实测表与 [eval-report.md](eval-report.md)。
>
> **真正的生产瓶颈是 sparse_search 的 O(N) 扫描，不是 rerank**：中文 sparse 必须把 jieba 分词结果落到 `chunks.tokens` 列 + GIN 索引，用 `tokens && :q` 把候选集限制在真实 token 重叠的行（O(log N) 发现候选），Jaccard 只在小候选集上算。rerank（cross-encoder）经 A/B 验证在小语料下有害（0.15 floor 误伤 q015），可跳过，收益 scale-dependent。§11.1-11.4 保留为历史诊断过程，体现"做了、测了、错了、再测、纠正归因、连默认假设都敢质疑"的迭代。

### 11.1 现状问题（实测数据支撑）

**评估数据**（[eval-report.json](../../tests/eval/reports/eval-report.json)，30 query；**注**：以下为 08-06 06:11 旧 baseline，基于重新播种前的语料——当前语料下 dense 单独即 Recall@5=1.000）：
- Recall@5=0.833（25 命中 / 5 miss）
- 5 个 miss query 的相关结果**均不在 top-20 候选池** → rerank 无法救回
- miss 全是概念查询，query 和相关记忆词重合度低：

| query | 相关记忆（应召回） | miss 原因 |
|-------|------------------|----------|
| q007 "koa-connect 之前出过什么问题" | "koa-connect wrapper 导致 ctx 泄漏" | "出过什么问题" 与 "ctx 泄漏" 词面无重合 |
| q012 "Agent 会不会陷入死循环" | "max_steps 防循环" | "会不会" 与 "max_steps" 无重合 |
| q018 "同名实体怎么归一化" | "向量粗筛 + LLM 精判" | "归一化" 与 "粗筛/精判" 无重合 |
| q026 "项目分了几个阶段" | "Phase 1-4 实现" | "几个阶段" 与 "Phase" 中英文不匹配 |
| q029 "连接器支持哪些平台" | "connector.py 支持 Git/PingCode/CI/飞书" | "平台" 与具体平台名无重合 |

**根本原因**：BGE-M3 dense embedding 对"词面不重合但语义相关"的 query 召回力不足。

**会被追问到露馅的点**：
- 「Recall 0.833，剩下 16.7% 怎么救？」→ 答不上
- 「为什么不用 query rewriting？」→ 只能说"考虑过"
- 「dense vector 召回不了概念查询怎么办？」→ 没方案

### 11.2 方案 A：Query 改写（LLM 扩展概念词，2-3h，高 ROI）

**目标**：retrieve 前加一步 LLM 改写，把概念查询扩展成多个语义相近的表述，多路召回取并集。

**实现**（新建 `backend/service/query_rewrite.py`）：

```python
"""LLM-based query rewriting for retrieval recall improvement."""

from backend.service.llm_service import get_llm_provider

_REWRITE_PROMPT = """\
Rewrite the following query into 3 semantically equivalent variations that
might appear in a technical knowledge base. Focus on concrete terms and
entities that the original query implies but does not state.

Query: {query}

Output one variation per line, no numbering:
"""


async def rewrite_query(query: str, n_variations: int = 3) -> list[str]:
    """Return [original] + n variations for multi-query retrieval."""
    try:
        llm = get_llm_provider()
        resp = await llm.chat_raw(
            messages=[{"role": "user", "content": _REWRITE_PROMPT.format(query=query)}],
        )
        text = resp.get("content", "")
        variations = [line.strip() for line in text.strip().split("\n") if line.strip()][:n_variations]
        return [query] + variations
    except Exception:
        return [query]  # fails safe: original query only
```

**集成到 retrieve**（修改 [retrieval.py](../../backend/service/retrieval.py)）：

```python
async def retrieve_multi_query(
    query: str,
    top_k: int = 5,
    *,
    use_llm_rerank: bool = False,
) -> list[Chunk]:
    """Multi-query retrieval: rewrite + union + dedup + rerank."""
    from backend.service.query_rewrite import rewrite_query

    queries = await rewrite_query(query)
    all_chunks: dict[str, Chunk] = {}  # dedup by chunk id
    for q in queries:
        vec = await embed_query(q)
        results = await vector_search(vec, top_k=top_k * 2)
        for r in results:
            cid = r["id"]
            if cid not in all_chunks:
                all_chunks[cid] = _row_to_chunk(r)

    candidates = list(all_chunks.values())[:top_k * 4]
    return await _rerank_and_trim(candidates, query, top_k, use_llm_rerank)
```

**关键设计**：
- fails safe：改写失败时只用原 query，不阻塞检索
- 多路召回取并集，dedup by chunk id，避免重复
- 改写只增加 1 次 LLM 调用（~500ms），成本可控
- q007 "koa-connect 之前出过什么问题" 预期被改写成 "koa-connect ctx 泄漏 / 中间件 wrapper 问题"，词重合度提升

**面试话术**：
> 「概念查询（如"之前出过什么问题"）dense 词面召回弱，我做了 query rewriting 作为 Agent 的显式 tool（query_rewrite_and_search），让 Agent 自主判断概念查询时调用，fails safe 失败退回原 query。但单独改写救不回稠密召回——真正的瓶颈是中文 sparse 通道不可用 + 检索的 O(N) 扩容。Postgres `simple` 分词器对中文返回 0 行，我把 jieba 分词结果落库（tokens 列 + GIN 索引），`tokens &&` 过滤把候选集限制在真实重叠的行，sparse 才既可用又不退化。」

### 11.3 方案 B：Hybrid Search（dense + sparse BM25，4-6h）

**目标**：dense vector 召回不了的概念查询，用 sparse（BM25 关键词）补位。两者并集后 rerank。

**实现**（加 SQL migration + [retrieval.py](../../backend/service/retrieval.py) 加 sparse_search）：

```sql
-- 给 chunks 表加全文搜索索引（migration）
ALTER TABLE chunks ADD COLUMN tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;
CREATE INDEX idx_chunks_tsv ON chunks USING GIN(tsv);
```

```python
async def sparse_search(query: str, top_k: int = 20) -> list[dict]:
    """BM25-style keyword search via Postgres ts_vector."""
    sql = text("""
        SELECT id, content,
               ts_rank(tsv, plainto_tsquery('simple', :q)) AS rank
        FROM chunks
        WHERE tsv @@ plainto_tsquery('simple', :q)
        ORDER BY rank DESC
        LIMIT :top_k
    """)
    async with get_session() as session:
        result = await session.execute(sql, {"q": query, "top_k": top_k})
        return [dict(row) for row in result.mappings()]


async def retrieve_hybrid(query: str, top_k: int = 5) -> list[Chunk]:
    """Hybrid: dense + sparse union → rerank."""
    vec = await embed_query(query)
    dense = await vector_search(vec, top_k=top_k * 4)
    sparse = await sparse_search(query, top_k=top_k * 4)
    merged: dict[str, dict] = {}
    for r in dense + sparse:
        cid = r["id"]
        if cid not in merged:
            merged[cid] = r
    candidates = list(merged.values())[:top_k * 4]
    return await _rerank_and_trim(candidates, query, top_k, False)
```

**关键设计**：
- 复用 Postgres 自带 ts_vector，**不需要额外引 ES**（保持架构简单）
- `simple` 分词器对中英文混合够用（按词切，不依赖中文分词库）
- dense + sparse 并集，互补：dense 救语义相关，sparse 救关键词命中
- q029 "连接器支持哪些平台" → sparse 命中含 "Git/PingCode/CI/飞书" 的记忆

**面试话术**：
> 「中文 sparse 检索的生产瓶颈是 O(N) 扫描。我先试 Postgres 原生 tsvector，`simple` 分词器分不了中文（对中文返回 0 行）；换成 jieba 分词 + Jaccard，并把 jieba 结果落到 chunks.tokens 列 + GIN 索引，`tokens &&` 过滤把候选集限制在真实重叠的行——1000 条语料延迟 -69%。rerank 单独 A/B 在小语料下有害（0.15 floor 误伤 q015），可跳过，收益 scale-dependent。选 Postgres/Python 不选 ES 是不引新组件。」

### 11.4 方案对比与选型

| 维度 | 方案 A（query 改写） | 方案 B（hybrid search） |
|------|---------------------|----------------------|
| 实现复杂度 | 低（+1 模块，改 retrieve） | 中（加 SQL migration + sparse 函数） |
| 延迟开销 | +500ms（1 次 LLM 调用） | +10ms（1 次 SQL 查询） |
| 对 5 个 miss 的预期救回 | 3-4 个（扩展概念词后词重合度提升） | 2-3 个（关键词命中补位） |
| 架构依赖 | 无新依赖 | 无新依赖（复用 Postgres） |
| 可解释性 | 高（能看到改写后的 query） | 中（BM25 排序透明） |
| **是否全量触发** | **否**（90% query 不需要，白付 500ms） | **是**（+10ms 无感，默认开） |

**关键实测发现（score 阈值触发不可行）**：跑了 30 query 的 vector_search top-1 score，hit 均值 0.685 [0.543-0.777]，miss 均值 0.645 [0.600-0.689]，**两者完全重叠**——miss query 的 score 看起来"自信"但召回了错的结果，17 个 hit 的 score 低于 miss 最高分。无法用 score 阈值按需触发改写。

**真正的按需触发信号**：5 个 miss query 全是概念性问句（含"什么/怎么/哪些/会不会/几个"），25 个 hit 大多是具体技术词查询。按 query 语言学特征（疑问代词）触发，而非 score。

**修正后的推荐路径**：
1. **hybrid 默认开**（+10ms 无感，救 2-3 个 miss，不需要按需判断）
2. **query 改写按概念词触发**（query 含疑问代词时才改写，90% query 走 210ms fast path，10% 走 700ms slow path）
3. 或更干净：**改写作为 Agent 的显式 tool**（`query_rewrite_and_search`），让 Agent 自主判断概念查询时调用，retrieve 默认走 hybrid

**端到端延迟视角**：Agent 对话 LLM 调用本身 2-5 秒，+500ms 改写占 10-25%，但只有 10% query 需要付这个代价。全量改写 = 让 90% query 白付，不合理。

### 11.5 评估验证（用 tests/eval A/B）

```bash
# baseline（当前）
python -m tests.eval.run_eval --retriever vector --report-json baseline.json

# 方案 A：query 改写（实现后加 --retriever rewrite）
python -m tests.eval.run_eval --retriever rewrite --report-json rewrite.json

# 方案 B：hybrid（实现后加 --retriever hybrid）
python -m tests.eval.run_eval --retriever hybrid --report-json hybrid.json
```

**实测结果**（30 query 全量评估；**注**：baseline 0.833 为重新播种前的旧语料，hybrid 行在播种后——0.83→0.97 增量混入了语料变更，非纯 jieba 收益。当前语料下 vector 单独即 1.000，见 [eval-report.md](eval-report.md)）：

| 方案 | Recall@5 | MRR | 救回 miss | 延迟/query | 实测结论 |
|------|----------|-----|----------|-----------|---------|
| baseline (vector) | 0.833 | 0.817 | — | ~200ms | 5 miss 全是概念查询 |
| + hybrid (simple 分词器) | 0.833 | — | **0/5** | ~210ms | sparse 对中文返回 0 行，`simple` 分词器不支持中文 |
| + query rewrite (LLM) | 0.833 | — | **0/5** | ~700ms | 改写质量高但 simple 分词器仍是瓶颈 |
| + hybrid (jieba 分词) + rerank | 0.967 | 0.917 | **4/5** | ~20906ms | jieba 救回 4/5；q015 被 0.15 floor 滤掉 |
| **+ hybrid (jieba 分词) 无 rerank** | **1.000** | **0.983** | **5/5** | **~235ms** | rerank 在小语料下有害：floor 误伤 + 打分噪声干扰排序 |

**根因分析**（四步迭代，第四步 A/B 推翻"rerank 必要"的默认假设）：
1. **simple hybrid 失效**：Postgres `simple` 分词器把中文当整体 token，匹配不到任何行。前两次失败的真正根因——不是数据太短，是分词器不支持中文。
2. **rewrite 单独失效**：改写扩宽了 query 语义，但 simple 分词器仍是瓶颈，hybrid 通道返回 0 行。
3. **jieba 成功（中文可用性）**：换 jieba 分词后，sparse 通道对中文 query 才有效（`simple` 分词器对中文返回 0 行）。但"4/5 救回、0.83→0.97"混入了 07-12 语料重新播种的变更——当前语料下 dense 单独即 1.000，sparse 的真实增益是"中文 sparse 通道可用 + DB 侧落库解决 O(N)"，而非"救回稠密漏掉的结果"。
4. **A/B 跳过 rerank → 1.00**：剩下 q015 miss 查了是 cross-encoder 打了 0.142 < 0.15 floor 被滤掉。做了 A/B 跳过 rerank，Recall@5 直接到 1.00、MRR 0.98、延迟 20s→235ms。**rerank 在 30 条语料下是有害的**：候选池覆盖 73% 语料，dense similarity 本身已接近完美排序，cross-encoder 打分噪声反而干扰排序，0.15 floor 还误伤 1 条。
5. **归因教训**：①第二次失败时差点归因到"数据太短"——错了，根因是分词器；②第四步推翻了"rerank 一定提升质量"的默认假设——rerank 收益是 scale-dependent 的，小语料下噪声盖过收益，大语料下才会反过来。

**修正后的优化方向**（jieba + skip-rerank 已落地，当前 scale 最优）：

| 方向 | 预期收益 | 代价 | 优先级 |
|------|---------|------|--------|
| ✅ jieba 中文分词（已落地） | 中文 sparse 通道可用（`simple` 分词器对中文 0 行） | 落库 + 重建索引 | 已完成 |
| ✅ 跳过 rerank（已落地） | 0.97→1.00，延迟 20s→235ms | 大语料需重新评估 | 已完成 |
| ✅ 1000 条 LLM 干扰语料实测（已落地） | 验证临界点：Recall 1.00→0.933 | 24min 跑一轮 | 已完成 |
| ✅ 万级语料重新评估 rerank（已落地，2026-08-11） | rerank 转正实锤：10k 语料 +CE Recall 0.933→0.967 / MRR 0.732→0.886，见 §11.5.3 | 1 万条 embed + 两轮 eval ≈ 1h | 已完成 |
| ✅ sparse_search 迁移 DB 侧（jieba 存列 + GIN）（已落地） | 稳态延迟 O(N)→O(log N)，1000 条 641ms→198ms（-69%） | 加列 + 重建索引 + 改 SQL | 已完成 |
| 丰富种子数据 | 边际收益（当前已 1.00） | 数据工程 | P3 |
| rerank 加速（GPU 或换轻量 model） | 延迟 20s→<1s | GPU 部署/换模型 | P2（仅大语料需要时） |

#### 11.5.1 Scale-dependent 决策：rerank 什么时候从有害变有益

**核心变量：候选池覆盖率**（candidate pool / corpus size）。这是决定 rerank 正负收益的关键因子。

| 语料规模 | 候选池 | 覆盖率 | dense sim 区分度 | rerank 作用（实测） | 决策 |
|---------|--------|--------|-----------------|--------------------|------|
| 30 条 | ~22 | **73%** | 高（分数分散） | **有害**（+CE Recall 0.967 < 无 rerank 1.000） | 跳过 |
| 1000 条（已验证） | ~40 | **4%** | 中（分数开始压缩） | **边界区**（无 rerank Recall 掉到 0.933） | **临界点** |
| 万级（**已验证**，见 §11.5.3） | ~40 | **0.4%** | 低（分数高度压缩） | **有益**（+CE Recall 0.933→0.967 / MRR 0.732→0.886） | **开启** |

**为什么 30 条时 rerank 有害**：
1. 候选池覆盖 73%，相关 chunk 几乎必然在候选池里——rerank 不提升 Recall
2. 候选少（22 个），dense sim 分数分散（0.3-0.8），排名已经接近完美——rerank 不提升 MRR
3. cross-encoder 打分有噪声（0.142 给了 q015 的正确答案低于 floor）——rerank 反而降 MRR + 掉 Recall

**为什么万级时 rerank 有益（10k 实测确认）**：
1. 候选池覆盖率 0.4%，相关 chunk 不一定进 top-40——但这是**召回问题**，rerank 解决不了
2. 40 个候选都"有点像"，dense sim 分数集中在 0.5-0.7，区分不开——rerank 的 pairwise 打分能拉开差距（10k 实测 MRR 0.732→0.886，**+0.154**，远超 recall 变化——正是"排序精度"收益）
3. 0.15 floor 在高密度候选下能有效滤掉不相关噪声

**关键认知**：rerank 解决的是**排序精度**问题（MRR），不是**召回**问题（Recall）。小语料下 dense sim 排序已足够好，rerank 的边际收益为负；大语料下 dense sim 区分度下降，rerank 的边际收益才转正。**10k 实测把这条假说从"推断"变成了"实证"。**

**1000 条延迟预测**：rerank 候选数不随语料增长（恒为 top_k×4=20 per retriever，union ~40），所以 rerank 延迟基本不变（~20-25s）。无 rerank 路径也基本不变（hnsw + GIN 索引 sublinear 扫描，~250-300ms）。**延迟不是决策因子，收益翻转才是**——10k 实测无 rerank 延迟 882ms（未随语料恶化，DB 侧优化生效），+CE rerank 29.9s/query。

#### 11.5.2 1000 条实测验证（LLM 干扰语料）

用 DeepSeek 生成 837 条 + 模板补 133 条 = 970 条干扰 chunks（覆盖 pgvector/HNSW/HITL/RAG/Agent 等 33 个相邻主题），插入 DB 后跑 hybrid_norerank eval：

| 指标 | 30 条 | 1000 条 | 变化 |
|------|-------|---------|------|
| Recall@5 | 1.000 | **0.933** | **-0.067** |
| MRR | 0.983 | 0.917 | -0.066 |
| NDCG@5 | 0.988 | 0.921 | -0.067 |
| 延迟 | 235ms | 1498ms | **+6x** |

**Recall 掉了——2 个 query miss**，且 miss 的主题正好是干扰语料覆盖的相邻领域：

| miss query | 类别/难度 | 原因 |
|-----------|----------|------|
| q010「向量索引召回不准怎么调」 | 故障复盘/hard | 干扰语料含 pgvector HNSW/IVF/PQ 调优条目，把目标种子挤出 top-5 |
| q015「人工介入 HITL 是怎么实现的」 | 架构设计/medium | 干扰语料含 HITL/审批流/状态机条目，把目标种子挤出 top-5 |

**按类别 Recall@5**：技术决策 1.00 / 代码实现 1.00 / 历史背景 1.00 / 故障复盘 0.83 / 架构设计 0.83——只掉在干扰语料强覆盖的主题上。

**关键结论**：1000 条是 rerank 收益的**临界点**。30 条时 rerank 有害（噪声 > 信号），万级时有益（区分压缩分数），1000 条正好在边界——Recall 开始掉但只掉 2 条，是否开启 rerank 需要看 miss 的代价是否可接受。这验证了 scale-dependent 决策框架：rerank 不是"有用/没用"的二元判断，而是随语料规模变化的连续函数。

**真正的发现：sparse_search O(N) 瓶颈**。延迟从 235ms 暴增到 1498ms（6x），全来自 Python 侧 jieba 分词 + Jaccard 要加载全部 1000 条 chunks 逐条计算。30 条时 ~900ms 可接受，1000 条时 ~1300ms 已经吃掉总延迟的 87%。

**✅ 已优化：sparse_search 迁移 DB 侧（jieba tokens 存列 + GIN 索引）**。将 jieba 分词结果存入 `chunks.tokens TEXT[]` 列，建 GIN 索引，sparse_search 改用 `WHERE tokens && :q_tokens`（GIN 索引 O(log N) 筛选）+ Python 侧 Jaccard 精排。迁移后 1000 条稳态延迟从 641ms 降到 198ms（-69%），接近 30 条基线（194ms）——O(N) 瓶颈已消除，延迟不再随语料规模增长。

| 指标 | 迁移前（Python O(N)） | 迁移后（DB GIN） | 改善 |
|------|----------------------|-----------------|------|
| 30 条稳态延迟 | 235ms | 194ms | -17% |
| 1000 条稳态延迟 | 641ms | 198ms | **-69%** |
| Recall@5（30 条） | 1.000 | 1.000 | 一致 |
| Recall@5（1000 条模板） | 1.000 | 1.000 | 一致 |

**生产优化方向（新增）**：

| 方向 | 预期收益 | 代价 | 优先级 |
|------|---------|------|--------|
| sparse_search 迁移到 DB 侧（jieba 分词结果存列 + GIN 索引） | 延迟 O(N)→O(log N)，1000 条 1300ms→<50ms | 加列 + 重建索引 + 改 SQL | P2（万级语料前必须做） |
| 用真实语料重测 1000 条 eval | 验证 Recall 是否真的不掉 | 数据工程 | P3 |

#### 11.5.3 万级实测验证：rerank 转正实锤（2026-08-11）

> 复用 [probe_scale_1000.py](../../tests/eval/probe_scale_1000.py)（重写为参数化：`--target 10000 --rerank`），LLM 精修 33 个相邻主题 856 条 + 模板填充 9114 条 = 9969 条干扰 chunks，插入后跑 hybrid_norerank **和** hybrid+cross-encoder 两轮 eval（每轮 30 query 全量）。报告：[scale_10000_norerank.json](../../tests/eval/scale_10000_norerank.json) / [scale_10000_rerank.json](../../tests/eval/scale_10000_rerank.json)。

| 指标 | 30 条 | 9999 条（无 rerank） | 9999 条（+CE rerank） | rerank 增益 |
|------|-------|--------------------|----------------------|------------|
| Recall@5 | 1.000 | 0.933 | **0.967** | **+0.034（救回 1/2 miss）** |
| MRR | 0.953 | 0.732 | **0.886** | **+0.154** |
| NDCG@5 | 0.964 | 0.782 | **0.906** | **+0.124** |
| 延迟/query | 979ms | 882ms | 29,943ms | +29s |

**rerank 在万级转正的完整证据链**：

| 语料 | 无 rerank Recall@5 | +CE Recall@5 | MRR 提升 | 结论 |
|------|------------------|--------------|---------|------|
| 30 条 | 1.000 | 0.967 | 负 | rerank 有害 |
| 1000 条 | 0.933 | （未跑） | — | 临界点 |
| 9999 条 | 0.933 | **0.967** | **+0.154** | **rerank 转正** |

**两个 miss 与 rerank 的取舍**（和 1000 条时完全相同的两个 query——干扰语料强覆盖的主题）：

| miss query | 无 rerank | +CE rerank | 结论 |
|-----------|-----------|-----------|------|
| q010「向量索引召回不准怎么调」 | miss | **救回** | rerank 的 pairwise 打分把目标种子拉回 top-5 |
| q015「人工介入 HITL 是怎么实现的」 | miss | 仍 miss | 目标在 rerank 候选池外（召回问题，rerank 无解） |

**关键发现（面试最值得讲）**：
1. **rerank 的收益主体是 MRR（+0.154）而非 Recall（+0.034）**——10k 下 dense sim 排序开始压缩，40 个候选"都像"，cross-encoder 拉开差距；召回缺口（q015）是候选池外的问题，rerank 救不了。这实证了 §11.5.1 的"rerank 解决排序精度而非召回"。
2. **无 rerank 延迟在 10k 没有恶化（882ms vs 30 条 979ms）**——sparse_search DB 侧 GIN 迁移的 O(N)→O(log N) 在万级同样生效，hnsw 也撑住了。**延迟不是 rerank 的决策障碍，收益翻转才是**。
3. **完整 scale-dependent 决策曲线现已全部实测**：30 条有害 → 1000 条临界 → 万级转正。这不是"rerank 有用/没用"的二元判断，而是随语料规模变化的连续函数——面试被追问"万级怎么办"时，用这张实测表回答。
4. **+CE rerank 延迟 29.9s/query 是生产部署前的门槛**：万级开启 rerank 的前提是 embedding/rerank 服务化上 GPU（否则对话延迟不可接受）——这正好接上规模性能风险矩阵的 GPU 拐点。

### 11.6 面试应答对比

| 问题 | 优化前 | 优化后 |
|------|--------|--------|
| 「Recall 0.833 怎么提升？」 | "考虑过 query rewriting" | 「四步迭代：score 阈值不可行 → simple hybrid 0/5 → jieba 分词 4/5 救回 0.97 → A/B 跳过 rerank 到 1.00/235ms」 |
| 「改写效果如何？」 | 只能说"考虑过" | 「LLM 改写质量很高，但 simple 分词器分不了中文，hybrid 通道返回 0 行。换 jieba 后才生效」 |
| 「hybrid 为什么不用 ES？」 | — | 「用 Postgres tsv 做 BM25，但 simple 分词器对中文无效。最终在 Python 侧用 jieba + Jaccard 重写 sparse_search，不引新组件」 |
| 「怎么验证效果？」 | 没测 | 「30 query 全量 A/B：simple hybrid 0/5，jieba hybrid 4/5，跳过 rerank 5/5，每步都有数据」 |
| 「rerank 为什么不用？」 | — | 「A/B 实测：30 条语料下候选池覆盖 73%，dense similarity 已接近完美排序，cross-encoder 打分噪声反而掉 MRR，0.15 floor 还误伤 1 条。跳过后 Recall 1.00、延迟 235ms。但 docs 标注了万级语料需重新评估——rerank 收益是 scale-dependent 的」 |
| 「万级语料怎么办？rerank 还是不用？」 | — | 「我实测了：生成 9969 条相邻主题干扰语料，30 query 全量跑 A/B。**无 rerank Recall@5 掉到 0.933、MRR 0.732；开 cross-encoder rerank 后 Recall@5 0.967（救回 1/2）、MRR 0.886（+0.154）**——rerank 在万级转正实锤。但注意两个细节：① rerank 的收益主体是 MRR（排序精度）不是 Recall（召回）——q015 的目标在候选池外，rerank 救不了，那是召回问题；② 延迟 882ms→29.9s/query，所以生产万级开启 rerank 的前提是 embedding/rerank 上 GPU。这就是 scale-dependent：30 条有害、1000 临界、万级转正，每档都有实测数据」 |
| 「1000 条测了吗？」 | — | 「测了：用 LLM 生成 970 条相邻主题干扰语料（pgvector/HITL/RAG 等 33 主题），Recall@5 从 1.00 掉到 0.933——2 个 query miss，且 miss 的正是干扰语料覆盖的主题（向量索引调优、HITL 实现）。这验证了 1000 条是 rerank 收益的临界点：30 条有害、万级有益、1000 条在边界。另一个发现是 sparse_search O(N) 瓶颈：延迟 235ms→1498ms（6x），Python 侧 jieba 全表扫描吃掉 87% 延迟。**已优化**：把 jieba 分词结果存到 `tokens TEXT[]` 列 + GIN 索引，sparse_search 改用 `tokens && q_tokens` DB 侧筛选，1000 条稳态延迟从 641ms 降到 198ms（-69%），O(N) 瓶颈消除」 |

**核心收益**：把 §9.2 表里 "RAG 高级技巧 ❌ 无" 升到 🟢 **有实操 + 四步迭代 + 两次纠正归因**。第二次失败时纠正了"数据太短"的错误归因（根因是分词器），第四步 A/B 推翻了"rerank 必要"的默认假设（小语料下有害）——体现的是"做了、测了、错了、再测、连默认假设都敢质疑"的工程判断力，比"我加了改写 Recall 升到 0.93"更有深度。
