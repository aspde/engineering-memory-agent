# 短板诊断与评估体系补全方案

> 本文档针对 EMA 作为面试项目最危险的短板——**AI 工程化（评估体系 / 成本监控）**——给出可执行的补全方案。
> 目标：面试前用 1-2 天补齐，让"没做评估"变成"有最小评估体系"。

---

## 一、短板全景（基于代码核查）

### 1.1 已核查的短板证据

| 短板 | 代码证据 | 危险度 |
|------|---------|--------|
| 无用户认证 | [ADR-004](../decisions/ADR-004-no-multi-tenancy.md) 明确"当前无用户认证系统" | 🟡 有 ADR 撑腰 |
| 无监控指标 | grep 未发现 prometheus/opentelemetry，仅 `logging.warning` | 🔴 真实疏漏 |
| 未容器化 | [deployment.md](../deployment.md) "backend 待容器化" | 🔴 真实疏漏 |
| 无 CI/CD | 项目自身无 CI（虽接了 CI 连接器） | 🔴 真实疏漏 |
| 巡检内嵌主进程 | [main.py:122-194](../../backend/main.py) 调度器在 FastAPI 主进程 | 🟡 有意取舍 |
| Windows 降级 | [main.py:24-30](../../backend/main.py) PostgresSaver 降级 InMemorySaver | 🟢 已知平台差异 |
| 无评估体系 | 未发现 Recall@K / MRR / LLM-as-judge 实现 | 🔴 对口岗位核心短板 |
| 无成本监控 | [rate_limiter.py](../../backend/service/rate_limiter.py) 只限速不计费 | 🔴 AI 工程化短板 |
| Prompt 散落代码 | [extraction.py](../../backend/service/extraction.py) / [memory.py](../../backend/service/memory.py) 内嵌字符串 | 🟡 中等 |
| 连接池写死 | [db/__init__.py](../../backend/db/__init__.py) 5+10 | 🟢 低 |

### 1.2 优先补全顺序

```
评估体系（§2） → 关键数字实测（§3） → 成本监控（§4） → 压测（§5）
     ↑                                              ↑
   最危险，对口岗位必问                          面试官最爱追问
```

---

## 二、评估体系补全方案（最高优先级）

### 2.1 为什么这是最危险的短板

AI/LLM 应用工程师岗位面试**必问**的三连击：
1. 「你怎么知道你的 RAG 检索准不准？」
2. 「换了个 reranker，效果变好还是变差？怎么量化？」
3. 「你的记忆衰减加权有没有让检索变好？」

没有评估体系 = 这三问全部答不上来 = AI 工程化能力直接被质疑。

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

### 2.3 LLM-as-judge（生成质量评估，可选）

检索质量用 Recall@K，生成质量用 LLM-as-judge：

```python
# tests/eval/llm_judge.py
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

---

## 三、关键数字实测清单

面试官最爱追问的数字，必须实测填实：

### 3.1 必测数字

| 指标 | 测量方法 | 目标值（参考） | 面试话术 |
|------|---------|--------------|---------|
| 记忆库总条数 | `SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL` | [填实测] | "当前记忆库 [X] 条" |
| chunks 总条数 | `SELECT COUNT(*) FROM chunks` | [填实测] | "文档分块 [X] 条" |
| 向量召回 P95 延迟 | 在 `/api/memory/search` 加计时日志 | ~10-50ms | "向量召回 P95 [X]ms" |
| 完整检索 P95（含 rerank） | 同上，覆盖 rerank | ~100-500ms | "检索 P95 [X]ms" |
| Agent 单轮对话 P95 | 在 `/api/agent/chat` 加计时 | ~3-8s | "对话 P95 [X]s" |
| 单次对话平均 token 数 | LLMProvider 加计数（见 §4） | [填实测] | "平均 [X] token/轮" |
| 单次对话平均 tool 调用数 | 在 Agent 加计数 | ~2-3 | "平均调 [X] 个 tool" |
| BGE-M3 embed 单条延迟 | `time` 包裹 `embed()` | ~20-50ms | "embed [X]ms/条" |

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

## 四、成本监控补全

### 4.1 LLMProvider 加 token 计数

```python
# backend/service/llm_service.py —— 在现有 chat/chat_raw 方法加计数

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

在 `chat()` / `chat_raw()` 返回前调 `_record_usage("memory_search", response.usage)`。

### 4.2 面试话术

> 「我在 LLMProvider 加了 token 计数，按场景统计——记忆提取、冲突检测、rerank、Agent 对话分开算。目前日均消耗 [X] token，其中 Agent 对话占 [Y]%，冲突检测占 [Z]%。
>
> 成本控制策略：90% 写入只走向量（零 LLM 成本），只在 0.75-0.92 边界调 LLM；rerank 默认 cross-encoder 本地（零 API 成本）；max_steps=5 防止 Agent 失控烧 token。」

---

## 五、压测补全（locust）

### 5.1 最小压测脚本

```python
# tests/perf/locustfile.py
"""locust 压测 /api/memory/search"""
from locust import HttpUser, task, between
import random

QUERIES = [
    "PostgreSQL 连接池配置",
    "之前怎么修的 OOM",
    "微服务拆分决策",
    "EMA 架构设计",
    "技术债有哪些",
]


class MemorySearchUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://localhost:8000"

    @task
    def search_memories(self):
        self.client.post(
            "/api/memory/search",
            json={"query": random.choice(QUERIES), "top_k": 5},
        )
```

### 5.2 跑压测

```bash
pip install locust
locust -f tests/perf/locustfile.py
# 浏览器打开 http://localhost:8089，设 10 用户、爬升 10s、持续 60s
```

### 5.3 面试话术

> 「我用 locust 压过检索接口，10 并发下 QPS [X]，P95 [Y]ms。瓶颈在 BGE-M3 CPU 推理，单机大概能扛 [Z] QPS。高并发要把 embedding 服务化成独立 batch 服务，用 GPU 推理。」

---

## 六、面试应答总策略

### 6.1 评估体系类问题（必问）

| 问题 | 没补评估前 | 补了评估后 |
|------|-----------|-----------|
| 「怎么知道检索准不准？」 | 老实承认没做 | 「我建了 20 条标注集，Recall@5 是 0.72」 |
| 「换 reranker 效果如何？」 | 讲不出数字 | 「LLM rerank 让 Recall@5 从 0.72 升到 0.78，但成本 X 倍」 |
| 「衰减加权有用吗？」 | 只能讲理论 | 「我对比过开关衰减，开了之后高频记忆的 MRR 提升了 Y」 |

### 6.2 成本类问题

| 问题 | 没补前 | 补了后 |
|------|--------|--------|
| 「一次对话多少钱？」 | 讲不出 | 「平均 X token，按 DeepSeek 价格约 Y 元」 |
| 「怎么控制成本？」 | 讲策略 | 「策略 + 实测：90% 写入零 LLM 成本，日均 X token」 |

### 6.3 规模类问题

| 问题 | 没补前 | 补了后 |
|------|--------|--------|
| 「能扛多少并发？」 | 讲不出 | 「locust 压测 10 并发 QPS X，P95 Yms」 |
| 「瓶颈在哪？」 | 推测 | 「实测瓶颈在 BGE-M3 CPU 推理」 |

---

## 七、执行清单（面试前 2 天）

### Day 1（约 6 小时）

- [ ] 上午：构造 20 条标注集（从记忆库导出 + 人工标注）—— 2h
- [ ] 上午：实现 metrics.py + run_eval.py，跑出 Recall@5 / MRR / NDCG —— 1h
- [ ] 下午：对比 cross-encoder vs LLM rerank，对比开关衰减，记录数字 —— 1h
- [ ] 下午：LLMProvider 加 token 计数，跑 10 轮对话算成本 —— 1h
- [ ] 晚上：在 retrieval/agent 加计时日志，跑 20 次取 P95 —— 1h

### Day 2（约 4 小时）

- [ ] 上午：写 Dockerfile + GitHub Actions CI（跑 pytest）—— 2h
- [ ] 上午：locust 压测，拿 QPS + P95 数字 —— 1h
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

如果只有半天，按这个顺序补：

1. **实测记忆条数 + QPS + P95**（2h）—— 至少有"规模 + 性能"数字
2. **token 计数 + 跑 10 轮算成本**（1h）—— 至少能答"一次对话多少钱"
3. **填实所有占位符**（1h）—— 避免临场卡壳

评估体系（标注集）如果没时间，就用话术应对：「目前没有系统化评估，是已知改进项。如果做，我会建标注集算 Recall@K，用 LLM-as-judge 评生成质量。」—— 诚实 + 有方案，不扣大分。

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
| 双 HITL LangGraph | 🟢 深 | 3 层 | [graph.py:41-51](../../agent/graph.py) max_steps 防循环 + Command(goto) 动态路由 | **主打** |
| 实体归一化双层 | 🟢 深 | 2-3 层 | [entity.py:67-91](../../backend/service/entity.py) 向量粗筛 + [entity.py:160-189](../../backend/service/entity.py) LLM 精判 fails safe | **主打** |
| LLMProvider 抽象 | 🟢 中深 | 2 层 | [llm_service.py:191-199](../../backend/service/llm_service.py) Anthropic system 拆分 + [llm_service.py:157-174](../../backend/service/llm_service.py) tool_calls 双形态处理 | 主打 |
| 衰减加权整合 | 🟡 中 | 2 层 | [retrieval.py:161](../../backend/service/retrieval.py) top_k*4 + [retrieval.py:171](../../backend/service/retrieval.py) `_RERANK_FLOOR=0.15` | 讲整合不讲公式 |
| 三阶段提取 | 🟡 薄 | 1 层 | [extraction.py:153-156](../../backend/service/extraction.py) gather 并行 + zero-shot | 见 §10 优化 |
| rerank | 🟡 薄 | 1-2 层 | [rerank.py:57](../../backend/service/rerank.py) SDK 调用 + [rerank.py:95](../../backend/service/rerank.py) pointwise gather | cross-encoder 原理必须会 |
| vector_search | 🟡 薄 | 1 层 | [retrieval.py:99-104](../../backend/service/retrieval.py) 白名单 filter | 讲 SQL 可见 |
| RAG 高级技巧 | ❌ 无 | 0 层 | 无 query rewriting / HyDE / multi-query / parent-child | 不主动提 |
| Prompt 高级技巧 | ❌ 无 | 0 层 | 全 zero-shot，无 few-shot / CoT | 不主动提 |
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

| 优先级 | 动作 | 耗时 | 收益 |
|--------|------|------|------|
| 🟡 P1 | 加 few-shot examples 到 entity/relation prompt | 2h | 能讲"做过 prompt 优化" |
| 🟡 P1 | 跑 zero-shot vs few-shot 对比，记数字 | 1h | 有量化对比 |
| 🟢 P2 | 实现 function calling 版 extract_entities_v2 | 2h | 能讲"结构化输出" |
| 🟢 P2 | 构造 50 条标注集，算 precision/recall | 3h | 被追问准确率能答 |

### 10.6 面试应答对比

| 问题 | 优化前 | 优化后 |
|------|--------|--------|
| 「为什么不用 few-shot？」 | 答不上 | 「加 few-shot 后 recall 从 0.74 升到 0.81」 |
| 「JSON 解析失败怎么办？」 | 「fails safe 返回空」 | 「改用 function calling，失败率趋近 0」 |
| 「entity 抽取准确率？」 | 没测过 | 「50 条标注集 precision 0.82」 |
| 「prompt 怎么迭代？」 | 靠 git | 「有标注集，每次改动跑评估对比」 |

**核心收益**：把"三阶段提取"从 🟡 薄 升到 🟢 中深，多一个能扛 2 层追问的深度点。
