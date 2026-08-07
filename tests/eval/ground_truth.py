"""Labeled evaluation set for EMA retrieval — 30 queries, 5 categories.

Design choices:
    - **Content fingerprints** instead of UUIDs: ``relevant_fingerprints`` are
      distinctive substrings that must appear in the relevant memory's
      ``summary`` (and, for the chunks-table path, its ``content``). This
      keeps the labeled set portable across DB rebuilds and reproducible in CI.
    - ``seed_ids`` cross-references ``seed_memories.jsonl`` so
      ``dataset.validate()`` can assert every fingerprint is (a) present in
      exactly one seed memory and (b) present in the seed(s) it claims.
    - ``difficulty`` tags how much semantic lifting the retriever must do:
        easy   — query shares surface terms with the target summary
        medium — query is paraphrased or uses synonyms
        hard   — query is conceptual / asks "why" with no lexical overlap
    - Categories are chosen to mirror the five memory-buckets EMA actually
      stores, so per-category numbers map directly to product quality.

The corpus is intentionally EMA's own engineering history: every query is a
real question a new contributor (or an interviewer) would ask. This makes the
eval set double as interview material — the numbers answer "how well does EMA
remember its own decisions?".
"""

from __future__ import annotations

from collections.abc import Sequence

CATEGORIES: tuple[str, ...] = (
    "技术决策",
    "故障复盘",
    "架构设计",
    "代码实现",
    "历史背景",
)

# Difficulty buckets — mirrored by runner.by_difficulty and report._difficulty_table.
# Keep as a tuple so callers can iterate in stable display order.
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


class GroundTruthItem(dict):
    """Typed dict-like accessor for a single labeled query.

    Subclassing ``dict`` keeps JSON serialization trivial while providing
    attribute-style access for the runner.
    """

    @property
    def id(self) -> str:
        return str(self["id"])

    @property
    def query(self) -> str:
        return str(self["query"])

    @property
    def seed_ids(self) -> list[str]:
        return list(self["seed_ids"])

    @property
    def relevant_fingerprints(self) -> list[str]:
        return list(self["relevant_fingerprints"])

    @property
    def category(self) -> str:
        return str(self["category"])

    @property
    def difficulty(self) -> str:
        return str(self.get("difficulty", "medium"))

    @property
    def notes(self) -> str:
        return str(self.get("notes", ""))


GROUND_TRUTH: list[GroundTruthItem] = [
    # ── 技术决策 (6) ──────────────────────────────────────────────
    GroundTruthItem(
        id="q001",
        query="为什么用 pgvector 不用 Elasticsearch 做向量检索",
        seed_ids=["seed-001"],
        relevant_fingerprints=["pgvector 扩展而非 Elasticsearch"],
        category="技术决策",
        difficulty="easy",
        notes="表面词重合，应被向量召回轻松命中",
    ),
    GroundTruthItem(
        id="q002",
        query="LangGraph 比普通 LangChain Agent 好在哪",
        seed_ids=["seed-002"],
        relevant_fingerprints=["LangGraph 而非 LangChain"],
        category="技术决策",
        difficulty="medium",
        notes="改写为比较句式，需要语义理解",
    ),
    GroundTruthItem(
        id="q003",
        query="嵌入模型选型理由是什么",
        seed_ids=["seed-003"],
        relevant_fingerprints=["BGE-M3 嵌入模型"],
        category="技术决策",
        difficulty="hard",
        notes="无词重合，纯概念查询，最考验向量质量",
    ),
    GroundTruthItem(
        id="q004",
        query="Windows 环境下 LangGraph 持久化怎么降级",
        seed_ids=["seed-004"],
        relevant_fingerprints=["InMemorySaver 降级"],
        category="技术决策",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q005",
        query="rerank 为什么默认走本地不走 LLM",
        seed_ids=["seed-005"],
        relevant_fingerprints=["cross-encoder 本地 rerank"],
        category="技术决策",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q006",
        query="记忆去重的相似度阈值是怎么定的",
        seed_ids=["seed-006"],
        relevant_fingerprints=["0.92/0.75/0.60"],
        category="技术决策",
        difficulty="hard",
        notes="查询是抽象问法，目标是带具体数字的决策记忆",
    ),
    # ── 故障复盘 (6) ──────────────────────────────────────────────
    GroundTruthItem(
        id="q007",
        query="koa-connect 之前出过什么问题",
        seed_ids=["seed-007"],
        relevant_fingerprints=["ctx 泄漏"],
        category="故障复盘",
        difficulty="easy",
    ),
    GroundTruthItem(
        id="q008",
        query="PostgresSaver 在 Windows 上为什么不能用",
        seed_ids=["seed-008"],
        relevant_fingerprints=["PostgresSaver Windows 兼容性"],
        category="故障复盘",
        difficulty="easy",
    ),
    GroundTruthItem(
        id="q009",
        query="BGE-M3 性能瓶颈在哪",
        seed_ids=["seed-009"],
        relevant_fingerprints=["BGE-M3 CPU 推理"],
        category="故障复盘",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q010",
        query="向量索引召回不准怎么调",
        seed_ids=["seed-010"],
        relevant_fingerprints=["ivfflat lists=100"],
        category="故障复盘",
        difficulty="hard",
        notes="查询未提 ivfflat，考验能否召回具体技术细节",
    ),
    GroundTruthItem(
        id="q011",
        query="LLM 返回的 JSON 解析失败怎么处理",
        seed_ids=["seed-011"],
        relevant_fingerprints=["JSON 解析失败"],
        category="故障复盘",
        difficulty="easy",
    ),
    GroundTruthItem(
        id="q012",
        query="Agent 会不会陷入死循环",
        seed_ids=["seed-012"],
        relevant_fingerprints=["强制进入 finalize 节点"],
        category="故障复盘",
        difficulty="hard",
        notes="概念问法，目标是带 max_steps 的实现记忆",
    ),
    # ── 架构设计 (6) ──────────────────────────────────────────────
    GroundTruthItem(
        id="q013",
        query="Agent 的状态图有几个节点",
        seed_ids=["seed-013"],
        relevant_fingerprints=["5-node StateGraph"],
        category="架构设计",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q014",
        query="怎么同时支持 OpenAI 和 Anthropic 两家 API",
        seed_ids=["seed-014"],
        relevant_fingerprints=["LLMProvider 抽象层"],
        category="架构设计",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q015",
        query="人工介入 HITL 是怎么实现的",
        seed_ids=["seed-015"],
        relevant_fingerprints=["第一个 HITL 在 resolve_conflict 节点"],
        category="架构设计",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q016",
        query="遗忘曲线衰减怎么整合进检索",
        seed_ids=["seed-016"],
        relevant_fingerprints=["衰减加权整合"],
        category="架构设计",
        difficulty="hard",
    ),
    GroundTruthItem(
        id="q017",
        query="记忆提取为什么要分三个阶段",
        seed_ids=["seed-017"],
        relevant_fingerprints=["asyncio.gather 并行调 LLM"],
        category="架构设计",
        difficulty="hard",
    ),
    GroundTruthItem(
        id="q018",
        query="同名实体怎么归一化",
        seed_ids=["seed-018"],
        relevant_fingerprints=["实体归一化双层判断"],
        category="架构设计",
        difficulty="medium",
    ),
    # ── 代码实现 (6) ──────────────────────────────────────────────
    GroundTruthItem(
        id="q019",
        query="去重逻辑的四个分支怎么实现",
        seed_ids=["seed-019"],
        relevant_fingerprints=["四分支去重"],
        category="代码实现",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q020",
        query="_deferred 字段是干什么的",
        seed_ids=["seed-020"],
        relevant_fingerprints=["state['_deferred']"],
        category="代码实现",
        difficulty="easy",
    ),
    GroundTruthItem(
        id="q021",
        query="关系三元组怎么避免重复写入",
        seed_ids=["seed-021"],
        relevant_fingerprints=["关系三元组去重"],
        category="代码实现",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q022",
        query="token 计数在并发下怎么保证正确",
        seed_ids=["seed-022"],
        relevant_fingerprints=["token 计数线程安全"],
        category="代码实现",
        difficulty="hard",
    ),
    GroundTruthItem(
        id="q023",
        query="retrieve 为什么要过采样 4 倍",
        seed_ids=["seed-023"],
        relevant_fingerprints=["top_k*4 过采样"],
        category="代码实现",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q024",
        query="rerank 最低分阈值 0.15 是什么",
        seed_ids=["seed-024"],
        relevant_fingerprints=["_RERANK_FLOOR=0.15 阈值"],
        category="代码实现",
        difficulty="easy",
    ),
    # ── 历史背景 (6) ──────────────────────────────────────────────
    GroundTruthItem(
        id="q025",
        query="EMA 这个项目是怎么演进的",
        seed_ids=["seed-025"],
        relevant_fingerprints=["从 RAG demo 演进"],
        category="历史背景",
        difficulty="hard",
    ),
    GroundTruthItem(
        id="q026",
        query="项目分了几个阶段",
        seed_ids=["seed-026"],
        relevant_fingerprints=["EMA Phase 1-4 阶段划分"],
        category="历史背景",
        difficulty="easy",
    ),
    GroundTruthItem(
        id="q027",
        query="为什么用 Ebbinghaus 遗忘曲线做衰减",
        seed_ids=["seed-027"],
        relevant_fingerprints=["Ebbinghaus 衰减"],
        category="历史背景",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q028",
        query="为什么要分 chunks 和 memories 两张表",
        seed_ids=["seed-028"],
        relevant_fingerprints=["chunks 和 memories 两表"],
        category="历史背景",
        difficulty="medium",
    ),
    GroundTruthItem(
        id="q029",
        query="连接器支持哪些平台",
        seed_ids=["seed-029"],
        relevant_fingerprints=["扩展到 Feishu"],
        category="历史背景",
        difficulty="hard",
        notes="查询未提具体平台，考验能否召回扩展历史",
    ),
    GroundTruthItem(
        id="q030",
        query="巡检为什么放在主进程里",
        seed_ids=["seed-030"],
        relevant_fingerprints=["巡检内嵌主进程"],
        category="历史背景",
        difficulty="medium",
    ),
]


def by_category(items: Sequence[GroundTruthItem]) -> dict[str, list[GroundTruthItem]]:
    """Group labeled items by category, preserving order."""
    out: dict[str, list[GroundTruthItem]] = {c: [] for c in CATEGORIES}
    for it in items:
        out.setdefault(it.category, []).append(it)
    return out


def difficulty_distribution(items: Sequence[GroundTruthItem]) -> dict[str, int]:
    """Count items per difficulty bucket — used in the report header."""
    out: dict[str, int] = {d: 0 for d in DIFFICULTIES}
    for it in items:
        out[it.difficulty] = out.get(it.difficulty, 0) + 1
    return out


def assert_complete() -> None:
    """Sanity-check the labeled set at import time of the CLI.

    Cheap invariants that catch typos before a 20-minute eval run:
        - IDs are unique
        - Every category has ≥1 item
        - Every item has ≥1 fingerprint and ≥1 seed_id
    """
    ids = [it.id for it in GROUND_TRUTH]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate query ids: {sorted(dupes)}")
    for cat in CATEGORIES:
        if not any(it.category == cat for it in GROUND_TRUTH):
            raise ValueError(f"category has no items: {cat}")
    for it in GROUND_TRUTH:
        if not it.relevant_fingerprints:
            raise ValueError(f"{it.id}: empty relevant_fingerprints")
        if not it.seed_ids:
            raise ValueError(f"{it.id}: empty seed_ids")
