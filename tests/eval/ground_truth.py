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
real question a new contributor would ask. This makes the eval set double as
onboarding material — the numbers answer "how well does EMA remember its own
decisions?".
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
    # ── 技术决策扩充 (q031-q038) ──────────────────────────────────
    GroundTruthItem(
        id="q031",
        query="衰减公式的系数后来为什么调大",
        seed_ids=["seed-031"],
        relevant_fingerprints=["0.10 保留 floor"],
        category="技术决策",
        difficulty="hard",
        notes="问公式演进，目标是带校准结果的决策记忆",
    ),
    GroundTruthItem(
        id="q032",
        query="去重阈值现在的值是多少",
        seed_ids=["seed-032"],
        relevant_fingerprints=["0.85 合并"],
        category="技术决策",
        difficulty="easy",
        notes="直接问当前阈值，应命中标定后的决策",
    ),
    GroundTruthItem(
        id="q033",
        query="提取的 prompt 做了什么优化",
        seed_ids=["seed-033"],
        relevant_fingerprints=["relation_f1 从 0.356"],
        category="技术决策",
        difficulty="medium",
        notes="问提取优化，few-shot 是关键词",
    ),
    GroundTruthItem(
        id="q034",
        query="对话为什么不让模型自己开 LLM 重排",
        seed_ids=["seed-034"],
        relevant_fingerprints=["use_llm_rerank 参数被移除"],
        category="技术决策",
        difficulty="hard",
        notes="概念问法，目标是锁死重排的决策",
    ),
    GroundTruthItem(
        id="q035",
        query="运行监控用什么可视化",
        seed_ids=["seed-035"],
        relevant_fingerprints=["Grafana 看板"],
        category="技术决策",
        difficulty="easy",
    ),
    GroundTruthItem(
        id="q036",
        query="数据库备份多久做一次",
        seed_ids=["seed-036"],
        relevant_fingerprints=["-Fc 到 ./backups/"],
        category="技术决策",
        difficulty="easy",
    ),
    GroundTruthItem(
        id="q037",
        query="rerank 什么时候才值得开",
        seed_ids=["seed-037"],
        relevant_fingerprints=["1000 条临界"],
        category="技术决策",
        difficulty="hard",
        notes="概念问法，目标是 scale-dependent 决策记忆",
    ),
    GroundTruthItem(
        id="q038",
        query="LLM 评测怎么避免单次运气影响判断",
        seed_ids=["seed-038"],
        relevant_fingerprints=["多轮均值门禁"],
        category="技术决策",
        difficulty="medium",
        notes="改写问法，目标是 multi_run_gate",
    ),
    # ── 故障复盘扩充 (q039-q046) ──────────────────────────────────
    GroundTruthItem(
        id="q039",
        query="cross-encoder 在小数据集上会怎样",
        seed_ids=["seed-041"],
        relevant_fingerprints=["小语料下有害"],
        category="故障复盘",
        difficulty="medium",
        notes="改写问法，目标是 rerank 有害复盘",
    ),
    GroundTruthItem(
        id="q040",
        query="衰减太激进会有什么后果",
        seed_ids=["seed-042"],
        relevant_fingerprints=["19/30 条目标记忆"],
        category="故障复盘",
        difficulty="hard",
        notes="概念问法，目标是衰减过度复盘",
    ),
    GroundTruthItem(
        id="q041",
        query="一次对话为什么能拖到一分多钟",
        seed_ids=["seed-043"],
        relevant_fingerprints=["14.1s vs 2.5s"],
        category="故障复盘",
        difficulty="medium",
        notes="问延迟根因，目标是 P95 复盘",
    ),
    GroundTruthItem(
        id="q042",
        query="阈值太高会导致什么现象",
        seed_ids=["seed-044"],
        relevant_fingerprints=["merge 从不触发"],
        category="故障复盘",
        difficulty="hard",
        notes="概念问法，目标是阈值失配复盘",
    ),
    GroundTruthItem(
        id="q043",
        query="函数调用抽取为什么精度掉",
        seed_ids=["seed-045"],
        relevant_fingerprints=["未约束只抽显著实体"],
        category="故障复盘",
        difficulty="medium",
        notes="问精度问题，目标是过度抽取复盘",
    ),
    GroundTruthItem(
        id="q044",
        query="审批被拒绝后操作还会执行吗",
        seed_ids=["seed-046"],
        relevant_fingerprints=["拒绝审批后写操作仍执行"],
        category="故障复盘",
        difficulty="easy",
        notes="直接问 bug 现象",
    ),
    GroundTruthItem(
        id="q045",
        query="关键词检索慢的根因是什么",
        seed_ids=["seed-047"],
        relevant_fingerprints=["O(N) 全表扫描"],
        category="故障复盘",
        difficulty="medium",
        notes="问 sparse 瓶颈，O(N) 是关键词",
    ),
    GroundTruthItem(
        id="q046",
        query="jieba 并发调用为什么出问题",
        seed_ids=["seed-048"],
        relevant_fingerprints=["线程安全锁"],
        category="故障复盘",
        difficulty="hard",
        notes="概念问法，目标是 jieba 线程竞争复盘",
    ),
    # ── 架构设计扩充 (q047-q054) ──────────────────────────────────
    GroundTruthItem(
        id="q047",
        query="memory 路径的 cross-encoder 重排和 chunk 路径有什么不同",
        seed_ids=["seed-051"],
        relevant_fingerprints=["竞争区候选重打分"],
        category="架构设计",
        difficulty="medium",
        notes="问 bounded CE 设计",
    ),
    GroundTruthItem(
        id="q048",
        query="LLM 调用数据是怎么存下来的",
        seed_ids=["seed-052"],
        relevant_fingerprints=["后台 flusher"],
        category="架构设计",
        difficulty="easy",
        notes="问 usage 埋点架构",
    ),
    GroundTruthItem(
        id="q049",
        query="熔断器怎么区分不同模型",
        seed_ids=["seed-053"],
        relevant_fingerprints=["base_url|model"],
        category="架构设计",
        difficulty="medium",
        notes="问熔断器命名，base_url|model 是关键",
    ),
    GroundTruthItem(
        id="q050",
        query="检索工具还能传重排参数吗",
        seed_ids=["seed-054"],
        relevant_fingerprints=["移除 use_llm_rerank"],
        category="架构设计",
        difficulty="easy",
        notes="直接问参数状态",
    ),
    GroundTruthItem(
        id="q051",
        query="结构化 LLM 输出怎么保证格式正确",
        seed_ids=["seed-055"],
        relevant_fingerprints=["JSON Schema 校验"],
        category="架构设计",
        difficulty="medium",
        notes="问 chat_structured 架构",
    ),
    GroundTruthItem(
        id="q052",
        query="主模型挂了怎么兜底",
        seed_ids=["seed-056"],
        relevant_fingerprints=["跨 provider failover"],
        category="架构设计",
        difficulty="hard",
        notes="概念问法，目标是 FallbackLLMProvider",
    ),
    GroundTruthItem(
        id="q053",
        query="巡检调度不引 Celery 的考虑是什么",
        seed_ids=["seed-057"],
        relevant_fingerprints=["ADR-007"],
        category="架构设计",
        difficulty="medium",
        notes="问巡检架构取舍",
    ),
    GroundTruthItem(
        id="q054",
        query="中文关键词检索怎么做才不慢",
        seed_ids=["seed-058"],
        relevant_fingerprints=["tokens && 过滤"],
        category="架构设计",
        difficulty="hard",
        notes="问 sparse DB 侧方案",
    ),
    # ── 代码实现扩充 (q055-q062) ──────────────────────────────────
    GroundTruthItem(
        id="q055",
        query="衰减批量更新怎么保证原子",
        seed_ids=["seed-061"],
        relevant_fingerprints=["UPDATE RETURNING"],
        category="代码实现",
        difficulty="medium",
        notes="问 update_decay_batch 实现",
    ),
    GroundTruthItem(
        id="q056",
        query="重复查询为什么不重新算向量",
        seed_ids=["seed-062"],
        relevant_fingerprints=["SSE 重连/复问"],
        category="代码实现",
        difficulty="easy",
        notes="问 embed_query 缓存",
    ),
    GroundTruthItem(
        id="q057",
        query="瞬时失败的调用还能看到吗",
        seed_ids=["seed-063"],
        relevant_fingerprints=["attempts 列"],
        category="代码实现",
        difficulty="hard",
        notes="概念问法，目标是 attempts 会计修复",
    ),
    GroundTruthItem(
        id="q058",
        query="工具签名里删掉了什么参数",
        seed_ids=["seed-064"],
        relevant_fingerprints=["去掉 use_llm_rerank"],
        category="代码实现",
        difficulty="easy",
        notes="直接问签名改动",
    ),
    GroundTruthItem(
        id="q059",
        query="衰减测试为什么要造假数据分布",
        seed_ids=["seed-065"],
        relevant_fingerprints=["cold 48-120h"],
        category="代码实现",
        difficulty="medium",
        notes="问 decay_ab 的 profile 设计",
    ),
    GroundTruthItem(
        id="q060",
        query="门禁判红的标准差怎么算的",
        seed_ids=["seed-066"],
        relevant_fingerprints=["硬编码 t 值"],
        category="代码实现",
        difficulty="hard",
        notes="问 multi_run_gate 统计实现",
    ),
    GroundTruthItem(
        id="q061",
        query="提取的类型约束在哪里生效",
        seed_ids=["seed-067"],
        relevant_fingerprints=["非法类型从机制上产生不出来"],
        category="代码实现",
        difficulty="medium",
        notes="问函数调用工具的 enum",
    ),
    GroundTruthItem(
        id="q062",
        query="备份坏了怎么验证能恢复",
        seed_ids=["seed-068"],
        relevant_fingerprints=["pg_restore 恢复"],
        category="代码实现",
        difficulty="easy",
        notes="问恢复流程",
    ),
    # ── 历史背景扩充 (q063-q070) ──────────────────────────────────
    GroundTruthItem(
        id="q063",
        query="评估怎么从自证走到判别力",
        seed_ids=["seed-071"],
        relevant_fingerprints=["hard-negative 判别集"],
        category="历史背景",
        difficulty="hard",
        notes="概念问法，目标是评估体系演进",
    ),
    GroundTruthItem(
        id="q064",
        query="rerank 的结论中途翻过车吗",
        seed_ids=["seed-072"],
        relevant_fingerprints=["有害到转正"],
        category="历史背景",
        difficulty="medium",
        notes="问 rerank 认知反转",
    ),
    GroundTruthItem(
        id="q065",
        query="面试材料为什么都要带报告链接",
        seed_ids=["seed-073"],
        relevant_fingerprints=["docs/interview"],
        category="历史背景",
        difficulty="medium",
        notes="问材料化 + 诚实口径",
    ),
    GroundTruthItem(
        id="q066",
        query="CI 为什么不直接用 LLM 打分判红",
        seed_ids=["seed-074"],
        relevant_fingerprints=["确定性通道起步"],
        category="历史背景",
        difficulty="hard",
        notes="概念问法，目标是门禁演进",
    ),
    GroundTruthItem(
        id="q067",
        query="对抗性审查查出了什么",
        seed_ids=["seed-075"],
        relevant_fingerprints=["P1 缺陷一批"],
        category="历史背景",
        difficulty="easy",
        notes="直接问审查产出",
    ),
    GroundTruthItem(
        id="q068",
        query="项目现在怎么一键跑起来",
        seed_ids=["seed-076"],
        relevant_fingerprints=["后端+前端单镜像"],
        category="历史背景",
        difficulty="easy",
        notes="问部署演进",
    ),
    GroundTruthItem(
        id="q069",
        query="阈值的数字来源变过吗",
        seed_ids=["seed-077"],
        relevant_fingerprints=["从拍板到标定"],
        category="历史背景",
        difficulty="medium",
        notes="问阈值认知演进",
    ),
    GroundTruthItem(
        id="q070",
        query="对话变快经历了哪几步",
        seed_ids=["seed-078"],
        relevant_fingerprints=["从测量到锁死"],
        category="历史背景",
        difficulty="hard",
        notes="概念问法，目标是 P95 优化三步走",
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
