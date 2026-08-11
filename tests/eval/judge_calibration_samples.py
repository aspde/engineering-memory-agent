"""Small-sample calibration set for the LLM answer judge.

The eval's LLM-as-judge (:func:`tests.eval.llm_judge.judge_answer`) grades
whether a model answer is grounded in the provided context and which required
facts it covers.  Its verdicts have no ground truth by construction — they
*are* the measurement.  This module supplies the missing oracle: a
hand-authored sample set with human verdicts, so the judge's agreement with a
human grader can be quantified (see ``tests.eval.judge_calibration``).

Design rules:

- Every sample reuses the ``query`` / ``context`` / ``required_facts`` of one
  :data:`tests.eval.llm_ground_truth.ANSWER_ITEMS` item verbatim (drift-free
  by construction); only the ``answer`` is hand-authored.  The base item is
  referenced by id and resolved at import, so a reworded context never
  desyncs the calibration sample.
- Half the answers are clearly *grounded* (built entirely from the context,
  covering the required facts); half are clearly *ungrounded* — they fabricate
  a fact the context does not support, or contradict it.  Every ungrounded
  answer carries an explicit ``[UNGROUNDED: ...]`` marker naming the
  fabrication point, so the human verdict is deterministic and a disagreement
  with the judge is a real judge error, not label noise.
- Two grounded answers are deliberate *synonym rewrites*: the answer stays
  faithful but does not repeat the required-fact strings verbatim.  This is
  the exact failure mode the semantic baseline flagged on ``ans-006`` (a
  correct paraphrase judged ``grounded=False`` because the judge over-applied
  the "covered_facts must be verbatim" instruction); the samples re-expose it
  so the calibration can quantify how often the judge honours
  "允许同义改写".

The set is small by design: each sample costs one real judge call per run, so
a full pass is 12 calls.  Keep answers hand-authored (never generated) so the
human labels stay reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tests.eval.llm_ground_truth import ANSWER_ITEMS, AnswerItem


# ── Sample model ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationSample:
    id: str
    #: The ``ans-XXX`` item whose query/context/required_facts this sample
    #: reuses verbatim.
    base_id: str
    query: str
    context: str  # the only fact source the answer may use
    required_facts: list[str]
    #: Hand-authored model answer (may contain a ``[UNGROUNDED: ...]`` marker).
    answer: str
    human_grounded: bool
    #: The subset of ``required_facts`` the human judges the answer to cover.
    human_covered_facts: list[str]
    notes: str = ""


_BASE_BY_ID: dict[str, AnswerItem] = {it.id: it for it in ANSWER_ITEMS}


def _base(item_id: str) -> AnswerItem:
    try:
        return _BASE_BY_ID[item_id]
    except KeyError:
        raise ValueError(
            f"judge_calibration_samples: unknown base ANSWER_ITEMS id {item_id!r}"
        ) from None


def _sample(
    sid: str,
    base_id: str,
    answer: str,
    human_grounded: bool,
    covered: list[str],
    notes: str = "",
) -> CalibrationSample:
    """Build a sample that reuses *base_id*'s query/context/required_facts."""
    base = _base(base_id)
    return CalibrationSample(
        id=sid,
        base_id=base_id,
        query=base.query,
        context=base.context,
        required_facts=list(base.required_facts),
        answer=answer,
        human_grounded=human_grounded,
        human_covered_facts=list(covered),
        notes=notes,
    )


# ── Sample set ────────────────────────────────────────────────────
# 12 samples: 6 grounded (2 synonym rewrites) + 6 ungrounded.  The synonym
# rewrites re-expose the ans-006 misjudgement (a faithful paraphrase judged
# ungrounded); the ungrounded answers carry explicit [UNGROUNDED: ...] markers
# so their fabrication points are unambiguous.

CALIBRATION_SAMPLES: list[CalibrationSample] = [
    # ── grounded, direct ──────────────────────────────────────────
    _sample(
        "cal-001",
        "ans-001",
        answer=(
            "最终选用 PostgreSQL 的 pgvector 扩展，而不是 Elasticsearch。"
            "原因：pgvector 与业务库同库，事务一致性有保障；原生支持 cosine "
            "距离；还省去了独立 ES 集群的运维成本。"
        ),
        human_grounded=True,
        covered=["pgvector", "Elasticsearch", "cosine"],
        notes="grounded, 直接复述",
    ),
    # ── grounded, synonym rewrite (pgvector / cosine paraphrased) ──
    _sample(
        "cal-002",
        "ans-001",
        answer=(
            "我们定的是 PostgreSQL 生态里的向量检索扩展方案，彻底放弃了 "
            "Elasticsearch。理由是同库部署、事务一致性有保证、原生支持余弦"
            "距离计算，也省掉了单独运维一套检索集群的成本。"
        ),
        human_grounded=True,
        covered=["pgvector", "Elasticsearch", "cosine"],
        notes="grounded, 同义改写：pgvector→向量检索扩展方案, cosine→余弦距离",
    ),
    # ── grounded, direct ──────────────────────────────────────────
    _sample(
        "cal-003",
        "ans-002",
        answer=(
            "线上 502 的根因是数据库连接池被占满。链路是：流量上涨 → 连接池"
            "没有空闲连接 → 新请求排队超时 → 网关返回 502。修复方式是重启"
            "连接池并增加连接泄漏监控。"
        ),
        human_grounded=True,
        covered=["连接池", "占满", "超时"],
        notes="grounded, 直接复述",
    ),
    # ── grounded, synonym rewrite (事件循环 → ProactorEventLoop) ──
    # Mirrors the exact answer the semantic baseline flagged as a false
    # negative on ans-006: the required fact "事件循环" is replaced by its
    # specific instance "ProactorEventLoop".  The answer fabricates nothing.
    _sample(
        "cal-004",
        "ans-006",
        answer=(
            "开发环境用 InMemorySaver 是因为 psycopg3 的异步驱动和 Windows 的 "
            "ProactorEventLoop 存在兼容性冲突，AsyncPostgresSaver 因此在 Windows "
            "下不可用；生产目标是 Linux，可以正常用 PostgresSaver。"
        ),
        human_grounded=True,
        covered=["psycopg3", "事件循环", "InMemorySaver"],
        notes="grounded, 同义改写：事件循环→ProactorEventLoop — 针对 ans-006 已知误判",
    ),
    # ── grounded, direct ──────────────────────────────────────────
    _sample(
        "cal-005",
        "ans-003",
        answer=(
            "在 Windows 开发环境下不行，重启后对话状态会丢失。AsyncPostgresSaver "
            "因为 psycopg3 异步实现的兼容性问题在 Windows 下不可用，所以降级成了 "
            "InMemorySaver；生产 Linux 环境用 PostgresSaver，可以跨会话恢复。"
        ),
        human_grounded=True,
        covered=["InMemorySaver", "丢失"],
        notes="grounded, 直接复述（否定类）",
    ),
    # ── grounded, direct ──────────────────────────────────────────
    _sample(
        "cal-006",
        "ans-005",
        answer=(
            "嵌入模型选的是 BGE-M3，1024 维，支持中英双语，本地推理零 API 成本；"
            "对比 OpenAI 的 text-embedding-3-large，准确率相当但成本和延迟更优。"
        ),
        human_grounded=True,
        covered=["BGE-M3", "1024", "本地推理"],
        notes="grounded, 直接复述",
    ),
    # ── ungrounded: fabricates Qdrant ──────────────────────────────
    _sample(
        "cal-007",
        "ans-001",
        answer=(
            "最终选型是 Qdrant，而不是 Elasticsearch。"
            "[UNGROUNDED: 声称使用了 Qdrant，但 context 只提到 pgvector 与 "
            "Elasticsearch，从未提及 Qdrant] 理由是同库部署、原生支持 cosine 距离。"
        ),
        human_grounded=False,
        covered=["Elasticsearch", "cosine"],
        notes="ungrounded, 捏造 Qdrant",
    ),
    # ── ungrounded: wrong root cause ───────────────────────────────
    _sample(
        "cal-008",
        "ans-002",
        answer=(
            "线上 502 的根因是内存泄漏。"
            "[UNGROUNDED: 声称根因是内存泄漏，context 明确记载根因是数据库"
            "连接池被占满，且禁止编造「内存泄漏」这个根因] 流量上涨后内存持续"
            "增长，最终请求超时返回 502。"
        ),
        human_grounded=False,
        covered=["超时"],
        notes="ungrounded, 捏造根因（内存泄漏）",
    ),
    # ── ungrounded: contradicts the negation ───────────────────────
    _sample(
        "cal-009",
        "ans-003",
        answer=(
            "Windows 开发环境下重启后对话状态不会丢，状态会被完整保存下来。"
            "[UNGROUNDED: 声称 Windows 下状态不会丢失，context 明确说降级为 "
            "InMemorySaver 后重启状态会丢失]"
        ),
        human_grounded=False,
        covered=[],
        notes="ungrounded, 与否定性事实相反",
    ),
    # ── ungrounded: presents the comparison OpenAI as chosen ───────
    _sample(
        "cal-010",
        "ans-005",
        answer=(
            "嵌入模型最终选了 OpenAI 的 text-embedding-3-large。"
            "[UNGROUNDED: 声称选用了 OpenAI 的 text-embedding-3-large，context "
            "明确说选用的是 BGE-M3，OpenAI 只是对比项] 因为它的准确率更高。"
        ),
        human_grounded=False,
        covered=[],
        notes="ungrounded, 把对比项说成已选用",
    ),
    # ── ungrounded: invented upload capability ─────────────────────
    _sample(
        "cal-011",
        "ans-004",
        answer=(
            "导入本地仓库提交历史很简单，把仓库文件夹拖到网页上传即可，系统会"
            "自动解析提交历史。"
            "[UNGROUNDED: 声称通过网页上传导入，context 只说明要用 "
            "ingest_git_repo_tool 并传入 repo_path 参数，从未提到网页上传]"
        ),
        human_grounded=False,
        covered=[],
        notes="ungrounded, 捏造网页上传能力",
    ),
    # ── ungrounded: fabricated performance reason ──────────────────
    _sample(
        "cal-012",
        "ans-006",
        answer=(
            "开发环境用 InMemorySaver 是因为它比 PostgresSaver 性能更好、启动"
            "更快。"
            "[UNGROUNDED: 声称选 InMemorySaver 是出于性能优势，context 给出的"
            "真实原因是 psycopg3 异步实现与 Windows 事件循环冲突导致 "
            "AsyncPostgresSaver 不可用]"
        ),
        human_grounded=False,
        covered=["InMemorySaver"],
        notes="ungrounded, 捏造性能理由",
    ),
]


def load_calibration_samples() -> list[CalibrationSample]:
    """Return the full calibration set (a copy)."""
    return list(CALIBRATION_SAMPLES)


def validate_calibration_samples(
    samples: list[CalibrationSample] | None = None,
) -> list[str]:
    """Check internal consistency of the sample set.

    Returns a list of human-readable warnings (empty == clean).  Raises
    ``ValueError`` for hard failures that would make calibration results
    meaningless:

        - duplicate sample ids
        - empty query / context / required_facts / answer
        - a ``human_covered_facts`` entry that is not in ``required_facts``
          (the human labels must be a subset of the facts the judge is asked
          about)
        - ``human_grounded`` disagreeing with the answer's ``[UNGROUNDED: ...]``
          marker (an ungrounded answer without a marker, or a grounded answer
          with one) — the marker is what makes the human verdict deterministic,
          so a mismatch means the label is ambiguous.
    """
    warnings: list[str] = []
    seen: set[str] = set()
    for s in samples if samples is not None else CALIBRATION_SAMPLES:
        if s.id in seen:
            raise ValueError(f"duplicate calibration sample id: {s.id}")
        seen.add(s.id)
        if not s.query.strip():
            raise ValueError(f"{s.id}: empty query")
        if not s.context.strip():
            raise ValueError(f"{s.id}: empty context")
        if not s.required_facts:
            raise ValueError(f"{s.id}: required_facts is empty")
        if not s.answer.strip():
            raise ValueError(f"{s.id}: empty answer")
        required = set(s.required_facts)
        for fact in s.human_covered_facts:
            if fact not in required:
                raise ValueError(
                    f"{s.id}: human_covered_facts {fact!r} is not in required_facts "
                    f"{sorted(required)!r} — the human label must be a subset of "
                    "the facts the judge is asked about"
                )
        marker = "[UNGROUNDED:" in s.answer
        if s.human_grounded and marker:
            raise ValueError(
                f"{s.id}: human_grounded=True but the answer contains an "
                "[UNGROUNDED:] marker — grounded labels must not name a fabrication"
            )
        if not s.human_grounded and not marker:
            raise ValueError(
                f"{s.id}: human_grounded=False but the answer has no [UNGROUNDED:] "
                "marker — the fabrication point must be stated so the human "
                "verdict is unambiguous"
            )
    return warnings
