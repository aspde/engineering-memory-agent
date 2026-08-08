"""Labeled datasets for the LLM behavior eval — three suites.

The retrieval eval (``tests.eval.ground_truth``) measures *which memories a
search returns*; it cannot answer "did the agent call the right tool", "did
extraction produce correct entities", or "is the final answer grounded in
the retrieved context".  This module carries the labeled sets for those
three LLM-behavior dimensions:

- **tool_selection** — a query + the tool(s) the agent *must* call, the ones
  it must *not* call, and acceptable alternatives.  Driven through the real
  ``call_llm_node`` (real system prompt + real tool schemas) so the eval
  measures production behavior, not a stripped-down harness.
- **extraction** — a source text + the entities / relations / summary
  keywords a correct ``extract_memory`` run should produce.  Entity and
  relation types follow the enum in ``backend.service.extraction``.
- **answer** — a query + a golden retrieved *context* (the only fact source
  the model may use) + ``required_facts`` the answer must cover and
  ``prohibited_claims`` it must not make.  Measures groundedness and
  coverage of the final-answer path in isolation.

The sets are small by design: each item costs 1-6 real LLM calls per run, so
a full pass is ~60-90 calls — cheap enough for a weekly scheduled job.  Keep
them hand-authored (never generated) so drift is human-reviewable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Per-suite categories ───────────────────────────────────────────

TOOL_SELECTION_CATEGORIES: tuple[str, ...] = (
    "memory_search",
    "doc_search",
    "write",
    "ingest",
    "entity",
    "extract",
    "notify",
    "no_tool",
)

EXTRACTION_CATEGORIES: tuple[str, ...] = (
    "code_decision",
    "incident",
    "architecture",
    "process",
)

ANSWER_CATEGORIES: tuple[str, ...] = (
    "factual",
    "causal",
    "instruction",
    "negation",
)

# Entity / relation type enums — must match backend.service.extraction's
# schemas (duplicated here so the dataset module stays decoupled from the
# production prompt internals; validation flags drift below).
ENTITY_TYPES: tuple[str, ...] = (
    "person",
    "project",
    "technology",
    "decision",
    "event",
    "file",
    "concept",
)
RELATION_TYPES: tuple[str, ...] = (
    "depends_on",
    "causes",
    "part_of",
    "contradicts",
    "supersedes",
    "relates_to",
)

# Contexts over 800 chars get truncated by the agent's tool-result cap
# (agent/nodes.py ``_truncate_tool_content``); the answer suite injects
# context directly, but keeping items under the cap keeps the eval honest to
# what the model would actually see through the retrieval path.
ANSWER_CONTEXT_HARD_CAP = 2000
ANSWER_CONTEXT_SOFT_CAP = 800


# ── Tool selection ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSelectionItem:
    id: str
    query: str
    expected_tools: list[str]
    category: str
    #: Tools the agent must never call for this query.
    forbidden_tools: list[str] = field(default_factory=list)
    #: Tools that are acceptable substitutes for the expected ones (near-miss
    #: calls do not count as wrong, but do not satisfy the expected set).
    allowed_tools: list[str] = field(default_factory=list)
    #: Optional arg-precision check: tool name → substrings that must appear
    #: in that call's JSON-serialized arguments (e.g. the search query).
    expected_args: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""


TOOL_SELECTION_ITEMS: list[ToolSelectionItem] = [
    ToolSelectionItem(
        id="tsel-001",
        query="之前 koa-connect 的 ctx 泄漏问题最后是怎么解决的？",
        expected_tools=["search_memories_tool"],
        category="memory_search",
        notes="明确问历史记忆中的解决方案",
    ),
    ToolSelectionItem(
        id="tsel-002",
        query="飞书群里讨论过内存泄漏的问题吗？",
        expected_tools=["search_memories_tool"],
        category="memory_search",
        notes="问飞书来源的历史讨论",
    ),
    ToolSelectionItem(
        id="tsel-003",
        query="我们仓库的 git 历史里有没有关于 pgvector 索引的提交？",
        expected_tools=["search_memories_tool"],
        category="memory_search",
        notes="git 提交已入库为记忆，属记忆检索而非文档检索",
    ),
    ToolSelectionItem(
        id="tsel-004",
        query="查一下 Postman 接口文档里关于认证鉴权的说明",
        expected_tools=["retrieve_chunks_tool"],
        category="doc_search",
        allowed_tools=["query_rewrite_and_search_tool"],
        notes="明确指文档，走 chunks 检索",
    ),
    ToolSelectionItem(
        id="tsel-005",
        query="这个项目 README 里怎么写的部署步骤？",
        expected_tools=["retrieve_chunks_tool"],
        category="doc_search",
        allowed_tools=["query_rewrite_and_search_tool"],
        notes="README 是文档",
    ),
    ToolSelectionItem(
        id="tsel-006",
        query="之前出过什么问题，会不会陷入死循环？",
        expected_tools=["query_rewrite_and_search_tool"],
        category="doc_search",
        allowed_tools=["search_memories_tool", "retrieve_chunks_tool"],
        notes="概念性查询，工具文档明确建议走 query rewrite",
    ),
    ToolSelectionItem(
        id="tsel-007",
        query="记录一下：我们决定下周把 CI 从 GitHub Actions 迁到自建 runner",
        expected_tools=["write_memory_tool"],
        category="write",
        expected_args={"write_memory_tool": ["CI"]},
        notes="用户明确要求记住",
    ),
    ToolSelectionItem(
        id="tsel-008",
        query="帮我记住这个结论：embedding 维度必须统一为 1024",
        expected_tools=["write_memory_tool"],
        category="write",
        expected_args={"write_memory_tool": ["1024"]},
        notes="用户明确要求记住",
    ),
    ToolSelectionItem(
        id="tsel-009",
        query="把 G:\\Projects\\ema 的 git 历史导入进来",
        expected_tools=["ingest_git_repo_tool"],
        category="ingest",
        expected_args={"ingest_git_repo_tool": ["ema"]},
        notes="导入本地仓库",
    ),
    ToolSelectionItem(
        id="tsel-010",
        query="帮我把这份接口文档索引进来，document_id 是 api-doc",
        expected_tools=["ingest_document_tool"],
        category="ingest",
        expected_args={"ingest_document_tool": ["api-doc"]},
        notes="导入文档",
    ),
    ToolSelectionItem(
        id="tsel-011",
        query="分析一下这段文本里提到了哪些技术和人：我们用了 LangGraph 编排，由张工负责。",
        expected_tools=["extract_memory_tool"],
        category="extract",
        notes="只抽取不持久化，必须用 extract 而非 write",
    ),
    ToolSelectionItem(
        id="tsel-012",
        query="PostgreSQL 这个实体在系统里关联了哪些记忆？",
        expected_tools=["query_entity_tool"],
        category="entity",
        allowed_tools=["search_memories_tool"],
        notes="明确查实体档案",
    ),
    ToolSelectionItem(
        id="tsel-013",
        query="上线前飞书通知一下大家 3 点发布",
        expected_tools=["notify_feishu_tool"],
        category="notify",
        notes="外发通知",
    ),
    ToolSelectionItem(
        id="tsel-014",
        query="你好",
        expected_tools=[],
        category="no_tool",
        notes="寒暄，不应触发任何工具",
    ),
    ToolSelectionItem(
        id="tsel-015",
        query="谢谢，我知道了",
        expected_tools=[],
        category="no_tool",
        notes="礼貌收尾，不应触发任何工具",
    ),
]


# ── Extraction ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractionItem:
    id: str
    content: str
    expected_entities: list[dict[str, str]]  # [{"name", "type"}]
    expected_relations: list[dict[str, str]]  # [{"from", "to", "type"}]
    category: str
    summary_keywords: list[str]
    notes: str = ""


EXTRACTION_ITEMS: list[ExtractionItem] = [
    ExtractionItem(
        id="ext-001",
        content=(
            "经过讨论，团队决定用 pgvector 而不是 Elasticsearch 做向量检索，"
            "因为 pgvector 是 PostgreSQL 的扩展，和业务库同库，省去了独立 ES 集群的运维成本。"
        ),
        expected_entities=[
            {"name": "pgvector", "type": "technology"},
            {"name": "Elasticsearch", "type": "technology"},
            {"name": "PostgreSQL", "type": "technology"},
        ],
        expected_relations=[
            {"from": "pgvector", "to": "Elasticsearch", "type": "supersedes"},
            {"from": "pgvector", "to": "PostgreSQL", "type": "part_of"},
        ],
        category="code_decision",
        summary_keywords=["pgvector", "Elasticsearch", "同库"],
    ),
    ExtractionItem(
        id="ext-002",
        content=(
            "昨天线上服务 502 持续 20 分钟，根因是数据库连接池被占满，"
            "修复方式是重启连接池并增加了连接泄漏的监控告警。"
        ),
        expected_entities=[
            {"name": "连接池", "type": "concept"},
            {"name": "502", "type": "event"},
            {"name": "监控告警", "type": "concept"},
        ],
        expected_relations=[
            {"from": "连接池", "to": "502", "type": "causes"},
            {"from": "监控告警", "to": "连接池", "type": "relates_to"},
        ],
        category="incident",
        summary_keywords=["连接池", "502"],
    ),
    ExtractionItem(
        id="ext-003",
        content=(
            "EMA 的 Agent 层通过 LLMProvider 抽象接口调用大模型，不直接依赖 OpenAI SDK；"
            "LLMProvider 定义了 chat、chat_raw 等方法。"
        ),
        expected_entities=[
            {"name": "Agent", "type": "concept"},
            {"name": "LLMProvider", "type": "concept"},
            {"name": "OpenAI", "type": "technology"},
        ],
        expected_relations=[
            {"from": "Agent", "to": "LLMProvider", "type": "depends_on"},
        ],
        category="architecture",
        summary_keywords=["LLMProvider", "抽象接口"],
    ),
    ExtractionItem(
        id="ext-004",
        content=(
            "发布流程包含测试、构建、部署三个阶段；测试失败会导致发布阻塞，"
            "需要先修复再重新发布。"
        ),
        expected_entities=[
            {"name": "发布流程", "type": "concept"},
            {"name": "测试", "type": "concept"},
            {"name": "构建", "type": "concept"},
            {"name": "部署", "type": "concept"},
            {"name": "测试失败", "type": "concept"},
            {"name": "发布阻塞", "type": "concept"},
        ],
        expected_relations=[
            {"from": "测试", "to": "发布流程", "type": "part_of"},
            {"from": "构建", "to": "发布流程", "type": "part_of"},
            {"from": "部署", "to": "发布流程", "type": "part_of"},
            {"from": "测试失败", "to": "发布阻塞", "type": "causes"},
        ],
        category="process",
        summary_keywords=["发布流程", "测试"],
    ),
    ExtractionItem(
        id="ext-005",
        content=(
            "We chose Redis for caching because it gives sub-millisecond reads "
            "and the ops team already runs it in production."
        ),
        expected_entities=[
            {"name": "Redis", "type": "technology"},
            {"name": "caching", "type": "concept"},
        ],
        expected_relations=[
            {"from": "caching", "to": "Redis", "type": "depends_on"},
        ],
        category="code_decision",
        summary_keywords=["Redis", "caching"],
    ),
    ExtractionItem(
        id="ext-006",
        content=(
            "The database connection leak caused memory usage to grow until the "
            "pod was OOM-killed; the fix was closing cursors in a finally block."
        ),
        expected_entities=[
            {"name": "connection leak", "type": "concept"},
            {"name": "OOM-killed", "type": "event"},
            {"name": "cursor", "type": "concept"},
        ],
        expected_relations=[
            {"from": "connection leak", "to": "OOM-killed", "type": "causes"},
        ],
        category="incident",
        summary_keywords=["connection leak", "finally"],
    ),
    ExtractionItem(
        id="ext-007",
        content=(
            "pgvector 是 PostgreSQL 的扩展，支持 cosine 距离，EMA 用它做记忆的向量检索。"
        ),
        expected_entities=[
            {"name": "pgvector", "type": "technology"},
            {"name": "PostgreSQL", "type": "technology"},
            {"name": "EMA", "type": "project"},
        ],
        expected_relations=[
            {"from": "pgvector", "to": "PostgreSQL", "type": "part_of"},
            {"from": "EMA", "to": "pgvector", "type": "depends_on"},
        ],
        category="architecture",
        summary_keywords=["pgvector", "cosine"],
    ),
    ExtractionItem(
        id="ext-008",
        content=(
            "新方案和旧方案在回滚策略上冲突：旧方案支持原地升级，"
            "新方案要求先备份再升级。"
        ),
        expected_entities=[
            {"name": "新方案", "type": "concept"},
            {"name": "旧方案", "type": "concept"},
            {"name": "回滚策略", "type": "concept"},
            {"name": "备份", "type": "concept"},
        ],
        expected_relations=[
            {"from": "新方案", "to": "旧方案", "type": "contradicts"},
            {"from": "备份", "to": "新方案", "type": "part_of"},
        ],
        category="process",
        summary_keywords=["回滚策略", "备份"],
    ),
]


# ── Final answer ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnswerItem:
    id: str
    query: str
    context: str  # the only fact source the answer may use
    required_facts: list[str]
    category: str
    prohibited_claims: list[str] = field(default_factory=list)
    notes: str = ""


ANSWER_ITEMS: list[AnswerItem] = [
    AnswerItem(
        id="ans-001",
        query="我们向量检索后端最终选型是什么？为什么？",
        context=(
            "向量检索后端选型：决定使用 PostgreSQL 的 pgvector 扩展而非 Elasticsearch。"
            "原因：pgvector 与业务库同库，事务一致性有保障；原生支持 cosine 距离；"
            "免去独立 ES 集群的运维成本。"
        ),
        required_facts=["pgvector", "Elasticsearch", "cosine"],
        category="factual",
        prohibited_claims=["最终选用了 Elasticsearch", "选择了 Qdrant", "选择了 Milvus"],
        notes="答案是明确的选型陈述，禁止虚构未提及的备选",
    ),
    AnswerItem(
        id="ans-002",
        query="为什么线上会突然 502？",
        context=(
            "线上 502 事故复盘：根因是数据库连接池被占满。触发链路：流量上涨 → "
            "连接池无空闲连接 → 新请求排队超时 → 网关返回 502。"
            "修复：重启连接池并增加连接泄漏监控。"
        ),
        required_facts=["连接池", "占满", "超时"],
        category="causal",
        prohibited_claims=["内存泄漏", "代码语法错误"],
        notes="根因是连接池占满，禁止编造其他根因",
    ),
    AnswerItem(
        id="ans-003",
        query="Windows 下 Agent 的对话状态能跨重启保存吗？",
        context=(
            "持久化降级：Windows 开发环境下 AsyncPostgresSaver 因 psycopg3 异步实现"
            "兼容性问题不可用，降级为 InMemorySaver。重启后 Agent 状态丢失；"
            "生产 Linux 环境仍用 PostgresSaver，可跨会话恢复。"
        ),
        required_facts=["InMemorySaver", "丢失"],
        category="negation",
        prohibited_claims=["重启后状态不会丢失"],
        notes="正确答案是否定的，禁止肯定化",
    ),
    AnswerItem(
        id="ans-004",
        query="怎样把本地仓库的提交历史导入 EMA？",
        context=(
            "导入能力：EMA 的 ingest_git_repo_tool 支持从本地 Git 仓库读取提交历史并"
            "写入记忆。参数：repo_path 为仓库绝对路径，max_commits 默认 50，"
            "branch 默认 HEAD。"
        ),
        required_facts=["ingest_git_repo_tool", "repo_path"],
        category="instruction",
        prohibited_claims=[],
        notes="操作说明，应引用具体工具与参数",
    ),
    AnswerItem(
        id="ans-005",
        query="嵌入模型选的是什么？为什么不用 OpenAI 的？",
        context=(
            "嵌入模型选型：选用 BGE-M3，1024 维，支持中英双语，本地推理零 API 成本；"
            "对比 OpenAI text-embedding-3-large 准确率相当但成本和延迟更优。"
        ),
        required_facts=["BGE-M3", "1024", "本地推理"],
        category="factual",
        prohibited_claims=["选用了 OpenAI 的 text-embedding-3-large"],
        notes="答案应给出对比理由，禁止把备选说成已选用",
    ),
    AnswerItem(
        id="ans-006",
        query="为什么开发环境要用 InMemorySaver？",
        context=(
            "开发环境持久化决策：psycopg3 的异步实现与 Windows 的 ProactorEventLoop 冲突，"
            "导致 AsyncPostgresSaver 在 Windows 下不可用，因此开发环境降级为 InMemorySaver。"
            "生产目标是 Linux，可正常使用 PostgresSaver。"
        ),
        required_facts=["psycopg3", "事件循环", "InMemorySaver"],
        category="causal",
        prohibited_claims=[],
        notes="根因是事件循环兼容性冲突",
    ),
    AnswerItem(
        id="ans-007",
        query="Agent 一轮最多能执行几步工具调用？",
        context=(
            "Agent 循环防护：max_agent_steps 限制每轮最多执行的 LLM 调用次数，"
            "达到上限后强制进入最终回答节点，防止工具循环失控。"
            "默认值由配置 MAX_AGENT_STEPS 控制。"
        ),
        required_facts=["max_agent_steps", "最终回答"],
        category="factual",
        prohibited_claims=[],
        notes="答案应提到循环上限机制",
    ),
    AnswerItem(
        id="ans-008",
        query="怎么记录一条新的长期记忆？",
        context=(
            "记忆写入：用户要求记住某件事时，Agent 应调用 write_memory_tool 并带上"
            "用户的原话作为 content 参数。冲突检测由工具内置，命中时会暂停等待人工处理，"
            "不需要 Agent 预先自行判断。"
        ),
        required_facts=["write_memory_tool", "content"],
        category="instruction",
        prohibited_claims=["调用前要自己先判断是否冲突"],
        notes="操作说明，禁止给出与上下文相反的建议",
    ),
]


# ── Loading & validation ───────────────────────────────────────────


def load_tool_selection_items() -> list[ToolSelectionItem]:
    return list(TOOL_SELECTION_ITEMS)


def load_extraction_items() -> list[ExtractionItem]:
    return list(EXTRACTION_ITEMS)


def load_answer_items() -> list[AnswerItem]:
    return list(ANSWER_ITEMS)


def _normalize_name(name: str) -> str:
    """Lowercase + strip all whitespace — the same normalization the metrics use."""
    return re.sub(r"\s+", "", str(name).strip().lower())


def _valid_tool_names() -> set[str]:
    """Names of the tools registered in ``agent.tools.ALL_TOOLS``."""
    from agent.tools import ALL_TOOLS

    return {t.name for t in ALL_TOOLS}


def validate_llm_dataset() -> list[str]:
    """Check internal consistency of all three labeled sets.

    Returns a list of human-readable warnings (empty == clean).  Raises
    ``ValueError`` only for hard failures that would make eval results
    meaningless (e.g. a required tool name that does not exist).

    Checks:
        - tool_selection: unique ids; non-empty queries; every tool name in
          expected/allowed/forbidden/expected_args exists in ``ALL_TOOLS``.
        - extraction: unique ids; non-empty content; entity names/types valid;
          every relation's endpoints exist among the golden entities and use
          a valid relation type.
        - answer: unique ids; non-empty query/context/facts; context under the
          hard cap.
    """
    warnings: list[str] = []
    tool_names = _valid_tool_names()

    # ── tool_selection ──
    seen: set[str] = set()
    for it in TOOL_SELECTION_ITEMS:
        if it.id in seen:
            raise ValueError(f"duplicate tool_selection item id: {it.id}")
        seen.add(it.id)
        if not it.query.strip():
            raise ValueError(f"{it.id}: empty query")
        if it.category not in TOOL_SELECTION_CATEGORIES:
            raise ValueError(
                f"{it.id}: unknown category {it.category!r} "
                f"(expected one of {TOOL_SELECTION_CATEGORIES})"
            )
        for tool in it.expected_tools + it.allowed_tools + it.forbidden_tools:
            if tool not in tool_names:
                raise ValueError(f"{it.id}: unknown tool {tool!r} (not in ALL_TOOLS)")
        for tool in it.expected_args:
            if tool not in tool_names:
                raise ValueError(f"{it.id}: expected_args tool {tool!r} not in ALL_TOOLS")
        if not it.expected_tools and it.forbidden_tools:
            warnings.append(
                f"{it.id}: expected_tools empty but forbidden_tools set — "
                "forbidden checks are vacuous unless the model calls something"
            )

    # ── extraction ──
    seen.clear()
    for it in EXTRACTION_ITEMS:
        if it.id in seen:
            raise ValueError(f"duplicate extraction item id: {it.id}")
        seen.add(it.id)
        if not it.content.strip():
            raise ValueError(f"{it.id}: empty content")
        if it.category not in EXTRACTION_CATEGORIES:
            raise ValueError(
                f"{it.id}: unknown category {it.category!r} "
                f"(expected one of {EXTRACTION_CATEGORIES})"
            )
        if not it.expected_entities:
            raise ValueError(f"{it.id}: expected_entities is empty")
        entity_names: list[str] = []
        for e in it.expected_entities:
            name = str(e.get("name", "")).strip()
            etype = str(e.get("type", "")).strip()
            if not name:
                raise ValueError(f"{it.id}: entity with empty name")
            if etype not in ENTITY_TYPES:
                raise ValueError(f"{it.id}: entity {name!r} has invalid type {etype!r}")
            entity_names.append(name)
        norm_names = [_normalize_name(n) for n in entity_names]
        for r in it.expected_relations:
            rtype = str(r.get("type", "")).strip()
            if rtype not in RELATION_TYPES:
                raise ValueError(
                    f"{it.id}: relation has invalid type {rtype!r} "
                    f"(expected one of {RELATION_TYPES})"
                )
            for endpoint in (str(r.get("from", "")), str(r.get("to", ""))):
                if _normalize_name(endpoint) not in norm_names:
                    raise ValueError(
                        f"{it.id}: relation endpoint {endpoint!r} is not a golden "
                        "entity — extraction filters such relations out, so this "
                        "golden label can never match"
                    )
        if not it.summary_keywords:
            raise ValueError(f"{it.id}: summary_keywords is empty")

    # ── answer ──
    seen.clear()
    for it in ANSWER_ITEMS:
        if it.id in seen:
            raise ValueError(f"duplicate answer item id: {it.id}")
        seen.add(it.id)
        if not it.query.strip():
            raise ValueError(f"{it.id}: empty query")
        if not it.context.strip():
            raise ValueError(f"{it.id}: empty context")
        if not it.required_facts:
            raise ValueError(f"{it.id}: required_facts is empty")
        if len(it.context) > ANSWER_CONTEXT_HARD_CAP:
            raise ValueError(
                f"{it.id}: context {len(it.context)} chars exceeds hard cap "
                f"{ANSWER_CONTEXT_HARD_CAP}"
            )
        if len(it.context) > ANSWER_CONTEXT_SOFT_CAP:
            warnings.append(
                f"{it.id}: context {len(it.context)} chars exceeds the agent's "
                f"{ANSWER_CONTEXT_SOFT_CAP}-char tool-result cap — the model would "
                "only see the truncated head through the real retrieval path"
            )
        if it.category not in ANSWER_CATEGORIES:
            raise ValueError(
                f"{it.id}: unknown category {it.category!r} "
                f"(expected one of {ANSWER_CATEGORIES})"
            )

    return warnings
