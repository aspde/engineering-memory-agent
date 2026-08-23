"""LLM-as-judge for the LLM behavior eval.

The extraction and answer suites need to judge *generated* content, which
substring matching cannot: a correct summary can paraphrase a keyword, and a
hallucinated answer can share surface tokens with the truth.  This module
runs a second (cheap) LLM call per item to grade:

- **answer** — which required facts the answer actually covers, whether it
  stays grounded in the provided context, and which claims are ungrounded.
- **summary** — faithfulness (does the summary invent anything) and
  completeness (does it keep the key facts), both on a 0-1 scale.

Judges use :func:`backend.service.structured.chat_structured` so the verdict
is schema-validated and retried — the raw-response "parse a score out of
whatever the model said" pattern from the original gap-remediation sketch is
deliberately avoided.  Prompts live here (not in ``backend.service.prompts``)
because they are eval-only and never reach production call sites.
"""

from __future__ import annotations

import json
from typing import Any

from backend.model.llm import LLMProvider
from backend.service.llm_service import get_judge_provider
from backend.service.structured import chat_structured

# ── Answer judge ────────────────────────────────────────────────────
# The judge is asked to name which of the given required_facts appear in the
# answer (verbatim, so coverage is computable by set membership), whether the
# answer stays grounded in the context, and — when not — to list the
# ungrounded claims for the report's forensic detail.

ANSWER_JUDGE_PROMPT = """\
你是一个严格的答案评测助手。判断下面模型答案是否忠实于给定的上下文、是否回答了给定的问题。

被评测的问题:
{query}

上下文（唯一事实来源，答案只能依据它）:
{context}

模型答案:
{answer}

必需事实列表（每一项都应出现在答案中，允许同义改写）:
{required_facts}

请输出 JSON：
- covered_facts: 答案中确实出现的必需事实（取上方列表中的原文，未出现的不要列入）
- grounded: 布尔值，答案是否只包含上下文支持的内容、没有额外捏造
- ungrounded_claims: 当 grounded 为 false 时，列出答案中缺乏上下文支撑的具体论断；为 true 时给空数组
"""

ANSWER_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["covered_facts", "grounded", "ungrounded_claims"],
    "properties": {
        "covered_facts": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "grounded": {"type": "boolean"},
        "ungrounded_claims": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}


async def judge_answer(
    query: str,
    context: str,
    answer: str,
    required_facts: list[str],
    *,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Grade one final answer.  Returns the validated verdict dict.

    The judge LLM runs on the dedicated judge provider
    (``get_judge_provider``) when one is configured — an independent model
    from the one being evaluated, avoiding same-model self-judging.  When no
    ``LLM_JUDGE_*`` config is set it falls back to the primary provider.
    ``provider`` is an explicit injection point for tests.
    """
    prompt = ANSWER_JUDGE_PROMPT.format(
        query=query or "(未提供问题)",
        context=context,
        answer=answer or "(空答案)",
        required_facts=json.dumps(required_facts, ensure_ascii=False),
    )
    verdict = await chat_structured(
        [{"role": "user", "content": prompt}],
        json_schema=ANSWER_JUDGE_SCHEMA,
        scenario="eval_answer_judge",
        provider=provider or get_judge_provider(),
    )
    # Guard against a schema-valid-but-empty verdict.
    if not isinstance(verdict, dict):
        raise ValueError(f"answer judge returned non-object: {verdict!r}")
    return {
        "covered_facts": list(verdict.get("covered_facts") or []),
        "grounded": bool(verdict.get("grounded", False)),
        "ungrounded_claims": list(verdict.get("ungrounded_claims") or []),
    }


# ── Summary judge (extraction suite) ────────────────────────────────

SUMMARY_JUDGE_PROMPT = """\
你是一个摘要质量评测助手。根据原文评判给定摘要。

原文:
{source}

摘要:
{summary}

请输出 JSON：
- faithfulness: 0 到 1 的数字，摘要是否只包含原文支持的信息、没有捏造（0=大量捏造，1=完全忠实）
- completeness: 0 到 1 的数字，摘要是否覆盖了原文的关键事实（0=遗漏大量关键信息，1=完整覆盖）
"""

SUMMARY_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["faithfulness", "completeness"],
    "properties": {
        "faithfulness": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "completeness": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
    },
}


async def judge_summary(
    source: str,
    summary: str,
    *,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    """Grade one extracted summary.  Returns the validated verdict dict.

    Judge LLM selection mirrors ``judge_answer`` — the dedicated judge
    provider when configured, the primary otherwise.
    """
    prompt = SUMMARY_JUDGE_PROMPT.format(
        source=source,
        summary=summary or "(空摘要)",
    )
    verdict = await chat_structured(
        [{"role": "user", "content": prompt}],
        json_schema=SUMMARY_JUDGE_SCHEMA,
        scenario="eval_summary_judge",
        provider=provider or get_judge_provider(),
    )
    if not isinstance(verdict, dict):
        raise ValueError(f"summary judge returned non-object: {verdict!r}")
    return {
        "faithfulness": float(verdict.get("faithfulness", 0.0)),
        "completeness": float(verdict.get("completeness", 0.0)),
    }
