"""LLM-based query rewriting for multi-query retrieval.

Dense vector recall fails on conceptual queries where the query and the
relevant memory share no surface tokens (e.g. "之前出过什么问题" vs
"koa-connect ctx 泄漏").  ``rewrite_query()`` asks the LLM to expand such
queries into concrete-term variations; ``retrieval.retrieve_multi_query()``
embeds each variation, unions the candidates (dedup by chunk id), then
reranks.

This module is deliberately tiny — only the LLM call lives here.  The
multi-query retrieval orchestration stays in ``retrieval.py`` alongside
``retrieve`` and ``retrieve_hybrid`` so all read paths are co-located.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = """\
Rewrite the following query into {n_variations} semantically equivalent
variations that might appear in a technical knowledge base. Focus on concrete
terms, component names, and error types that the query implies but does not
state.

Query: {query}

Output one variation per line, no numbering, no preamble:
"""


async def rewrite_query(query: str, n_variations: int = 3) -> list[str]:
    """Return ``[original] + n`` LLM-generated variations.

    Fails safe: on any error (LLM unavailable, parse failure, empty output)
    returns ``[query]`` so retrieval degrades to the single-query baseline.
    The caller (``retrieve_multi_query``) treats the first element as the
    original query, so a failed rewrite is transparent to downstream code.
    """
    try:
        from backend.service.llm_service import get_llm_provider

        # ``replace`` (not ``format``) so a user query containing ``{…}``
        # can't raise KeyError; unknown placeholders are left verbatim.
        prompt = (
            _REWRITE_PROMPT.replace("{n_variations}", str(n_variations))
            .replace("{query}", query)
        )
        llm = get_llm_provider()
        resp = await llm.chat_raw(
            messages=[{"role": "user", "content": prompt}],
        )
        text_out = str(resp.get("content", "")).strip()
        variations = [
            line.strip()
            for line in text_out.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ][:n_variations]
        if not variations:
            return [query]
        logger.info(
            "query rewrite: %d variations for %r -> %s",
            len(variations), query[:60], variations,
        )
        return [query] + variations
    except Exception:
        logger.exception("query rewrite failed, falling back to original query")
        return [query]
