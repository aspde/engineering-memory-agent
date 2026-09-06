"""Tolerant JSON-object extraction from LLM output.

Both the patrol runner and the event-analysis runner instruct the agent
to emit pure JSON, but models wrap output in markdown fences or add
explanatory text.  This module is the single extraction path: direct
parse → fenced block → greedy brace regex, degrading to a ``raw_output``
wrapper so the original text is never silently dropped.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_RAW_OUTPUT_CAP = 5000


def strip_markdown_fence(text: str) -> str:
    """Return the contents of the first markdown code fence, else *text*.

    Models instructed to emit pure JSON still wrap it in ``` fences (observed
    on the structured-output channel: the retry after a provider timeout
    came back fenced and the strict ``json.loads`` in ``chat_structured``
    rejected it).  Shared by :func:`extract_json_object` (already had this
    logic inline) and ``chat_structured``'s parse retry.
    """
    if "```" not in text:
        return text
    lines = text.split("\n")
    in_fence = False
    fence_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            if in_fence:
                break
            in_fence = True
            continue
        if in_fence:
            fence_lines.append(line)
    return "\n".join(fence_lines) if fence_lines else text


def extract_json_object(raw_text: str, *, label: str = "findings") -> dict | None:
    """Try to parse *raw_text* as a JSON object.

    Empty input returns ``None`` (nothing to store).  Unparseable input
    returns ``{"raw_output": ...}`` — a dict either way so callers can
    persist it uniformly and the UI can show what the model actually said.
    """
    text = raw_text.strip()
    if not text:
        return None

    # Direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Markdown code fences
    fenced = strip_markdown_fence(text)
    if fenced != text:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass

    # Greedy last-{ to first-} window
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse %s as JSON, storing raw text", label)
    return {"raw_output": text[:_RAW_OUTPUT_CAP]}
