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
    if "```" in text:
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
        if fence_lines:
            try:
                return json.loads("\n".join(fence_lines))
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
