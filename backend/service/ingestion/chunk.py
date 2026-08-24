"""Chunk strategies — plain functions, testable in isolation.

- chunk_text(): recursive separator splitter for prose / docs
- chunk_code(): AST-aware splitter for source files
"""

from __future__ import annotations

import ast
import re

# Default separators ordered from coarsest to finest.
# The algorithm tries to split at the coarsest boundary first,
# falling back to finer ones only when a chunk would exceed max_size.
#
# Sentence boundary covers BOTH Latin and CJK punctuation:
#   (?<=[.!?])\s+   — Latin: period/question/exclamation followed by whitespace
#   (?<=[。！？])     — CJK: full stop / exclamation / question mark (zero-width
#                     lookbehind — a Chinese sentence ends with the mark and
#                     usually no trailing space, so no \s+ is required).  This
#                     is the separator that keeps a long Chinese paragraph
#                     (a commit message / discussion / doc in the target
#                     corpus) from falling through to the fixed-width
#                     _hard_split and being cut mid-sentence.
#
# IMPORTANT: the last entry must stay the word/whitespace fallback ``\s+`` —
# the overlap re-split path uses ``_DEFAULT_SEPARATORS[-1:]`` as the finest
# separator, and that must remain the generic whitespace splitter.
_DEFAULT_SEPARATORS = [
    r"\n\n",
    r"\n",
    r"(?<=[.!?])\s+",
    r"(?<=[。！？])",
    r"(?<=[;；])\s*",
    r"\s+",
]


def chunk_text(text: str, max_size: int = 512, overlap: int = 64) -> list[str]:
    """Split prose text into overlapping chunks at natural boundaries.

    Tries paragraph, then line, then sentence, then word boundaries.
    Each chunk is at most *max_size* characters; adjacent chunks share
    *overlap* characters of context.

    Note: ``max_size`` counts *characters*, not tokens.  512 chars stays well
    under BGE-M3's 8192-token ceiling even for dense CJK text (≈1 token per
    char), so no chunk risks being truncated at embed time.
    """
    if not text.strip():
        return []

    separators = _DEFAULT_SEPARATORS[:]
    return _split_recursive(text, separators, max_size, overlap)


def chunk_code(code: str, max_lines: int = 80) -> list[str]:
    """Split source code at function / class / method boundaries.

    Uses the stdlib ``ast`` module — works for Python files only.
    For unsupported languages, falls back to line-based chunking.
    """
    if not code.strip():
        return []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Not valid Python — fall back to line-based chunking
        return _chunk_by_lines(code, max_lines)

    chunks: list[str] = []
    lines = code.splitlines(keepends=True)

    # Collect top-level nodes with line ranges, sorted by start line
    nodes = _collect_nodes(tree)
    prev_end = 0
    current: list[str] = []

    for node_start, node_end, _ in nodes:
        # Lines between previous node end and this node start
        gap = lines[prev_end:node_start]
        if gap:
            if _line_count(current) + _line_count(gap) > max_lines and current:
                chunks.append("".join(current))
                current = []
            current.extend(gap)

        node_lines = lines[node_start:node_end]
        if _line_count(current) + _line_count(node_lines) > max_lines and current:
            chunks.append("".join(current))
            current = []

        if _line_count(node_lines) > max_lines:
            # Very large node — fall back to line chunks inside it
            if current:
                chunks.append("".join(current))
                current = []
            chunks.extend(_chunk_by_lines("".join(node_lines), max_lines))
        else:
            current.extend(node_lines)

        prev_end = node_end

    # Remaining trailing lines
    remainder = lines[prev_end:]
    if remainder:
        if _line_count(current) + _line_count(remainder) > max_lines and current:
            chunks.append("".join(current))
            current = []
        current.extend(remainder)

    if current:
        chunks.append("".join(current))

    return chunks


# ── internal helpers ────────────────────────────────────────────


def _split_recursive(
    text: str, separators: list[str], max_size: int, overlap: int
) -> list[str]:
    """Recursively split on the first separator, then the next, etc."""
    sep = separators[0]
    remaining = separators[1:]

    parts = re.split(f"({sep})", text)
    chunks: list[str] = []
    buf = ""

    for part in parts:
        if len(buf) + len(part) <= max_size:
            buf += part
        else:
            if buf:
                if remaining:
                    chunks.extend(_split_recursive(buf, remaining, max_size, overlap))
                else:
                    chunks.extend(_hard_split(buf, max_size))
            if len(part) > max_size:
                # A single part longer than max_size — prefer the finer
                # separators still left in ``remaining`` before hard-cutting,
                # so a long single paragraph is split at sentence / word
                # boundaries instead of at arbitrary fixed-width offsets.
                # Only when every separator is exhausted (``remaining`` empty)
                # is the run hard-cut in place.
                if remaining:
                    chunks.extend(_split_recursive(part, remaining, max_size, overlap))
                else:
                    chunks.extend(_hard_split(part, max_size))
                buf = ""
            else:
                buf = part

    if buf:
        if remaining and len(buf) > max_size:
            chunks.extend(_split_recursive(buf, remaining, max_size, overlap))
        else:
            chunks.extend(_hard_split(buf, max_size))

    # Optional: apply overlap by prepending tail of previous chunk,
    # then re-check size — overlap must not push a chunk past max_size.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            if len(prev) > overlap:
                combined = prev[-overlap:] + chunks[i]
            else:
                combined = chunks[i]

            if len(combined) > max_size:
                # Re-split via finest separator to bring back under max_size
                trimmed = _split_recursive(combined, _DEFAULT_SEPARATORS[-1:], max_size, overlap=0)
                overlapped.extend(trimmed)
            else:
                overlapped.append(combined)
        return overlapped

    return chunks


def _hard_split(text: str, max_size: int) -> list[str]:
    """Fallback: split a run with no usable separator into fixed-size pieces.

    Guarantees no output chunk exceeds *max_size* characters.  Only reached
    when recursion has exhausted every separator and a single token/word (or
    a separator-less tail) is still longer than the budget.
    """
    return [text[i : i + max_size] for i in range(0, len(text), max_size)]


def _collect_nodes(tree: ast.AST) -> list[tuple[int, int, str]]:
    """Return (start_line_0, end_line, name) for top-level definitions."""
    nodes: list[tuple[int, int, str]] = []
    for node in ast.iter_child_nodes(tree):
        start = getattr(node, "lineno", 0) - 1
        end = getattr(node, "end_lineno", start + 1)
        name = _node_name(node)
        nodes.append((start, end, name))
    nodes.sort(key=lambda x: x[0])
    return nodes


def _node_name(node: ast.AST) -> str:
    if isinstance(node, ast.FunctionDef):
        return node.name
    if isinstance(node, ast.AsyncFunctionDef):
        return node.name
    if isinstance(node, ast.ClassDef):
        return node.name
    return type(node).__qualname__


def _line_count(lines: list[str]) -> int:
    return len(lines)


def _chunk_by_lines(code: str, max_lines: int) -> list[str]:
    """Fallback: chunk source line-by-line."""
    all_lines = code.splitlines(keepends=True)
    chunks: list[str] = []
    for i in range(0, len(all_lines), max_lines):
        chunks.append("".join(all_lines[i : i + max_lines]))
    return chunks
