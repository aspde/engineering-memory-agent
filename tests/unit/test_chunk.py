"""Tests for chunk strategies."""

from backend.service.chunk import chunk_code, chunk_text


class TestTextChunk:
    def test_empty(self) -> None:
        assert chunk_text("") == []

    def test_single_chunk(self) -> None:
        chunks = chunk_text("Hello world. This is a test.", max_size=512)
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_splits_long_text(self) -> None:
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks = chunk_text(text, max_size=50)
        assert len(chunks) >= 2

    def test_overlap_applied(self) -> None:
        text = "X" * 80 + "\n\n" + "Y" * 80 + "\n\n" + "Z" * 80
        chunks = chunk_text(text, max_size=50, overlap=10)
        if len(chunks) > 1:
            # overlapping chunks share some text at the boundary
            prev_end = chunks[0][-10:]
            assert prev_end in chunks[1]


class TestCodeChunk:
    def test_empty(self) -> None:
        assert chunk_code("") == []

    def test_python_function_boundaries(self) -> None:
        code = '''\
def foo():
    pass


def bar():
    pass
'''
        chunks = chunk_code(code, max_lines=80)
        # Should produce at least 1 chunk containing both small functions
        assert len(chunks) >= 1

    def test_fallback_non_python(self) -> None:
        code = "line1\nline2\nline3\nline4\nline5\n"
        chunks = chunk_code(code, max_lines=2)
        assert len(chunks) >= 2

    def test_large_function_splits(self) -> None:
        lines = ["def big():\n"]
        lines.extend([f"    x = {i}\n" for i in range(200)])
        code = "".join(lines)
        chunks = chunk_code(code, max_lines=50)
        assert len(chunks) >= 2


class TestHardSplitFallback:
    """A run with no usable separator must never overflow max_size.

    Regression guard for the bug where a single token/word longer than
    ``max_size`` (with no separator left to recurse into) was emitted as an
    oversized chunk — the fallback hard-cuts it into fixed-size pieces.
    """

    def test_no_space_run_split_to_max_size(self) -> None:
        text = "A" * 2000
        chunks = chunk_text(text, max_size=512, overlap=0)
        assert len(chunks) >= 4  # 2000 chars / 512 → at least 4 pieces
        assert all(len(c) <= 512 for c in chunks)
        # Hard-cut preserves every character (no separator to drop).
        assert "".join(chunks) == text

    def test_oversized_token_with_overlap_stays_bounded(self) -> None:
        # Default overlap=64 must not push a hard-cut chunk past max_size.
        chunks = chunk_text("A" * 2000, max_size=512)
        assert len(chunks) > 0
        assert all(len(c) <= 512 for c in chunks)

    def test_oversized_word_mixed_with_separators(self) -> None:
        # A normal phrase followed by a single token far longer than max_size.
        text = "正常句子内容 " + "B" * 1000
        chunks = chunk_text(text, max_size=512, overlap=0)
        assert all(len(c) <= 512 for c in chunks)
        assert "".join(chunks) == text
