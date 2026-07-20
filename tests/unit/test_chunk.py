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
