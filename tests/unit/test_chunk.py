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

    def test_chinese_sentences_not_cut_mid_sentence(self) -> None:
        """A long Chinese paragraph without newlines must split at sentence
        boundaries (。！？), not fall through to the fixed-width hard split.

        Regression guard for the bug where _DEFAULT_SEPARATORS only matched
        ASCII sentence punctuation ([.!?] followed by whitespace), so a
        Chinese paragraph — sentences end with 。 and no trailing space —
        never split on the sentence separator and was cut mid-sentence.
        """
        text = (
            "这个模块负责记忆去重。"
            "它先计算内容哈希做精确去重。"
            "再按相似度分四档处理。"
            "超过阈值的调用LLM判断。"
            "最后落库并更新衰减因子。"
        )
        chunks = chunk_text(text, max_size=30, overlap=0)
        # Every chunk except the last ends at a sentence boundary; the full
        # text reconstructs exactly (nothing dropped by a mid-word cut).
        assert len(chunks) >= 3
        assert all(c.endswith("。") for c in chunks[:-1])
        assert "".join(chunks) == text

    def test_chinese_and_latin_sentence_boundaries(self) -> None:
        """Mixed CJK + Latin prose splits at both 。/！ and ASCII . ! ?.

        The CJK marks must be split on: a ！-terminated sentence survives as
        its own chunk, and the full text reconstructs (no mid-word cut).
        """
        text = "这是中文句。Then an English sentence follows. 再来一句中文！"
        chunks = chunk_text(text, max_size=40, overlap=0)
        # ！-terminated sentence survives intact as its own trailing chunk.
        assert chunks[-1].rstrip() == "再来一句中文！"
        # Full text reconstructs — nothing lost to a mid-word hard cut.
        assert "".join(chunks).strip() == text

    def test_long_latin_sentence_falls_back_to_word_split(self) -> None:
        """A single Latin sentence longer than max_size still splits at word
        boundaries (space), never mid-word — the CJK fix must not regress
        this."""
        text = "This is a sentence that is much longer than the chunk budget allows for a single piece."
        chunks = chunk_text(text, max_size=16, overlap=0)
        assert all(len(c) <= 16 for c in chunks)
        assert "".join(chunks) == text

    def test_chinese_sentences_with_overlap_keep_boundaries(self) -> None:
        """Overlap must not push a Chinese chunk past a sentence boundary."""
        text = "这是第一句。这是第二句。" + "这是第三句。这是第四句。"
        chunks = chunk_text(text, max_size=30, overlap=10)
        assert all(c.endswith("。") for c in chunks[:-1])
        assert "".join(chunks) == text

    def test_ascii_sentence_boundary_still_requires_whitespace(self) -> None:
        """The Latin sentence separator ([.!?] followed by whitespace) must
        not regress: "Hello.World" (no space) is not split mid-word."""
        text = "Hello.World"
        chunks = chunk_text(text, max_size=6, overlap=0)
        assert "".join(chunks) == text  # hard-split preserves chars, no split on '.'


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
