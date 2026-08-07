"""Tests for query_rewrite.rewrite_query — LLM rewrite + failsafe fallback.

The LLM provider is mocked at ``llm_service.get_llm_provider`` so no real
API is hit.  The contract under test: ``[original] + N variations``, comment
lines dropped, and every failure mode (empty output, exception) degrades to
``[query]`` so ``retrieve_multi_query`` stays on the single-query baseline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.service.query_rewrite import rewrite_query


def _patch_llm(monkeypatch, resp: str | None = None, *, raise_on_call: bool = False):
    """Patch get_llm_provider to return a fake LLM replying ``resp``."""
    llm = MagicMock()

    async def chat_raw(messages, **kwargs):
        if raise_on_call:
            raise RuntimeError("llm down")
        return {"content": resp}

    llm.chat_raw = chat_raw
    monkeypatch.setattr("backend.service.llm_service.get_llm_provider", lambda: llm)
    return llm


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_returns_original_first_then_variations(self, monkeypatch) -> None:
        _patch_llm(
            monkeypatch,
            "pgvector 兼容性 bug\nPostgresSaver 写入失败\nWindows checkpoint 问题",
        )
        result = await rewrite_query("为什么 Windows 上持久化失败")

        assert result == [
            "为什么 Windows 上持久化失败",
            "pgvector 兼容性 bug",
            "PostgresSaver 写入失败",
            "Windows checkpoint 问题",
        ]

    @pytest.mark.asyncio
    async def test_caps_variations_at_n(self, monkeypatch) -> None:
        _patch_llm(monkeypatch, "v1\nv2\nv3\nv4\nv5")
        result = await rewrite_query("q", n_variations=3)

        # [original] + up to n variations
        assert len(result) == 4

    @pytest.mark.asyncio
    async def test_drops_comment_and_blank_lines(self, monkeypatch) -> None:
        _patch_llm(monkeypatch, "# 这是注释\nv1\n\nv2\n")
        result = await rewrite_query("q", n_variations=3)

        assert result[1:] == ["v1", "v2"]

    @pytest.mark.asyncio
    async def test_empty_output_falls_back_to_original(self, monkeypatch) -> None:
        _patch_llm(monkeypatch, "   \n")
        assert await rewrite_query("q") == ["q"]

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_original(self, monkeypatch) -> None:
        _patch_llm(monkeypatch, raise_on_call=True)
        assert await rewrite_query("q") == ["q"]

    @pytest.mark.asyncio
    async def test_query_with_braces_does_not_crash(self, monkeypatch) -> None:
        """Regression: the prompt was built with str.format; a user query
        containing ``{...}`` raised KeyError.  replace() must not."""
        _patch_llm(monkeypatch, "v1\nv2\nv3")
        result = await rewrite_query("如何使用 {template} 语法", n_variations=2)

        assert result[0] == "如何使用 {template} 语法"
        assert result[1:] == ["v1", "v2"]
