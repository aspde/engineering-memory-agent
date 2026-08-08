"""Unit tests for tests/eval/llm_judge.py — prompt building + verdict parsing.

The judge functions are thin wrappers over ``chat_structured``; the contract
tested here is that (a) the prompt forwards the item content, and (b) the
verdict is normalized to the shape the runner consumes.  ``chat_structured``
is mocked — the judge LLM itself only runs in the scheduled eval workflow.
"""

from __future__ import annotations

import pytest

import tests.eval.llm_judge as judge_mod


class TestJudgeAnswer:
    @pytest.mark.asyncio
    async def test_normalizes_verdict(self, monkeypatch) -> None:
        async def _fake_structured(messages, *, json_schema, scenario, **kw):
            # Sanity: the item content must be in the prompt.
            assert "pgvector 而非 Elasticsearch" in messages[0]["content"]
            assert "必需事实" in messages[0]["content"]
            return {
                "covered_facts": ["pgvector"],
                "grounded": False,
                "ungrounded_claims": ["用了 Qdrant"],
            }

        monkeypatch.setattr(judge_mod, "chat_structured", _fake_structured)
        verdict = await judge_mod.judge_answer(
            "q", "上下文：pgvector 而非 Elasticsearch", "answer", ["pgvector"]
        )
        assert verdict == {
            "covered_facts": ["pgvector"],
            "grounded": False,
            "ungrounded_claims": ["用了 Qdrant"],
        }

    @pytest.mark.asyncio
    async def test_required_facts_forwarded_verbatim(self, monkeypatch) -> None:
        seen: list = []

        async def _fake_structured(messages, *, json_schema, scenario, **kw):
            seen.append((messages, json_schema, scenario))
            return {"covered_facts": [], "grounded": True, "ungrounded_claims": []}

        monkeypatch.setattr(judge_mod, "chat_structured", _fake_structured)
        await judge_mod.judge_answer("q", "ctx", "ans", ["连接池", "占满"])
        _, schema, scenario = seen[0]
        assert "连接池" in seen[0][0][0]["content"]
        assert schema["required"] == ["covered_facts", "grounded", "ungrounded_claims"]
        assert scenario == "eval_answer_judge"

    @pytest.mark.asyncio
    async def test_empty_answer_gets_placeholder(self, monkeypatch) -> None:
        seen: list = []

        async def _fake_structured(messages, *, json_schema, scenario, **kw):
            seen.append(messages[0]["content"])
            return {"covered_facts": [], "grounded": True, "ungrounded_claims": []}

        monkeypatch.setattr(judge_mod, "chat_structured", _fake_structured)
        await judge_mod.judge_answer("q", "ctx", "", ["f"])
        assert "空答案" in seen[0]


class TestJudgeSummary:
    @pytest.mark.asyncio
    async def test_normalizes_verdict(self, monkeypatch) -> None:
        async def _fake_structured(messages, *, json_schema, scenario, **kw):
            assert "原文" in messages[0]["content"]
            return {"faithfulness": 0.9, "completeness": 0.5}

        monkeypatch.setattr(judge_mod, "chat_structured", _fake_structured)
        verdict = await judge_mod.judge_summary("source text", "summary")
        assert verdict == {"faithfulness": 0.9, "completeness": 0.5}

    @pytest.mark.asyncio
    async def test_invalid_verdict_raises(self, monkeypatch) -> None:
        async def _fake_structured(messages, *, json_schema, scenario, **kw):
            return "not-a-dict"

        monkeypatch.setattr(judge_mod, "chat_structured", _fake_structured)
        with pytest.raises(ValueError, match="non-object"):
            await judge_mod.judge_summary("source", "summary")
