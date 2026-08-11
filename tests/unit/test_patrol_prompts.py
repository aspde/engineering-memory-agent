"""Tests for patrol prompt templates — deterministic text checks."""

from __future__ import annotations

from backend.service.patrol_prompts import (
    DAILY_PATROL_PROMPT,
    WEEKLY_PATROL_PROMPT,
)


class TestDailyPatrolPrompt:
    """Verify the daily patrol prompt includes all required sections."""

    def test_daily_prompt_includes_all_required_sections(self) -> None:
        prompt = DAILY_PATROL_PROMPT
        assert "pattern_matches" in prompt
        assert "knowledge_gaps" in prompt
        assert "new_entities" in prompt
        assert "output format" in prompt.lower() or "Output your findings" in prompt

    def test_daily_prompt_specifies_similarity_threshold(self) -> None:
        prompt = DAILY_PATROL_PROMPT
        assert "0.85" in prompt

    def test_daily_prompt_requires_json_output(self) -> None:
        prompt = DAILY_PATROL_PROMPT
        assert "JSON" in prompt

    def test_daily_prompt_includes_search_instructions(self) -> None:
        prompt = DAILY_PATROL_PROMPT
        assert "search_memories_tool" in prompt.lower()


class TestWeeklyPatrolPrompt:
    """Verify the weekly patrol prompt includes all required sections."""

    def test_weekly_prompt_includes_contradiction_instructions(self) -> None:
        prompt = WEEKLY_PATROL_PROMPT
        assert "contradiction" in prompt.lower()
        assert "contradictions" in prompt

    def test_weekly_prompt_includes_stale_memory_instructions(self) -> None:
        prompt = WEEKLY_PATROL_PROMPT
        assert "stale_memories" in prompt
        assert "recall_count" in prompt

    def test_weekly_prompt_includes_entity_coverage_instructions(self) -> None:
        prompt = WEEKLY_PATROL_PROMPT
        assert "entity_coverage" in prompt
        assert "coverage" in prompt.lower() or "domains" in prompt.lower()

    def test_weekly_prompt_requires_json_output(self) -> None:
        prompt = WEEKLY_PATROL_PROMPT
        assert "JSON" in prompt

    def test_weekly_prompt_specifies_top_20_entities(self) -> None:
        prompt = WEEKLY_PATROL_PROMPT
        assert "20" in prompt
