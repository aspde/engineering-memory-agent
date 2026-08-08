"""Tests for the central prompt registry (``backend/service/prompts.py``)."""

from __future__ import annotations

import pytest

from backend.service.prompts import PromptSpec, get_prompt


class TestPromptRegistry:
    """get_prompt returns versioned, non-empty text for every registered key."""

    def test_all_registered_prompts_are_versioned_and_non_empty(self) -> None:
        from backend.service import prompts as mod

        assert mod._PROMPTS, "registry must not be empty"
        for key, spec in mod._PROMPTS.items():
            assert isinstance(spec, PromptSpec)
            assert spec.key == key
            assert spec.version, f"prompt {key} has empty version"
            assert spec.text.strip(), f"prompt {key} has empty text"

    def test_get_prompt_returns_version_then_text(self) -> None:
        version, text = get_prompt("agent.system")
        assert isinstance(version, str) and version
        assert "Engineering Memory Agent" in text

    def test_get_prompt_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError):
            get_prompt("does.not.exist")


class TestAgentSystemTemplate:
    """The merged agent system template is a semantic superset of the two
    prompts it replaced (call_llm's persona prompt + generate_final's
    synthesis prompt) and exposes a ``{context}`` placeholder."""

    def test_contains_persona_sections(self) -> None:
        _, text = get_prompt("agent.system")
        # call_llm SYSTEM_PROMPT content — persona, sources, write guidance
        assert "You are EMA, the Engineering Memory Agent for development teams." in text
        assert "search_memories_tool" not in text  # tools are described, not named in prose
        assert "write_memory_tool" in text
        assert "简体中文" in text
        assert "PingCode" in text and "飞书" in text

    def test_contains_synthesis_guidance(self) -> None:
        _, text = get_prompt("agent.system")
        # generate_final_node's inline prompt content
        assert "Answer the user's question based on the conversation" in text
        assert "retrieved context below" in text
        assert "Do NOT list or enumerate the sources" in text

    def test_has_context_placeholder(self) -> None:
        _, text = get_prompt("agent.system")
        assert "{context}" in text
        # Formatting with an empty context must leave no stray placeholder
        assert "{context}" not in text.format(context="")

    def test_context_placeholder_is_fillable(self) -> None:
        _, text = get_prompt("agent.system")
        rendered = text.format(context="\n\nContext:\nhello")
        assert "Context:\nhello" in rendered


class TestModuleReExports:
    """patrol / scenario modules re-export the registry text unchanged."""

    def test_patrol_constants_match_registry(self) -> None:
        from backend.service.patrol_prompts import (
            CI_FAILURE_PATROL_PROMPT,
            DAILY_PATROL_PROMPT,
            JIRA_RESOLVED_PATROL_PROMPT,
            WEEKLY_PATROL_PROMPT,
        )

        assert DAILY_PATROL_PROMPT == get_prompt("patrol.daily")[1]
        assert WEEKLY_PATROL_PROMPT == get_prompt("patrol.weekly")[1]
        assert CI_FAILURE_PATROL_PROMPT == get_prompt("patrol.ci_failure")[1]
        assert JIRA_RESOLVED_PATROL_PROMPT == get_prompt("patrol.jira_resolved")[1]

    def test_scenario_constants_match_registry(self) -> None:
        from backend.service.scenarios.code_review import CODE_REVIEW_SYSTEM_PROMPT
        from backend.service.scenarios.onboarding import ONBOARDING_SYSTEM_PROMPT
        from backend.service.scenarios.postmortem import POSTMORTEM_SYSTEM_PROMPT
        from backend.service.scenarios.tech_debt import TECH_DEBT_SYSTEM_PROMPT

        assert CODE_REVIEW_SYSTEM_PROMPT == get_prompt("scenario.code_review")[1]
        assert ONBOARDING_SYSTEM_PROMPT == get_prompt("scenario.onboarding")[1]
        assert POSTMORTEM_SYSTEM_PROMPT == get_prompt("scenario.postmortem")[1]
        assert TECH_DEBT_SYSTEM_PROMPT == get_prompt("scenario.tech_debt")[1]

    def test_service_prompts_match_registry(self) -> None:
        from backend.service.extraction import _ENTITIES_SCHEMA  # noqa: F401  (module imports fine)
        import backend.service.memory as memory_mod
        import backend.service.query_rewrite as qr_mod

        # The old module-level constants are gone; call sites use the registry.
        assert not hasattr(memory_mod, "_MERGE_PROMPT")
        assert not hasattr(memory_mod, "_CONFLICT_PROMPT")
        assert not hasattr(qr_mod, "_REWRITE_PROMPT")
