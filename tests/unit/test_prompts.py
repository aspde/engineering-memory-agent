"""Tests for the central prompt registry (``backend/service/prompts.py``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.service.prompts import PromptSpec, get_prompt

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "_prompt_snapshot.json"


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

    def test_prompt_text_changes_require_version_bump(self) -> None:
        """The registry must match the committed snapshot.

        A text edit that does not also bump the version breaks versioned
        traceability (call-site logs could not tell which behaviour an LLM
        actually saw).  On failure: bump the version of every changed prompt,
        then regenerate the snapshot with
        ``python -m tests.unit.regenerate_prompt_snapshot``.
        """
        from backend.service import prompts as mod

        snapshot = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8"))
        registry = {k: s for k, s in mod._PROMPTS.items()}
        assert registry.keys() == snapshot.keys(), (
            "prompt registry keys differ from the snapshot — register new "
            "prompts with a version, then regenerate the snapshot"
        )
        changed: list[str] = []
        for key, spec in registry.items():
            entry = snapshot[key]
            sha = hashlib.sha256(spec.text.encode("utf-8")).hexdigest()
            if entry["sha256"] != sha or entry["version"] != spec.version:
                changed.append(f"{key}: snapshot v{entry['version']} → registry v{spec.version}")
        assert not changed, (
            "prompt text or version drifted from the snapshot:\n  "
            + "\n  ".join(changed)
            + "\nBump the version of every changed prompt (a text edit is a "
            "behaviour change), then run "
            "`python -m tests.unit.regenerate_prompt_snapshot`."
        )


class TestAgentSystemTemplate:
    """The merged agent system template is a semantic superset of the two
    prompts it replaced (call_llm's persona prompt + generate_final's
    synthesis prompt) and exposes a ``{context}`` placeholder."""

    def test_contains_persona_sections(self) -> None:
        _, text = get_prompt("agent.system")
        # call_llm SYSTEM_PROMPT content — persona, sources, write guidance
        assert "You are EMA, the Engineering Memory Agent for development teams." in text
        # tools are described generically, never named in prose
        assert "search_memories_tool" not in text
        assert "write_memory_tool" not in text
        assert "简体中文" in text
        assert "PingCode" in text and "飞书" in text

    def test_contains_synthesis_guidance(self) -> None:
        _, text = get_prompt("agent.system")
        # generate_final_node's inline prompt content
        assert "Answer the user's question based on the conversation" in text
        assert "retrieved context below" in text
        # Traceability: claims grounded in retrieved content cite their source
        # ID inline, and invented citations are forbidden.
        assert "source ID" in text
        assert "Never invent a source" in text

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
            DAILY_PATROL_PROMPT,
            WEEKLY_PATROL_PROMPT,
        )

        assert DAILY_PATROL_PROMPT == get_prompt("patrol.daily")[1]
        assert WEEKLY_PATROL_PROMPT == get_prompt("patrol.weekly")[1]

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
        import backend.service.memory as memory_mod
        import backend.service.query_rewrite as qr_mod
        from backend.service.extraction import (
            _ENTITIES_SCHEMA,  # noqa: F401  (module imports fine)
        )

        # The old module-level constants are gone; call sites use the registry.
        assert not hasattr(memory_mod, "_MERGE_PROMPT")
        assert not hasattr(memory_mod, "_CONFLICT_PROMPT")
        assert not hasattr(qr_mod, "_REWRITE_PROMPT")


class TestInjectionIsolationDeclarations:
    """Patrol/scenario templates run with their own system message
    (``has_system=True``) and replace agent.system entirely, so each must
    carry its own untrusted-data isolation declaration — the same defence
    agent.system provides for chat (see ``agent.system`` lines ~116-121)."""

    def test_patrol_and_scenario_prompts_declare_untrusted_data(self) -> None:
        from backend.service import prompts as mod

        targets = [
            k for k in mod._PROMPTS
            if k.startswith("patrol.") or k.startswith("scenario.")
        ]
        assert targets, "expected patrol/scenario prompts to be registered"
        for key in targets:
            text = mod._PROMPTS[key].text
            assert "不可信" in text, f"{key} missing untrusted-data declaration"
            assert "忽略" in text, f"{key} missing ignore-instructions declaration"
            assert "绝不执行" in text, f"{key} missing never-execute declaration"
