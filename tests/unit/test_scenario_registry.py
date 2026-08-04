"""Tests for the scenario registry itself — visibility, fields, and status filtering."""

from __future__ import annotations

import pytest


class TestScenarioRegistry:
    """SCENARIOS dict completeness and structure."""

    def test_registry_includes_all_four_scenarios(self):
        from backend.service.scenarios import SCENARIOS

        assert set(SCENARIOS.keys()) == {
            "postmortem",
            "code_review",
            "onboarding",
            "tech_debt",
        }

    def test_each_scenario_has_required_fields(self):
        from backend.service.scenarios import SCENARIOS

        required = {"name", "compose", "triggers", "status"}
        for key, info in SCENARIOS.items():
            missing = required - set(info.keys())
            assert not missing, f"Scenario '{key}' missing fields: {missing}"

    def test_each_compose_path_is_valid_module(self):
        import importlib

        from backend.service.scenarios import SCENARIOS

        for key, info in SCENARIOS.items():
            compose_path = info["compose"]
            module_path, func_name = compose_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                assert hasattr(module, func_name), (
                    f"Scenario '{key}': module {module_path} has no {func_name}"
                )
            except ImportError as exc:
                pytest.fail(f"Scenario '{key}': cannot import {module_path}: {exc}")

    def test_each_compose_function_is_callable(self):
        import importlib
        import inspect

        from backend.service.scenarios import SCENARIOS

        for key, info in SCENARIOS.items():
            compose_path = info["compose"]
            module_path, func_name = compose_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            assert inspect.iscoroutinefunction(func), (
                f"Scenario '{key}': {func_name} must be an async function"
            )

    def test_each_status_is_valid(self):
        from backend.service.scenarios import SCENARIOS

        valid_statuses = {"active", "beta", "inactive"}
        for key, info in SCENARIOS.items():
            assert info["status"] in valid_statuses, (
                f"Scenario '{key}': invalid status '{info['status']}'"
            )


class TestVisibleScenarios:
    """visible_scenarios() filtering behaviour."""

    def test_default_excludes_inactive(self):
        from backend.service.scenarios import SCENARIOS, visible_scenarios

        original = SCENARIOS["postmortem"]["status"]
        try:
            SCENARIOS["postmortem"]["status"] = "inactive"
            visible = visible_scenarios()
            assert "postmortem" not in visible
        finally:
            SCENARIOS["postmortem"]["status"] = original

    def test_include_beta_shows_beta(self):
        from backend.service.scenarios import SCENARIOS, visible_scenarios

        original = SCENARIOS["postmortem"]["status"]
        try:
            SCENARIOS["postmortem"]["status"] = "beta"
            # Without include_beta
            visible_no_beta = visible_scenarios(include_beta=False)
            assert "postmortem" not in visible_no_beta
            # With include_beta
            visible_with_beta = visible_scenarios(include_beta=True)
            assert "postmortem" in visible_with_beta
        finally:
            SCENARIOS["postmortem"]["status"] = original

    def test_active_always_visible(self):
        from backend.service.scenarios import visible_scenarios

        visible = visible_scenarios()
        assert "code_review" in visible  # code_review is active by default
        assert visible["code_review"]["status"] == "active"
