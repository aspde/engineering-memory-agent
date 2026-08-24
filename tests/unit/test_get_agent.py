"""Tests for ``agent_service.get_agent`` argument handling.

The approval-set override had a falsy-or bug: an explicitly passed empty
frozenset (the "unattended" marker used by patrol / scenarios / event
analysis) was swallowed back to the default write/ingest approval set
because ``frozenset() or DEFAULT`` evaluates to ``DEFAULT``.  These tests
pin the corrected semantics at the ``build_agent_graph`` boundary.
"""

from __future__ import annotations

from unittest.mock import patch

from backend.runner.agent_service import APPROVAL_REQUIRED_TOOLS, get_agent


class TestGetAgentApprovalSet:
    def test_none_falls_back_to_default_approval_set(self):
        with patch(
            "backend.runner.agent_service.build_agent_graph"
        ) as mock_build:
            mock_build.return_value = "graph"
            get_agent()
        assert (
            mock_build.call_args.kwargs["approval_required_tools"]
            is APPROVAL_REQUIRED_TOOLS
        )

    def test_empty_frozenset_is_honoured_not_swallowed(self):
        """Unattended runs pass frozenset() and must get exactly that —
        a re-armed approval gate would pause an agent no human watches."""
        with patch(
            "backend.runner.agent_service.build_agent_graph"
        ) as mock_build:
            mock_build.return_value = "graph"
            get_agent(approval_required_tools=frozenset())
        assert mock_build.call_args.kwargs["approval_required_tools"] == frozenset()

    def test_explicit_set_passes_through(self):
        from backend.agent.nodes import CHAT_APPROVAL_TOOLS

        with patch(
            "backend.runner.agent_service.build_agent_graph"
        ) as mock_build:
            mock_build.return_value = "graph"
            get_agent(approval_required_tools=CHAT_APPROVAL_TOOLS)
        assert (
            mock_build.call_args.kwargs["approval_required_tools"]
            is CHAT_APPROVAL_TOOLS
        )
