"""Tests for AgentState TypedDict schema."""

from langchain_core.messages import HumanMessage, AIMessage

from agent.state import AgentState


class TestAgentState:
    def test_initial_state_keys(self) -> None:
        state = AgentState(
            messages=[],
            final_response=None,
            error=None,
        )
        assert "messages" in state
        assert "final_response" in state
        assert "error" in state

    def test_messages_field_holds_base_messages(self) -> None:
        """messages field accepts list of BaseMessage."""
        msg = HumanMessage(content="hello")
        state = AgentState(
            messages=[msg],
            final_response=None,
            error=None,
        )
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "hello"

    def test_final_response_can_be_set(self) -> None:
        state = AgentState(
            messages=[],
            final_response="Here is the answer.",
            error=None,
        )
        assert state["final_response"] == "Here is the answer."

    def test_error_state(self) -> None:
        state = AgentState(
            messages=[],
            final_response=None,
            error="LLM call failed",
        )
        assert state["error"] == "LLM call failed"

    def test_pending_approval_defaults_to_none(self) -> None:
        state = AgentState(
            messages=[],
            final_response=None,
            error=None,
            pending_approval=None,
        )
        assert state["pending_approval"] is None

    def test_pending_approval_holds_approval_data(self) -> None:
        data = {"tool_name": "write_memory_tool", "tool_args": {"content": "test"}}
        state = AgentState(
            messages=[],
            final_response=None,
            error=None,
            pending_approval=data,
        )
        assert state["pending_approval"] == data
