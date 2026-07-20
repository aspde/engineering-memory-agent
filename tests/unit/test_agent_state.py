"""Tests for AgentState TypedDict schema."""

from langchain_core.messages import HumanMessage, AIMessage

from agent.state import AgentState


class TestAgentState:
    def test_initial_state_keys(self) -> None:
        state = AgentState(
            messages=[],
            retrieved_chunks=[],
            retrieved_memories=[],
            context_assembled="",
            final_response=None,
            error=None,
        )
        assert "messages" in state
        assert "retrieved_chunks" in state
        assert "retrieved_memories" in state
        assert "context_assembled" in state
        assert "final_response" in state
        assert "error" in state

    def test_messages_field_holds_base_messages(self) -> None:
        """messages field accepts list of BaseMessage."""
        msg = HumanMessage(content="hello")
        state = AgentState(
            messages=[msg],
            retrieved_chunks=[],
            retrieved_memories=[],
            context_assembled="",
            final_response=None,
            error=None,
        )
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "hello"

    def test_state_accepts_retrieved_data(self) -> None:
        state = AgentState(
            messages=[],
            retrieved_chunks=[{"content": "chunk", "score": 0.9}],
            retrieved_memories=[{"summary": "memory", "weighted_score": 0.85}],
            context_assembled="",
            final_response=None,
            error=None,
        )
        assert len(state["retrieved_chunks"]) == 1
        assert len(state["retrieved_memories"]) == 1

    def test_final_response_can_be_set(self) -> None:
        state = AgentState(
            messages=[],
            retrieved_chunks=[],
            retrieved_memories=[],
            context_assembled="",
            final_response="Here is the answer.",
            error=None,
        )
        assert state["final_response"] == "Here is the answer."

    def test_error_state(self) -> None:
        state = AgentState(
            messages=[],
            retrieved_chunks=[],
            retrieved_memories=[],
            context_assembled="",
            final_response=None,
            error="LLM call failed",
        )
        assert state["error"] == "LLM call failed"
