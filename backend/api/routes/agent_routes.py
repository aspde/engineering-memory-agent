"""Agent chat API routes."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from backend.service.agent_service import get_agent_for_thread

router = APIRouter(prefix="/agent", tags=["agent"])


# ── Request / Response models ────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    thread_id: str = Field(default_factory=lambda: str(uuid4()))


class ChatResponse(BaseModel):
    thread_id: str
    response: str
    tool_calls: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)


# ── Routes ───────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the EMA agent and receive a response.

    The agent autonomously decides which tools to call (memory search,
    document retrieval, ingestion, etc.) based on the message content.

    Provide a *thread_id* to continue an existing conversation.
    """
    agent = get_agent_for_thread()
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    final_response: str = result.get("final_response", "") or ""
    if not final_response:
        # Fallback: extract last AIMessage content if generate_final didn't fire
        for m in reversed(result.get("messages", [])):
            if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None):
                final_response = str(m.content)
                break

    # Extract tool call traces for transparency
    tool_call_traces: list[dict] = []
    for m in result.get("messages", []):
        if isinstance(m, ToolMessage):
            tool_call_traces.append({
                "tool": getattr(m, "name", "unknown"),
                "content": str(m.content)[:300],
            })

    # Collect sources
    sources: list[dict] = []
    for chunk in result.get("retrieved_chunks", []):
        sources.append({
            "type": "chunk",
            "content": str(chunk.get("content", ""))[:200],
            "score": chunk.get("score", chunk.get("rerank_score", 0)),
        })
    for mem in result.get("retrieved_memories", []):
        sources.append({
            "type": "memory",
            "content": str(mem.get("summary", ""))[:200],
            "score": mem.get("weighted_score", mem.get("rerank_score", 0)),
        })

    return ChatResponse(
        thread_id=req.thread_id,
        response=final_response,
        tool_calls=tool_call_traces,
        sources=sources,
    )
