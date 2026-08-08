"""Tests for agent route source/trace extraction (``_extract_tool_traces``).

Covers the traceability behaviour: read-only retrieval results surface as
*sources* (memories by id, chunks by document_id), write/other tools appear
as tool-call traces, and both source kinds deduplicate across a ReAct loop.
"""

from __future__ import annotations

import json

from langchain_core.messages import ToolMessage

from backend.api.routes.agent_routes import _extract_tool_traces


def _tool_message(name: str, content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id="c1", name=name)


class TestExtractToolTraces:
    def test_memory_sources_and_dedup_by_id(self) -> None:
        env = json.dumps(
            {
                "display": "Found 1 relevant memories:",
                "sources": [
                    {"id": "mem-001", "type": "memory", "summary": "A", "relevance": 0.9}
                ],
            },
            ensure_ascii=False,
        )
        traces, sources = _extract_tool_traces(
            [
                _tool_message("search_memories_tool", env),
                _tool_message("search_memories_tool", env),  # same memory twice
            ]
        )
        assert traces == []
        assert len(sources) == 1
        assert sources[0]["id"] == "mem-001"

    def test_chunk_sources_surface_and_dedup_by_document_chunk(self) -> None:
        env = json.dumps(
            {
                "display": "Found 1 relevant chunks:",
                "sources": [
                    {
                        "type": "chunk",
                        "document_id": "docs/architecture.md",
                        "chunk_index": 2,
                        "snippet": "EMA uses pgvector",
                        "relevance": 0.8,
                    }
                ],
            },
            ensure_ascii=False,
        )
        traces, sources = _extract_tool_traces(
            [
                _tool_message("retrieve_chunks_tool", env),
                _tool_message("retrieve_chunks_tool", env),  # same chunk twice
            ]
        )
        # Chunk retrieval is read-only — no tool-call trace.
        assert traces == []
        assert len(sources) == 1
        assert sources[0]["type"] == "chunk"
        assert sources[0]["document_id"] == "docs/architecture.md"
        assert sources[0]["chunk_index"] == 2

    def test_distinct_chunks_are_kept(self) -> None:
        def _env(idx: int) -> str:
            return json.dumps(
                {
                    "display": "Found 1 relevant chunks:",
                    "sources": [
                        {
                            "type": "chunk",
                            "document_id": "docs/architecture.md",
                            "chunk_index": idx,
                            "snippet": f"chunk {idx}",
                            "relevance": 0.8,
                        }
                    ],
                },
                ensure_ascii=False,
            )

        _, sources = _extract_tool_traces(
            [
                _tool_message("retrieve_chunks_tool", _env(2)),
                _tool_message("retrieve_chunks_tool", _env(5)),
            ]
        )
        assert len(sources) == 2

    def test_query_rewrite_is_read_only_source(self) -> None:
        env = json.dumps(
            {
                "display": "Found 1 relevant chunks (query rewritten):",
                "sources": [
                    {
                        "type": "chunk",
                        "document_id": "postmortem.md",
                        "chunk_index": 0,
                        "snippet": "ctx 泄漏",
                        "relevance": 0.7,
                    }
                ],
            },
            ensure_ascii=False,
        )
        traces, sources = _extract_tool_traces(
            [_tool_message("query_rewrite_and_search_tool", env)]
        )
        assert traces == []
        assert len(sources) == 1
        assert sources[0]["document_id"] == "postmortem.md"

    def test_write_tools_appear_as_traces(self) -> None:
        traces, sources = _extract_tool_traces(
            [_tool_message("ingest_git_repo_tool", "Ingested 3 commits")]
        )
        assert len(traces) == 1
        assert traces[0]["tool"] == "ingest_git_repo_tool"

    def test_silent_tools_are_excluded(self) -> None:
        traces, sources = _extract_tool_traces(
            [_tool_message("write_memory_tool", "wrote memory 1")]
        )
        assert traces == []
        assert sources == []
