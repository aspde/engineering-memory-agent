"""Scenario API routes — list available scenarios and trigger runs."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.api import conversations
from backend.runner.scenarios import (
    SCENARIOS,
    ScenarioBusyError,
    ScenarioErrorKind,
    execute_scenario,
    visible_scenarios,
)
from backend.shared.config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


# ── Request / Response models ──────────────────────────────────────────


class ScenarioInfo(BaseModel):
    key: str
    name: str
    description: str
    triggers: list[str]
    status: str


class ScenarioRunRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = Field(default=None, description="Client-side thread ID for persistence")


class ScenarioRunResponse(BaseModel):
    scenario: str
    status: str
    result: str = ""
    run_id: str = ""


# ── Routes ─────────────────────────────────────────────────────────────


@router.get("", response_model=list[ScenarioInfo])
async def list_scenarios():
    """Return active scenarios visible to the UI.  Beta scenarios are hidden by default."""
    visible = visible_scenarios(include_beta=False)
    return [
        ScenarioInfo(
            key=key,
            name=info["name"],
            description=info.get("description", ""),
            triggers=info.get("triggers", []),
            status=info.get("status", "active"),
        )
        for key, info in visible.items()
    ]


@router.post("/{name}/run", response_model=ScenarioRunResponse)
async def run_scenario(name: str, body: ScenarioRunRequest | None = None):
    """Trigger a scenario by name with optional *params*.

    Execution and ``scenario_runs`` persistence live in
    :func:`execute_scenario`; this route only maps its outcomes onto HTTP
    statuses — 404 unknown/inactive, 503 concurrency cap (refused before
    any row is written), 504 deadline, 422 invalid parameters, and the
    recorded failure otherwise.
    """
    if body is None:
        body = ScenarioRunRequest()
    if name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{name}' not found")
    if SCENARIOS[name].get("status") == "inactive":
        raise HTTPException(status_code=400, detail=f"Scenario '{name}' is inactive")

    # Concurrency cap: beyond MAX_SCENARIO_CONCURRENCY simultaneous scenario
    # runs the request is refused (503), not queued — each run invokes the
    # full agent (recursion_limit=50) for up to SCENARIO_TIMEOUT_SECONDS, so
    # an unbounded pile-up would saturate the provider rate limit and can
    # only be stopped by a restart.  Checked before any import/DB setup so a
    # refused request costs nothing.
    tid = body.thread_id or ""
    if tid:
        # Create the conversation row before the (potentially slow) run so
        # the thread is in the history sidebar while it executes — the
        # behaviour the pre-execute_scenario route had and the sidebar
        # regression (2026-09 review) lost.
        await conversations.upsert_conversation(
            tid, SCENARIOS[name].get("name", name)
        )
    try:
        outcome = await execute_scenario(
            name,
            params=body.params,
            trigger="manual",
            thread_id=tid,
        )
    except ScenarioBusyError:
        raise HTTPException(
            status_code=503,
            detail="系统繁忙，同时运行的场景数已达上限，请稍后重试。",
        ) from None
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Scenario '{name}' not found") from None

    if outcome["status"] == "failed":
        error = outcome.get("error") or ""
        kind = outcome.get("error_kind")
        if kind == ScenarioErrorKind.TIMEOUT:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"场景执行超时（超过 {config.scenario_timeout} 秒），已停止本轮处理，"
                    "请稍后重试。"
                ),
            )
        if kind == ScenarioErrorKind.INVALID_PARAMS:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid parameters for scenario '{name}': {error}",
            )
        logger.error(
            "Scenario '%s' failed (run %s): %s", name, outcome["run_id"], error
        )
        raise HTTPException(
            status_code=500,
            detail=f"Scenario '{name}' failed: {error}",
        )

    return ScenarioRunResponse(
        scenario=name,
        status="completed",
        result=outcome["result"],
        run_id=outcome["run_id"],
    )


@router.get("/runs/{run_id}")
async def get_scenario_run(run_id: str):
    """Fetch one scenario run row (findings + contract state for save-as-memory)."""
    from sqlalchemy import text

    from backend.db import get_session_factory

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT id::text, scenario_key, trigger, status, params, "
                "result_md, findings, contract_ok, error, created_at, completed_at "
                "FROM scenario_runs WHERE id = CAST(:id AS UUID)"
            ),
            {"id": run_id},
        )
        row = result.mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return dict(row)


@router.post("/runs/{run_id}/save-as-memory")
async def save_run_as_memory(run_id: str):
    """Persist a completed postmortem run into the memory store.

    The user clicking the button *is* the confirmation (same rationale as
    the chat force-write path skipping re-approval), so this writes
    directly via ``write_memory`` — no agent loop, no approval gate.  A
    similarity conflict is deferred to the non-interactive pending-
    conflicts queue instead of pausing mid-request; the caller polls
    ConflictsPage like every other connector-sourced conflict.
    """
    import json as _json

    from sqlalchemy import text

    from backend.db import get_session_factory
    from backend.service.conflicts import persist_pending_conflict
    from backend.service.ingestion.entity import link_memory_to_entities
    from backend.service.memory import write_memory

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT scenario_key, status, result_md, findings "
                "FROM scenario_runs WHERE id = CAST(:id AS UUID)"
            ),
            {"id": run_id},
        )
        row = result.mappings().first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    if row["scenario_key"] != "postmortem":
        raise HTTPException(
            status_code=400,
            detail=f"Save-as-memory is not supported for '{row['scenario_key']}'",
        )
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail="运行尚未完成或已失败，无法保存")

    # Idempotency: a run already saved once resolves to that memory again
    # instead of writing a duplicate postmortem on a double click.
    async with session_factory() as session:
        existing = await session.execute(
            text(
                "SELECT id::text, meta FROM memories "
                "WHERE source_type = 'postmortem' AND deleted_at IS NULL "
                "AND meta->>'scenario_run_id' = :rid "
                "LIMIT 1"
            ),
            {"rid": run_id},
        )
        prior = existing.mappings().first()
    if prior is not None:
        prior_meta = prior["meta"] if isinstance(prior["meta"], dict) else {}
        return {
            "action": str(prior_meta.get("save_action", "inserted")),
            "memory_id": str(prior["id"]),
            "summary": "",
            "duplicate": True,
        }

    content = build_save_content(row["result_md"] or "", row["findings"])
    try:
        write_result = await write_memory(
            content,
            source_type="postmortem",
            metadata={"scenario_run_id": run_id},
        )
    except Exception as exc:
        logger.exception("save-as-memory failed for run %s", run_id)
        raise HTTPException(status_code=500, detail=f"写入记忆失败: {exc}") from exc

    action = write_result.get("action")
    if action == "conflict":
        conflict = await persist_pending_conflict("postmortem", write_result)
        return {
            "action": "conflict",
            "memory_id": None,
            "conflict_id": conflict.get("id"),
            "existing_summary": write_result.get("existing_summary", ""),
        }

    memory_id = str(write_result.get("id", ""))
    # Story 5: the saved postmortem is linked to the entities the draft's
    # contract resolved (exact canonical-name match — see
    # link_memory_to_entities), so entity-graph queries surface it like any
    # other incident memory.  Contract-miss saves have no structured entity
    # list; their names stay in the content only.
    if memory_id and isinstance(row["findings"], dict):
        entity_names = [
            str(e.get("name", "")).strip()
            for e in row["findings"].get("related_entities") or []
            if isinstance(e, dict) and str(e.get("name", "")).strip()
        ]
        if entity_names:
            await link_memory_to_entities(memory_id, entity_names)

    # Record how the save landed so a repeated click reports the original
    # action rather than a fresh duplicate probe.
    async with session_factory() as session:
        await session.execute(
            text(
                "UPDATE memories SET meta = meta || CAST(:patch AS JSONB) "
                "WHERE id = CAST(:mid AS UUID)"
            ),
            {
                "mid": memory_id,
                "patch": _json.dumps({"save_action": action}, ensure_ascii=False),
            },
        )
        await session.commit()

    return {
        "action": action,
        "memory_id": memory_id,
        "summary": write_result.get("summary", ""),
    }


def build_save_content(result_md: str, findings: dict[str, Any] | None) -> str:
    """Assemble the text handed to ``write_memory`` from a run's outputs.

    Structured findings win when present (the distilled contract form);
    otherwise the raw markdown stands in so a contract-missed draft can
    still be saved deliberately by the user.
    """
    import json as _json

    if isinstance(findings, dict) and findings:
        lines = ["故障复盘报告：", findings.get("overview", "")]
        timeline = findings.get("timeline")
        if timeline:
            lines.append("时间线：" + _json.dumps(timeline, ensure_ascii=False))
        root_cause = findings.get("root_cause")
        if root_cause:
            lines.append(f"根因分析：{root_cause}")
        recommendations = findings.get("recommendations")
        if recommendations:
            lines.append(
                "改进建议：" + _json.dumps(recommendations, ensure_ascii=False)
            )
        similar = findings.get("similar_incidents")
        if similar:
            lines.append(
                "相似故障：" + _json.dumps(similar, ensure_ascii=False)
            )
        entities = findings.get("related_entities")
        if entities:
            lines.append(
                "关联实体：" + _json.dumps(entities, ensure_ascii=False)
            )
        return "\n".join(line for line in lines if line)
    header = "故障复盘报告（草稿，未经结构化解析）：\n" if result_md else ""
    return header + result_md
