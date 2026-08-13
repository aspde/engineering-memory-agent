"""Scenario API routes — list available scenarios and trigger runs."""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.scenarios import (
    SCENARIOS,
    _release_scenario_slot,
    _try_acquire_scenario_slot,
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

    The scenario's compose function is dynamically imported and called.
    Returns 404 if the scenario is not found, 400 if it is inactive, 503 if
    the scenario-run concurrency cap is reached, 504 if the run exceeds its
    deadline, 422 on invalid parameters, and 500 on failure.
    """
    if body is None:
        body = ScenarioRunRequest()
    scenario = SCENARIOS.get(name)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{name}' not found")

    if scenario.get("status") == "inactive":
        raise HTTPException(
            status_code=400, detail=f"Scenario '{name}' is inactive"
        )

    # Concurrency cap: beyond MAX_SCENARIO_CONCURRENCY simultaneous scenario
    # runs the request is refused (503), not queued — each run invokes the
    # full agent (recursion_limit=50) for up to SCENARIO_TIMEOUT_SECONDS, so
    # an unbounded pile-up would saturate the provider rate limit and can
    # only be stopped by a restart.  Checked before any import/DB setup so a
    # refused request costs nothing.
    if not _try_acquire_scenario_slot():
        logger.warning(
            "run_scenario refused — concurrency cap reached (max=%d) scenario=%s",
            config.max_scenario_concurrency, name,
        )
        raise HTTPException(
            status_code=503,
            detail="系统繁忙，同时运行的场景数已达上限，请稍后重试。",
        )

    try:
        compose_path = scenario["compose"]
        try:
            module_path, func_name = compose_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            compose_func = getattr(module, func_name)
        except (ImportError, AttributeError) as exc:
            logger.exception("Failed to load compose function for scenario '%s'", name)
            raise HTTPException(
                status_code=500,
                detail=f"Scenario '{name}' compose function not loadable: {exc}",
            ) from exc

        # Persist the client thread_id through the compose chain via ContextVar.
        from backend.service.scenarios import scenario_thread_id

        tid = body.thread_id or ""
        if tid:
            scenario_thread_id.set(tid)
            # Create the conversation record immediately so the thread appears
            # in the history sidebar before the (potentially slow) scenario completes.
            label = scenario.get("name", name)
            try:
                session_factory = get_session_factory()
                async with session_factory() as session:
                    await session.execute(
                        text(
                            "INSERT INTO conversations (thread_id, title, updated_at) "
                            "VALUES (:tid, :title, now()) "
                            "ON CONFLICT (thread_id) DO UPDATE SET "
                            "title = :title, "
                            "updated_at = now()"
                        ),
                        {"tid": tid, "title": label},
                    )
                    await session.commit()
            except Exception:
                logger.warning("Failed to persist scenario conversation", exc_info=True)

        try:
            # Bound the whole compose chain (agent ReAct loop + formatting)
            # so a stuck scenario returns 504 instead of hanging the request
            # forever.
            async with asyncio.timeout(config.scenario_timeout):
                result = await compose_func(**body.params)
        except TimeoutError:
            logger.warning(
                "Scenario '%s' timed out after %ds",
                name, config.scenario_timeout,
            )
            raise HTTPException(
                status_code=504,
                detail=(
                    f"场景执行超时（超过 {config.scenario_timeout} 秒），已停止本轮处理，"
                    "请稍后重试。"
                ),
            ) from None
        except TypeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid parameters for scenario '{name}': {exc}",
            ) from exc
        except Exception as exc:
            logger.exception("Scenario '%s' execution failed", name)
            raise HTTPException(
                status_code=500,
                detail=f"Scenario '{name}' failed: {exc}",
            ) from exc

        # compose functions return a string (the formatted scenario output)
        return ScenarioRunResponse(
            scenario=name,
            status="completed",
            result=str(result),
        )
    finally:
        # Release the concurrency slot on every exit path — success, timeout,
        # 422/500 during setup, or an internal error.
        _release_scenario_slot()
