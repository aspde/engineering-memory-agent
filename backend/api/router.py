"""API router — aggregates all route modules."""

from fastapi import APIRouter, Depends

from backend.api.auth import require_api_key
from backend.api.routes.agent_routes import router as agent_router
from backend.api.routes.conflict_routes import router as conflict_router
from backend.api.routes.connector_routes import router as connector_router
from backend.api.routes.entity_routes import router as entity_router
from backend.api.routes.memory_routes import router as memory_router
from backend.api.routes.patrol_routes import router as patrol_router
from backend.api.routes.scenario_routes import router as scenario_router
from backend.api.routes.usage_routes import router as usage_router
from backend.api.routes.webhook_routes import router as webhook_router
from backend.shared.config import config

# Router-level dependency: every route aggregated below (and, transitively,
# every route of every included sub-router) requires a valid Bearer API key.
# Bypassed in APP_ENV=test — see backend/api/auth.py.
api_router = APIRouter(dependencies=[Depends(require_api_key)])

# ── Core loop — always mounted ─────────────────────────────────────────
# chat + memory read/write + entity graph + conflict arbitration + LLM
# usage observability.  These are EMA's primary surface.
api_router.include_router(memory_router)
api_router.include_router(agent_router)
api_router.include_router(entity_router)
api_router.include_router(conflict_router)
api_router.include_router(usage_router)

# ── Breadth layers — off by default (ADR-011) ─────────────────────────
# Connectors/webhooks, vertical scenarios and the patrol scheduler serve
# data sources EMA does not have yet.  Their routes are only mounted when
# the corresponding flag is set; ``*_active`` folds in the ``APP_ENV=test``
# exemption so the API suite keeps exercising them.  Unmounted routers
# return 404 — the entry point does not exist, rather than existing but
# refusing (see backend/shared/config.py).
if config.connectors_active:
    api_router.include_router(webhook_router)
    api_router.include_router(connector_router)
if config.scenarios_active:
    api_router.include_router(scenario_router)
if config.patrol_active:
    api_router.include_router(patrol_router)
