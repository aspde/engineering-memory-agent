"""API router — aggregates all route modules."""

from fastapi import APIRouter

from backend.api.routes.agent_routes import router as agent_router
from backend.api.routes.connector_routes import router as connector_router
from backend.api.routes.entity_routes import router as entity_router
from backend.api.routes.memory_routes import router as memory_router
from backend.api.routes.patrol_routes import router as patrol_router
from backend.api.routes.webhook_routes import router as webhook_router

api_router = APIRouter()
api_router.include_router(memory_router)
api_router.include_router(agent_router)
api_router.include_router(entity_router)
api_router.include_router(webhook_router)
api_router.include_router(connector_router)
api_router.include_router(patrol_router)
