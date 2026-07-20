"""API router — aggregates all route modules."""

from fastapi import APIRouter

from backend.api.routes.agent_routes import router as agent_router
from backend.api.routes.memory_routes import router as memory_router

api_router = APIRouter()
api_router.include_router(memory_router)
api_router.include_router(agent_router)
