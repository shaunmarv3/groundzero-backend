"""
health.py — GET /health
=======================
Liveness + readiness check. Reports whether the model finished loading and on
which device, so the frontend (and any deploy probe) can tell a cold/broken
backend from a warm one.
"""

from fastapi import APIRouter, Request

from config import settings
from app.schema import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health(request: Request) -> HealthResponse:
    orc = getattr(request.app.state, "orchestrator", None)
    loaded = orc is not None
    return HealthResponse(
        status="ok" if loaded else "loading",
        model_loaded=loaded,
        device=orc.device if loaded else settings.resolved_device,
        checkpoint=orc.checkpoint if loaded else settings.model_path,
    )
