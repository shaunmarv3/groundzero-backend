"""
main.py — GroundZero FastAPI Application Entry Point
=====================================================
Initialises the FastAPI app, registers middleware, and mounts all routers.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings

# ------------------------------------------------------------------ #
#  Logging setup                                                       #
# ------------------------------------------------------------------ #
logging.basicConfig(
    level=logging.INFO if settings.app_env == "production" else logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("groundzero")


# ------------------------------------------------------------------ #
#  Lifespan — startup / shutdown logic                                 #
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup (before first request) and once at shutdown.
    Use this to load the model into memory so /predict doesn't cold-start.
    """
    # -------- STARTUP --------
    logger.info("Starting GroundZero backend...")
    logger.info(str(settings))

    # Ensure upload directory exists
    settings.upload_path  # property creates dir if needed
    logger.info(f"Upload directory ready: {settings.upload_dir}")

    # Load the trained model + warm it up so the first /predict isn't cold.
    # Wrapped in try/except: if the model fails to load (e.g. missing checkpoint),
    # the app still starts and /predict returns 503 instead of crashing the server.
    app.state.orchestrator = None
    try:
        from app.pipeline.orchestrator import Orchestrator
        app.state.orchestrator = Orchestrator.load(settings)
        app.state.orchestrator.warmup()
        logger.info(f"Model loaded on {app.state.orchestrator.device} ✓")
    except Exception as e:
        logger.error(f"Model failed to load — /predict will return 503. Reason: {e}")

    logger.info("GroundZero backend ready ✓")
    yield

    # -------- SHUTDOWN --------
    logger.info("Shutting down GroundZero backend...")
    # Clean up model from memory if needed
    if getattr(app.state, "orchestrator", None) is not None:
        del app.state.orchestrator
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    logger.info("Shutdown complete.")


# ------------------------------------------------------------------ #
#  App instance                                                        #
# ------------------------------------------------------------------ #
app = FastAPI(
    title="GroundZero API",
    description=(
        "Temporal Video Grounding from Natural Language. "
        "Given a video and a text query, returns the precise start/end timestamps "
        "of the described event."
    ),
    version="0.1.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
#  Middleware                                                           #
# ------------------------------------------------------------------ #

# CORS — allow frontend origins defined in .env
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request size limiter (reject oversized uploads early)
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_upload_size_bytes:
        return JSONResponse(
            status_code=413,
            content={
                "error": "Payload too large",
                "detail": (
                    f"Max upload size is {settings.max_upload_size_mb} MB. "
                    f"Received {int(content_length) // (1024*1024)} MB."
                ),
            },
        )
    return await call_next(request)


# ------------------------------------------------------------------ #
#  Routers                                                             #
# ------------------------------------------------------------------ #
# Routes are imported lazily so the app starts even if ML deps are
# not yet installed (useful during early dev / setup).
try:
    from app.routes.health import router as health_router
    app.include_router(health_router, tags=["Health"])
except ImportError as e:
    logger.warning(f"Could not load health router: {e}")

try:
    from app.routes.predict import router as predict_router
    app.include_router(predict_router, prefix="/api", tags=["Prediction"])
except ImportError as e:
    logger.warning(f"Could not load predict router: {e}")

try:
    from app.routes.attention import router as attention_router
    app.include_router(attention_router, prefix="/api", tags=["Attention"])
except ImportError as e:
    logger.warning(f"Could not load attention router: {e}")


# ------------------------------------------------------------------ #
#  Root                                                                #
# ------------------------------------------------------------------ #
@app.get("/", tags=["Root"])
async def root():
    """Quick sanity check — confirms the API is reachable."""
    return {
        "service": "GroundZero API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }
