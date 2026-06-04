"""
predict.py — POST /api/predict
==============================
Accepts a video file upload + a text query (as multipart form fields) and
returns the predicted (start_sec, end_sec) of the described event.

The upload is streamed to a temp file (no full-in-memory read — videos can be
hundreds of MB), grounded, then deleted. Inference itself is synchronous and
GPU-bound, so it runs in a threadpool via run_in_threadpool to avoid blocking
the event loop.
"""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from config import settings
from app.schema import GroundingResult, CoarseResult

logger = logging.getLogger("groundzero.predict")
router = APIRouter()

# extensions ffmpeg can read that we accept
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video type '{suffix}'. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )
    dest = settings.upload_path / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@router.post("/predict", response_model=GroundingResult, tags=["Prediction"])
async def predict(
    request: Request,
    video: UploadFile = File(..., description="Video file to search"),
    query: str = Form(..., min_length=1, description="Natural language event description"),
    fps: float | None = Form(None, description="Coarse sampling rate (default 1.0)"),
    coarse_to_fine: bool | None = Form(None, description="Run 4fps fine pass (experimental)"),
) -> GroundingResult:
    orc = getattr(request.app.state, "orchestrator", None)
    if orc is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try /health.")

    dest = _save_upload(video)
    try:
        result = await run_in_threadpool(
            orc.predict,
            video_path=dest,
            query=query,
            fps=fps,
            coarse_to_fine=(settings.use_coarse_to_fine if coarse_to_fine is None else coarse_to_fine),
            confidence_threshold=settings.confidence_threshold,
            fine_fps=settings.fine_fps,
            fine_padding_sec=settings.fine_padding_sec,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: keep the message; full trace is logged
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    finally:
        dest.unlink(missing_ok=True)

    coarse = result.get("coarse")
    return GroundingResult(
        start_sec=result["start_sec"],
        end_sec=result["end_sec"],
        confidence=result["confidence"],
        found=result["found"],
        low_confidence=result["low_confidence"],
        duration_sec=result["duration_sec"],
        query=result["query"],
        fps=result["fps"],
        n_frames=result["n_frames"],
        took_ms=result["took_ms"],
        coarse=CoarseResult(**coarse) if coarse else None,
    )
