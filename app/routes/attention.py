"""
attention.py — POST /api/attention
===================================
Returns a per-frame query-relevance curve over the video timeline (timestamps +
weights in [0,1]) plus the predicted span, for the frontend heatmap.

NOTE: the weights are currently a span-score proxy derived from the start/end
logits, not raw cross-attention weights. Real cross-modal attention extraction
(forward hooks into CrossModalTransformer) is Phase 6.5 work; the response shape
is stable so the frontend won't need to change when it's upgraded.
"""

import logging
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool

from config import settings
from app.schema import AttentionResponse

logger = logging.getLogger("groundzero.attention")
router = APIRouter()

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "video.mp4").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported video type '{suffix}'.")
    dest = settings.upload_path / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


@router.post("/attention", response_model=AttentionResponse, tags=["Attention"])
async def attention(
    request: Request,
    video: UploadFile = File(...),
    query: str = Form(..., min_length=1),
    fps: float | None = Form(None),
) -> AttentionResponse:
    orc = getattr(request.app.state, "orchestrator", None)
    if orc is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try /health.")

    dest = _save_upload(video)
    try:
        out = await run_in_threadpool(orc.attention, video_path=dest, query=query, fps=fps)
    except Exception as e:
        logger.exception("attention failed")
        raise HTTPException(status_code=500, detail=f"Attention failed: {e}")
    finally:
        dest.unlink(missing_ok=True)

    return AttentionResponse(**out)
