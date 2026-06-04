"""
schemas.py — Pydantic request/response models for the GroundZero API.
======================================================================
These define the contract the Next.js frontend codes against. The video itself
is sent as a multipart file upload (FastAPI `UploadFile`), so the request model
here only carries the query + optional inference knobs that ride alongside it as
form fields.

Design notes baked in from Phase 6 eval findings:
  - `confidence` is RETURNED but must not be used as a hard "event not found"
    gate: the trained confidence head is uncalibrated (ECE 0.40) and only 48%
    accurate below 0.4. `found` is therefore advisory; `low_confidence` is a
    soft UI warning, not a rejection.
"""

from pydantic import BaseModel, Field


class PredictOptions(BaseModel):
    """Optional inference knobs sent as form fields next to the uploaded video."""

    query: str = Field(..., min_length=1, description="Natural language description of the event to find")
    fps: float | None = Field(
        default=None, ge=0.1, le=4.0,
        description="Coarse-pass sampling rate. Default: server's coarse_fps (1.0). "
                    "The model was trained at 1fps — changing this is experimental.",
    )
    coarse_to_fine: bool | None = Field(
        default=None,
        description="Override the server default for the 4fps fine refinement pass. "
                    "Default (None) uses settings.use_coarse_to_fine (False).",
    )


class CoarseResult(BaseModel):
    """The Pass-1 (1fps) prediction, exposed for debugging / comparison."""

    start_sec: float
    end_sec: float


class GroundingResult(BaseModel):
    """The grounding prediction returned by POST /api/predict."""

    start_sec: float = Field(..., description="Predicted event start, in seconds")
    end_sec: float = Field(..., description="Predicted event end, in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Event-presence score in [0,1] (UNCALIBRATED — advisory only)")
    found: bool = Field(..., description="confidence >= server threshold (advisory; not a hard gate)")
    low_confidence: bool = Field(..., description="Soft UI warning flag: confidence below threshold")

    duration_sec: float = Field(..., description="Total video duration in seconds")
    query: str = Field(..., description="Echo of the query that was grounded")
    fps: float = Field(..., description="Sampling rate used for the prediction")
    n_frames: int = Field(..., description="Number of frames the prediction was made over")
    took_ms: float = Field(..., description="End-to-end inference wall time in milliseconds")

    coarse: CoarseResult | None = Field(
        default=None,
        description="Coarse (1fps) result; present only when coarse-to-fine refinement ran",
    )


class AttentionResponse(BaseModel):
    """Cross-modal attention over the video timeline (for the heatmap viz)."""

    timestamps: list[float] = Field(..., description="Frame timestamps in seconds (x-axis)")
    weights: list[float] = Field(..., description="Per-frame query-relevance weight in [0,1] (y-axis)")
    start_sec: float = Field(..., description="Predicted span start")
    end_sec: float = Field(..., description="Predicted span end")
    query: str


class HealthResponse(BaseModel):
    """GET /health payload."""

    status: str = Field(..., description="'ok' once the model is loaded and warm")
    model_loaded: bool
    device: str
    checkpoint: str = Field(..., description="Resolved checkpoint path or HF repo id")
