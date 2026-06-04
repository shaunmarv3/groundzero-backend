"""
config.py — GroundZero Backend Configuration
=============================================
Uses pydantic-settings to load environment variables from .env (or system env).
Import `settings` anywhere in the app:

    from config import settings
    print(settings.device)
"""

from pathlib import Path
from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration for the GroundZero backend.
    Values are loaded from the .env file (falls back to environment variables).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ------ Server -------------------------------------------
    app_host: str = Field(default="0.0.0.0", description="Uvicorn bind host")
    app_port: int = Field(default=8000, description="Uvicorn bind port")
    app_env: Literal["development", "production"] = Field(
        default="development", description="Runtime environment"
    )

    # ------ Model --------------------------------------------
    # SigLIP 2 So400m — Google's recommended production checkpoint (400M params)
    # This is what PaliGemma 2 and most modern VLMs use as their vision encoder.
    siglip2_model_id: str = Field(
        default="google/siglip2-so400m-patch14-384",
        description="HuggingFace model ID for the SigLIP 2 visual + text encoder",
    )

    hf_model_repo: str = Field(
        default="shaunmarvell/qvhighlights-model",
        description="HuggingFace Hub repo ID for trained weights",
    )
    ckpt_name: str = Field(
        default="best.pt",
        description="Checkpoint filename inside hf_model_repo / model_path dir",
    )
    model_path: str = Field(
        default="models/best.pt",
        description="Local path to the trained head checkpoint (downloaded via "
                    "scripts/download_model.py). Loaded at startup if it exists; "
                    "otherwise the app falls back to fetching ckpt_name from hf_model_repo.",
    )
    model_cache_dir: str = Field(
        default="",
        description="Local directory to cache downloaded weights. Empty = HF default.",
    )
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Inference device. 'auto' selects CUDA if available.",
    )

    # ------ Inference Config ---------------------------------
    coarse_fps: float = Field(
        default=1.0, ge=0.1, le=4.0,
        description="Frame sampling rate for coarse pass (Pass 1)",
    )
    fine_fps: float = Field(
        default=4.0, ge=1.0, le=8.0,
        description="Frame sampling rate for fine-grained re-sampling (Pass 2)",
    )
    fine_padding_sec: float = Field(
        default=5.0, ge=0.0,
        description="Seconds of padding around coarse prediction for fine re-sample",
    )
    use_coarse_to_fine: bool = Field(
        default=False,
        description="If True, run the 4fps fine refinement pass after the 1fps coarse "
                    "pass. Default False: the model was trained at 1fps and never saw "
                    "4fps frame density, so a single 1fps pass is the correctness-safe "
                    "baseline. Enable only after measuring that it actually helps.",
    )
    confidence_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Confidence below which a SOFT 'low confidence' warning is surfaced. "
                    "NOTE: the trained confidence head is uncalibrated (ECE 0.40) and "
                    "non-discriminative (48% correct below 0.4) — do NOT hard-gate on it.",
    )
    max_video_duration_sec: int = Field(
        default=7200,
        description="Maximum accepted video duration in seconds (2 hours)",
    )

    # ------ CORS ---------------------------------------------
    allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of allowed CORS origins",
    )

    # ------ W&B (optional) -----------------------------------
    wandb_api_key: str = Field(default="", description="Weights & Biases API key")
    wandb_project: str = Field(default="groundzero", description="W&B project name")
    wandb_entity: str = Field(default="", description="W&B team/entity")

    # ------ HuggingFace Auth ---------------------------------
    hf_token: str = Field(default="", description="HuggingFace access token")

    # ------ File Upload --------------------------------------
    upload_dir: str = Field(
        default="/tmp/groundzero_uploads",
        description="Temporary directory for uploaded video files",
    )
    max_upload_size_mb: int = Field(
        default=2000, description="Max upload size in megabytes"
    )

    # ------ Model Architecture (fixed, not env-overridable) --
    # Backbone: SigLIP 2 So400m (google/siglip2-so400m-patch14-384)
    # 400M params, embedding dim 1152, 27 transformer blocks
    # Reference: https://huggingface.co/google/siglip2-so400m-patch14-384
    d_model: int = 1152          # SigLIP 2 So400m hidden dim (vs 768 for ViT-L)
    n_heads: int = 8             # number of attention heads in CrossModalTransformer
    n_layers: int = 4            # number of transformer layers
    lora_rank: int = 8           # LoRA rank r
    lora_alpha: int = 16         # LoRA alpha scaling
    lora_layers: list[int] = [23, 24, 25, 26]  # last 4 blocks of SigLIP 2 So400m (27 total)

    # ------ Derived Properties --------------------------------
    @property
    def resolved_device(self) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu' based on torch availability."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    @property
    def model_full_path(self) -> Path:
        """Absolute path to the local checkpoint, resolved relative to the backend root."""
        p = Path(self.model_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        return p

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse comma-separated origins string into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def upload_path(self) -> Path:
        """Return upload_dir as a Path, creating it if it doesn't exist."""
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @field_validator("coarse_fps", "fine_fps", mode="before")
    @classmethod
    def parse_float(cls, v):
        return float(v)

    def __str__(self) -> str:
        return (
            f"GroundZero Config | env={self.app_env} | "
            f"device={self.resolved_device} | "
            f"model={self.hf_model_repo} | "
            f"coarse_fps={self.coarse_fps} | fine_fps={self.fine_fps}"
        )


# Singleton — import this everywhere
settings = Settings()
