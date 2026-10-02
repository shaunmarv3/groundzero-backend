"""
orchestrator.py — End-to-end inference pipeline for the GroundZero API.
=======================================================================
Owns the single loaded model and turns (video file + query) into a grounding
prediction. This is what the FastAPI /predict route calls.

WHY A DEDICATED ORCHESTRATOR (instead of GroundZeroModel.predict)
─────────────────────────────────────────────────────────────────
The model was trained with --use_cache, so the head (TemporalContext +
CrossModalTransformer + SpanExtractionHead) was trained on FROZEN SigLIP
embeddings produced by scripts/precompute_embeddings.py. LoRA was never trained.
To reproduce training-time features at inference we must run the *frozen* SigLIP
path with LoRA at its zero-init (identity), and the forward/decode must match
scripts/evaluate.py — the script that produced the published R@1=0.5413 numbers.

This orchestrator therefore mirrors evaluate.py precisely:
  • build GroundZeroModel(use_cache=False)  → instantiates the VisualEncoder so we
    can encode raw frames; LoRA stays identity (we never load LoRA weights).
  • load_state_dict(trainable_state, strict=False)  → loads only the head.
  • model.eval()                            → CRITICAL: turns off dropout in the
    cross-modal/span/LoRA layers (predict() never did this).
  • visual encode + head forward run under autocast(fp16); head inputs are float32.
  • to_seconds(s_idx, e_idx, n_frames, full_duration)  → same index→second mapping.

Coarse-to-fine: the model only ever saw 1fps frame density. A single 1fps pass is
the correctness-safe default (settings.use_coarse_to_fine = False). The optional
4fps fine pass is gated behind that flag for experimentation.
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from app.pipeline.groundzero_model import GroundZeroModel, resolve_model_cfg
from app.pipeline.span_extraction import decode_best_span, to_seconds

logger = logging.getLogger("groundzero.orchestrator")

MODEL_ID = "google/siglip2-so400m-patch14-384"


class Orchestrator:
    """Holds the warm model and runs frozen-SigLIP grounding inference."""

    def __init__(self, model: GroundZeroModel, device: str, checkpoint: str,
                 d_model: int = 1152, encode_chunk_size: int = 8, cfg: dict | None = None):
        self.model = model
        self.device = device
        self.checkpoint = checkpoint
        self.d_model = d_model
        self.encode_chunk_size = encode_chunk_size
        self.cfg = cfg or resolve_model_cfg({})

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, settings) -> "Orchestrator":
        """
        Build the model, load the trained head, and return a ready orchestrator.

        Resolution order for the checkpoint:
          1. settings.model_full_path (local models/best.pt) if it exists
          2. else hf_hub_download(settings.hf_model_repo, settings.ckpt_name)
        """
        device = settings.resolved_device
        d_model = settings.d_model

        # ── locate checkpoint ───────────────────────────────────────────
        ckpt_path = settings.model_full_path
        if ckpt_path.exists():
            checkpoint = str(ckpt_path)
        else:
            from huggingface_hub import hf_hub_download, login
            if settings.hf_token:
                login(token=settings.hf_token)
            logger.info(f"Local checkpoint not found; pulling {settings.ckpt_name} "
                        f"from {settings.hf_model_repo} ...")
            checkpoint = hf_hub_download(
                settings.hf_model_repo, settings.ckpt_name, repo_type="model",
                cache_dir=settings.model_cache_dir or None,
            )

        logger.info(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        logger.info(f"  checkpoint epoch={ckpt.get('epoch')} "
                    f"saved R@1@0.5={ckpt.get('r1_iou05')}")
        cfg = resolve_model_cfg(ckpt)   # shipped best.pt has no model_cfg → original design
        logger.info(f"  model_cfg: {cfg}")

        # ── build model with VISION tower (use_cache=False) ─────────────
        # use_cache=False so the VisualEncoder (frozen SigLIP + identity LoRA) exists
        # to encode raw frames. The head config must match training (train.py defaults).
        logger.info(f"Building GroundZeroModel on {device} (frozen SigLIP + identity LoRA)...")
        model = GroundZeroModel(
            model_id=MODEL_ID,
            d_model=d_model,
            n_heads=settings.n_heads,
            n_layers=settings.n_layers,
            lora_rank=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            lora_layers=settings.lora_layers,
            dropout=0.1,
            device=device,
            use_cache=False,
            use_gelu=cfg["use_gelu"],
            word_level=cfg["word_level"],
        )

        # ── load ONLY the trained head; SigLIP towers stay pretrained, LoRA identity
        missing, unexpected = model.load_state_dict(ckpt["trainable_state"], strict=False)
        n_loaded = len(ckpt["trainable_state"])
        logger.info(f"  loaded {n_loaded} head tensors; {len(unexpected)} unexpected "
                    f"(should be 0), {len(missing)} missing (frozen SigLIP keys — expected).")
        if unexpected:
            logger.warning(f"  UNEXPECTED keys (possible arch mismatch): {list(unexpected)[:5]}")

        # ── eval mode: dropout OFF everywhere (cross_modal, span_head, LoRA) ──
        model.eval().to(device)

        return cls(model, device, checkpoint, d_model=d_model, cfg=cfg)

    # ------------------------------------------------------------------ #
    #  Inference                                                          #
    # ------------------------------------------------------------------ #
    def _autocast(self):
        """Match evaluate.py: forward runs under fp16 autocast on CUDA."""
        if self.device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def _encode_visual(self, pil_imgs: list) -> torch.Tensor:
        """
        Frozen-SigLIP frame embeddings, identical to precompute_embeddings.py.
        Runs encode_frames under autocast so the fp32 pixel values match the
        fp16 backbone, then returns float32 (N, d_model) for the fp32 head.
        """
        with self._autocast():
            embs = self.model.visual_encoder.encode_frames(
                pil_imgs, chunk_size=self.encode_chunk_size
            )
        return embs.float()  # head weights are fp32

    @torch.no_grad()
    def _run_pass(
        self,
        video_path: Path,
        query_emb: torch.Tensor,
        duration: float,
        fps: float,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_frames: int | None = 1024,
        query_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        One grounding pass over [start_sec, end_sec] at `fps`.

        Mirrors evaluate.py's per-sample path: frac timestamps = t/duration,
        forward under autocast, decode_best_span, to_seconds(idx, n, region_dur).
        Returns dict with start_sec/end_sec/confidence/n_frames + relevance curve.
        """
        from app.pipeline.frame_extractor import extract_frames  # lazy (needs ffmpeg)

        frames_data = extract_frames(
            video_path, fps=fps, start_sec=start_sec, end_sec=end_sec, max_frames=max_frames
        )
        if not frames_data:
            return {"start_sec": start_sec, "end_sec": end_sec or duration,
                    "confidence": 0.0, "n_frames": 0, "timestamps": [], "relevance": []}

        ts_abs = [t for t, _ in frames_data]            # absolute seconds
        imgs = [img for _, img in frames_data]
        n = len(imgs)

        frame_embs = self._encode_visual(imgs).unsqueeze(0)          # (1, N, d)
        # fractional position t/T over the FULL video (matches training/eval)
        frac = torch.tensor([t / duration for t in ts_abs],
                            dtype=torch.float32, device=self.device)  # (N,)

        # single video → no batch padding, so no frame_mask needed
        with self._autocast():
            start_logits, end_logits, confidence = self.model(
                frame_embs, frac, query_emb, query_mask=query_mask)

        sl = start_logits[0, :n]
        el = end_logits[0, :n]
        s_idx, e_idx = decode_best_span(sl, el)
        s_idx, e_idx = min(s_idx, n - 1), min(e_idx, n - 1)

        # region-local seconds: index → fraction of the extracted span, offset by start
        region_dur = (ts_abs[-1] - ts_abs[0]) + (1.0 / fps)
        rel_s, rel_e = to_seconds(s_idx, e_idx, n, region_dur)
        pred_s = ts_abs[0] + rel_s
        pred_e = ts_abs[0] + rel_e

        # per-frame relevance proxy for the heatmap: softmax(start)+softmax(end), [0,1]
        rel = (torch.softmax(sl.float(), dim=0) + torch.softmax(el.float(), dim=0)) / 2.0
        rel = (rel / rel.max().clamp_min(1e-9)).cpu().tolist()

        return {
            "start_sec": float(pred_s),
            "end_sec": float(pred_e),
            "confidence": float(confidence[0].item()),
            "n_frames": n,
            "timestamps": [float(t) for t in ts_abs],
            "relevance": rel,
        }

    @torch.no_grad()
    def predict(
        self,
        video_path: str | Path,
        query: str,
        fps: float | None = None,
        coarse_to_fine: bool | None = None,
        confidence_threshold: float = 0.4,
        fine_fps: float = 4.0,
        fine_padding_sec: float = 5.0,
    ) -> dict:
        """
        Ground `query` in the video. Returns a dict matching GroundingResult.

        fps:            coarse-pass sampling rate (default 1.0).
        coarse_to_fine: if True, refine the coarse region at fine_fps (experimental;
                        the model never saw >1fps density).
        """
        from app.pipeline.frame_extractor import get_video_metadata  # lazy

        video_path = Path(video_path)
        fps = fps or 1.0
        t0 = time.perf_counter()

        meta = get_video_metadata(video_path)
        duration = float(meta["duration"])

        query_emb, query_mask = self._encode_query(query)

        # ── Pass 1: coarse over the whole video ─────────────────────────
        coarse = self._run_pass(video_path, query_emb, duration, fps=fps, query_mask=query_mask)

        result = dict(coarse)
        coarse_meta = None

        # ── Optional Pass 2: fine refinement (experimental) ─────────────
        if coarse_to_fine and coarse["n_frames"] > 0:
            region_start = max(0.0, coarse["start_sec"] - fine_padding_sec)
            region_end = min(duration, coarse["end_sec"] + fine_padding_sec)
            fine = self._run_pass(
                video_path, query_emb, duration, fps=fine_fps,
                start_sec=region_start, end_sec=region_end, query_mask=query_mask,
            )
            if fine["n_frames"] > 0:
                coarse_meta = {"start_sec": coarse["start_sec"], "end_sec": coarse["end_sec"]}
                result = dict(fine)
                fps = fine_fps

        took_ms = (time.perf_counter() - t0) * 1000.0
        conf = result["confidence"]
        return {
            "start_sec": result["start_sec"],
            "end_sec": result["end_sec"],
            "confidence": conf,
            # The confidence head is uncalibrated (Phase 6 eval: ECE 0.40 — it emits
            # ~1e-2/1e-6 even on IoU≈0.9 spans). The handoff explicitly says do NOT
            # hard-gate on it, so `found` reflects "we produced a real span over real
            # frames", and `low_confidence` is a permanent advisory flag (the raw conf
            # is still returned above for transparency). Revisit once the head is
            # recalibrated / word-level query is added (future SOTA phase).
            "found": result["n_frames"] > 0,
            "low_confidence": True,
            "duration_sec": duration,
            "query": query,
            "fps": fps,
            "n_frames": result["n_frames"],
            "took_ms": took_ms,
            "coarse": coarse_meta,
            # extra fields (not in GroundingResult) used by the /attention route:
            "_timestamps": result["timestamps"],
            "_relevance": result["relevance"],
        }

    @torch.no_grad()
    def _encode_query(self, query: str) -> tuple:
        """
        Query → model input, the way the checkpoint was trained (cfg): lowercased if
        trained lowercase; pooled (1,1,d) or word tokens (1,L,d)+mask (word_level).
        Runs under autocast like the training-time / precomputed text embeddings.
        """
        text = query.lower() if self.cfg["lowercase"] else query
        with self._autocast():
            if self.cfg["word_level"]:
                tokens, mask = self.model.text_encoder.encode_query_tokens(text)
                return tokens.float(), mask
            emb = self.model.text_encoder.encode_query(text)
        return emb.float().unsqueeze(0), None                       # (1, 1, d)

    @torch.no_grad()
    def attention(self, video_path: str | Path, query: str,
                  fps: float | None = None) -> dict:
        """
        Per-frame query-relevance over the timeline (for the heatmap viz).
        Currently a span-score proxy derived from the start/end logits, not raw
        cross-attention weights (real attention hooks are Phase 6.5 work).
        """
        out = self.predict(video_path, query, fps=fps, coarse_to_fine=False)
        return {
            "timestamps": out["_timestamps"],
            "weights": out["_relevance"],
            "start_sec": out["start_sec"],
            "end_sec": out["end_sec"],
            "query": query,
        }

    @torch.no_grad()
    def warmup(self) -> None:
        """Run a tiny dummy forward so the first real request isn't cold."""
        try:
            dummy = torch.zeros(1, 4, self.d_model, device=self.device)
            frac = torch.linspace(0, 1, 4, device=self.device)
            q = torch.zeros(1, 1, self.d_model, device=self.device)
            q_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device) if self.cfg["word_level"] else None
            with self._autocast():
                self.model(dummy, frac, q, query_mask=q_mask)
            logger.info("Warm-up forward complete.")
        except Exception as e:  # warm-up is best-effort
            logger.warning(f"Warm-up skipped: {e}")
