"""
groundzero_model.py — Full GroundZero Model Assembly.
Phase 3.7 (Chunk H).

Wires all 5 components into one nn.Module:

    frames ──► VisualEncoder ──► TemporalContext ──► CrossModalTransformer ──► SpanHead
    query  ──► TextEncoder   ──────────────────────────────────────────────↗

Two entry points:
    forward()  — used during training (expects pre-processed tensors)
    predict()  — used during inference (accepts raw video path + query string,
                 runs coarse-to-fine internally)
"""

import torch
import torch.nn as nn
from pathlib import Path

from app.pipeline.visual_encoder import VisualEncoder
from app.pipeline.text_encoder import TextEncoder
from app.pipeline.temporal_context import TemporalContextModule
from app.pipeline.cross_modal_transformer import CrossModalTransformer
from app.pipeline.span_extraction import SpanExtractionHead, decode_best_span, to_seconds
# frame_extractor requires ffmpeg — only needed for inference (predict/run_pass), not training
# imported lazily inside predict() so training environments without ffmpeg still work


class GroundZeroModel(nn.Module):
    """
    Full temporal video grounding model.

    Trainable params:
        LoRA adapters      (VisualEncoder)         ~0.147M
        Temporal conv      (TemporalContextModule)  ~10M
        Cross-modal attn   (CrossModalTransformer)  ~127.5M
        Span head          (SpanExtractionHead)     ~0.665M
        Total trainable:                            ~138M

    Frozen params:
        SigLIP 2 backbone  (Visual + Text towers)   ~1136M
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch14-384",
        d_model: int = 1152,
        n_heads: int = 8,
        n_layers: int = 4,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_layers: list = [23, 24, 25, 26],
        dropout: float = 0.1,
        device: str = "cuda",
        use_cache: bool = False,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.use_cache = use_cache

        if use_cache:
            # Cached-feature training (Phase 2): frame embeddings are pre-extracted
            # on disk, so the vision tower + LoRA are not needed at all — skipping
            # VisualEncoder saves ~1.75 GB VRAM and all the per-step vision compute.
            # Only the frozen SigLIP text tower is needed, for encoding queries.
            self.visual_encoder = None
            self.text_encoder   = TextEncoder(model_id, device)
        else:
            self.visual_encoder = VisualEncoder(model_id, lora_rank, lora_alpha, lora_layers, device)
            # Share the already-loaded SigLIP 2 backbone — avoids loading 1.1 GB twice
            self.text_encoder   = TextEncoder(model_id, device, shared_model=self.visual_encoder.model)

        self.temporal_context = TemporalContextModule(d_model).to(device)
        self.cross_modal      = CrossModalTransformer(d_model, n_heads, n_layers, dropout).to(device)
        self.span_head        = SpanExtractionHead(d_model, dropout).to(device)

    def forward(
        self,
        frames: torch.Tensor,
        timestamps: torch.Tensor,
        query_emb: torch.Tensor,
    ) -> tuple:
        """
        Training forward pass — accepts pre-encoded inputs.

        Args:
            frames:     (B, N, d_model) — frame embeddings from VisualEncoder
            timestamps: (N,)            — fractional positions t/T in [0, 1]
            query_emb:  (B, 1, d_model) — query embedding from TextEncoder

        Returns:
            start_logits: (B, N)
            end_logits:   (B, N)
            confidence:   (B,)
        """
        x = self.temporal_context(frames, timestamps)        # (B, N, d_model)
        x = self.cross_modal(x, query_emb)                   # (B, N, d_model)
        start_logits, end_logits, confidence = self.span_head(x)
        return start_logits, end_logits, confidence

    @torch.no_grad()
    def predict(
        self,
        video_path: str | Path,
        query: str,
        fps_coarse: float = 1.0,
        fps_fine: float = 4.0,
        fine_window_sec: float = 5.0,
        confidence_threshold: float = 0.4,
    ) -> dict:
        """
        Full coarse-to-fine inference on a raw video + text query.

        Pass 1 (coarse): 1fps over full video → rough region
        Pass 2 (fine):   4fps over predicted region ±fine_window_sec → precise boundaries

        Args:
            video_path:            path to video file
            query:                 natural language query string
            fps_coarse:            frame rate for coarse pass (default 1fps)
            fps_fine:              frame rate for fine pass (default 4fps)
            fine_window_sec:       buffer around coarse prediction for fine pass
            confidence_threshold:  below this, event is considered absent

        Returns:
            dict with keys:
                start_sec    (float): predicted event start in seconds
                end_sec      (float): predicted event end in seconds
                confidence   (float): event presence score in [0, 1]
                found        (bool):  confidence >= threshold
                coarse       (dict):  coarse pass result { start_sec, end_sec }
        """
        from app.pipeline.frame_extractor import get_video_metadata, extract_frames  # noqa: lazy
        video_path = Path(video_path)
        meta = get_video_metadata(video_path)
        duration = meta["duration"]

        # ── encode query once, reuse for both passes ──────────────────
        query_emb = self.text_encoder.encode_query(query)  # (1, 1152)
        query_emb = query_emb.unsqueeze(0)                          # (1, 1, 1152)

        # ── Pass 1: coarse (full video at fps_coarse) ──────────────────
        coarse_start, coarse_end, _ = self._run_pass(
            video_path, query_emb, duration,
            fps=fps_coarse,
        )

        # ── Pass 2: fine (region ±buffer at fps_fine) ──────────────────
        fine_region_start = max(0.0, coarse_start - fine_window_sec)
        fine_region_end   = min(duration, coarse_end + fine_window_sec)

        fine_start, fine_end, conf = self._run_pass(
            video_path, query_emb, duration,
            fps=fps_fine,
            start_sec=fine_region_start,
            end_sec=fine_region_end,
        )

        return {
            "start_sec":  fine_start,
            "end_sec":    fine_end,
            "confidence": conf,
            "found":      conf >= confidence_threshold,
            "coarse":     {"start_sec": coarse_start, "end_sec": coarse_end},
        }

    def _run_pass(
        self,
        video_path: Path,
        query_emb: torch.Tensor,
        duration: float,
        fps: float,
        start_sec: float = 0.0,
        end_sec: float | None = None,
        max_frames: int | None = 512,
    ) -> tuple:
        """Extract frames, encode, run forward, decode span. Returns (start_sec, end_sec, confidence)."""
        from app.pipeline.frame_extractor import extract_frames  # noqa: lazy
        frames_data = extract_frames(video_path, fps=fps, start_sec=start_sec, end_sec=end_sec, max_frames=max_frames)
        if not frames_data:
            return start_sec, end_sec or duration, 0.0

        timestamps_abs = [ts for ts, _ in frames_data]
        pil_imgs       = [img for _, img in frames_data]
        N              = len(pil_imgs)

        # fractional timestamps t/T relative to full video duration
        timestamps = torch.tensor(
            [ts / duration for ts in timestamps_abs],
            dtype=torch.float32, device=self.device,
        )

        frame_embs = self.visual_encoder.encode_frames(pil_imgs)  # (N, 1152)
        frame_embs = frame_embs.unsqueeze(0)                               # (1, N, 1152)

        # forward pass
        start_logits, end_logits, confidence = self.forward(frame_embs, timestamps, query_emb)

        # decode best span for sample 0
        s_idx, e_idx = decode_best_span(start_logits[0], end_logits[0])
        conf = confidence[0].item()

        # convert frame indices to absolute seconds
        start_s, end_s = to_seconds(s_idx, e_idx, N, timestamps_abs[-1] - timestamps_abs[0] + 1/fps)
        start_s += timestamps_abs[0]
        end_s   += timestamps_abs[0]

        return start_s, end_s, conf

    def count_params(self) -> dict:
        def _count(module):
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in module.parameters())
            return trainable, total

        rows = {
            "text_encoder":     _count(self.text_encoder),
            "temporal_context": _count(self.temporal_context),
            "cross_modal":      _count(self.cross_modal),
            "span_head":        _count(self.span_head),
        }
        if self.visual_encoder is not None:
            rows["visual_encoder"] = _count(self.visual_encoder)
        total_trainable = sum(v[0] for v in rows.values())
        total_all       = sum(v[1] for v in rows.values())
        return {
            "components": {k: {"trainable": v[0], "total": v[1]} for k, v in rows.items()},
            "total_trainable": total_trainable,
            "total_params":    total_all,
            "total_frozen":    total_all - total_trainable,
        }
