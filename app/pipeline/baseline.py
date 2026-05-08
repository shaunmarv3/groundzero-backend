"""
app/pipeline/baseline.py — SigLIP 2 Zero-Shot Temporal Baseline
================================================================
The simplest possible temporal grounding approach:
  1. Extract all frames at 1fps
  2. Encode every frame with SigLIP 2 visual tower → 1152-d vectors
  3. Encode the text query with SigLIP 2 text tower → 1152-d vector
  4. Cosine similarity → find the frame that matches best
  5. Return (best_frame_ts - window, best_frame_ts + window) as the prediction

This is the BASELINE — no training, no custom modules.
Expected accuracy: R@1 IoU=0.5 ≈ 0.30–0.38 on QVHighlights.

Used in:
  - scripts/run_baseline.py       (demo / sanity check)
  - notebook/02_baseline_measure  (formal evaluation on QVHighlights val set)

Returns a dict so callers can inspect scores, not just the final answer.

Pass an optional `frame_cache` dict to reuse frame embeddings across
    multiple queries on the same video — see function docstring for usage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from app.pipeline.frame_extractor import extract_frames, get_video_metadata

logger = logging.getLogger(__name__)


def siglip_zeroshot_baseline(
    video_path: str | Path,
    query: str,
    model,
    processor,
    fps: float = 1.0,
    window_sec: float = 5.0,
    device: str = "cuda",
    batch_size: int = 32,
    frame_cache: dict | None = None,
) -> dict:
    """
    Run the naive zero-shot baseline on a single video + query.

    Args:
        video_path:   Path to the video file
        query:        Natural language query, e.g. "when the speaker laughs"
        model:        Loaded SigLIP 2 model (AutoModel)
        processor:    Loaded SigLIP 2 processor (AutoProcessor)
        fps:          Frame sampling rate (default 1.0 fps)
        window_sec:   Half-width of the predicted span in seconds (default ±5s)
        device:       "cuda" or "cpu"
        batch_size:   How many frames to encode at once (reduce if OOM)
        frame_cache:  Optional dict { video_path_str: (frame_feats, all_ts, meta) }.
                      Pass the same dict across multiple queries on the same video —
                      frame encoding runs once and is reused for every subsequent query.
                      Example:
                          cache = {}
                          for query in queries:
                              result = siglip_zeroshot_baseline(..., frame_cache=cache)

    Returns:
        dict with keys:
            pred_start   (float): predicted start time in seconds
            pred_end     (float): predicted end time in seconds
            best_ts      (float): timestamp of the highest-similarity frame
            best_score   (float): cosine similarity score of the best frame
            video_dur    (float): total video duration in seconds
            n_frames     (int):   number of frames encoded
            all_ts       (list):  all frame timestamps
            all_scores   (list):  cosine similarity score per frame
    """
    video_path = Path(video_path)
    cache_key  = str(video_path.resolve())

    # ---- 1. Extract + encode frames (cache if caller provided dict) ----
    if frame_cache is not None and cache_key in frame_cache:
        frame_feats, all_ts, meta = frame_cache[cache_key]
        logger.info(f"Cache hit for '{video_path.name}' — skipping visual encoding")
    else:
        logger.info(f"Extracting frames from '{video_path.name}' at {fps}fps...")
        frames = extract_frames(video_path, fps=fps)

        if not frames:
            raise RuntimeError(f"No frames extracted from '{video_path.name}'")

        meta     = get_video_metadata(video_path)
        all_ts   = [ts for ts, _ in frames]
        pil_imgs = [img for _, img in frames]

        logger.info(f"Encoding {len(pil_imgs)} frames (batch_size={batch_size})...")
        all_frame_feats = []

        for i in range(0, len(pil_imgs), batch_size):
            batch = pil_imgs[i : i + batch_size]
            img_inputs = processor(images=batch, return_tensors="pt").to(device)

            with torch.no_grad():
                frame_feat = model.get_image_features(**img_inputs)  # (B, 1152)
            frame_feat = F.normalize(frame_feat, dim=-1)
            all_frame_feats.append(frame_feat)

        frame_feats = torch.cat(all_frame_feats, dim=0)  # (N, 1152)

        if frame_cache is not None:
            frame_cache[cache_key] = (frame_feats, all_ts, meta)
            logger.info(f"Cached frame embeddings for '{video_path.name}'")

    frame_feats = torch.cat(all_frame_feats, dim=0)  # (N, 1152)

    # ---- 4. Cosine similarity: every frame vs the query -----------
    # Since both are L2-normalised, dot product = cosine similarity
    scores = (frame_feats @ text_feat.T).squeeze(-1)  # (N,)
    scores_list = scores.cpu().float().tolist()

    # ---- 5. Best frame + fixed-width window -----------------------
    best_idx   = scores.argmax().item()
    best_ts    = all_ts[best_idx]
    best_score = scores_list[best_idx]

    pred_start = max(0.0, best_ts - window_sec)
    pred_end   = min(meta["duration"], best_ts + window_sec)

    logger.info(
        f"Best frame: t={best_ts:.2f}s  score={best_score:.4f}  "
        f"→ predicted span [{pred_start:.1f}s, {pred_end:.1f}s]"
    )

    return {
        "pred_start":  pred_start,
        "pred_end":    pred_end,
        "best_ts":     best_ts,
        "best_score":  best_score,
        "video_dur":   meta["duration"],
        "n_frames":    len(frames),
        "all_ts":      all_ts,
        "all_scores":  scores_list,
    }
