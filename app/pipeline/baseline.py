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

⚠️  KNOWN LIMITATION — frame embeddings are recomputed on EVERY call.
    If the same video is queried multiple times (e.g. user asks 3 questions
    about the same lecture), the SigLIP 2 visual tower runs 3 times
    on the same frames — wasteful and slow.

    TODO (Phase 7 — API Routes):
    Add an embedding cache so frame_feats are computed once per video
    and reused for every subsequent query on that video.
    Simple fix — just a Python dict or torch.save():

        cache = {}   # { video_path_str : frame_feats_tensor }

        if video_path in cache:
            frame_feats = cache[video_path]   # reuse, skip SigLIP visual tower
        else:
            frame_feats = encode_frames(...)  # slow, runs once
            cache[video_path] = frame_feats   # store for next query

    No vector database needed — a plain dict is fine since we search
    WITHIN one video, not ACROSS millions of videos.
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
) -> dict:
    """
    Run the naive zero-shot baseline on a single video + query.

    Args:
        video_path:  Path to the video file
        query:       Natural language query, e.g. "when the speaker laughs"
        model:       Loaded SigLIP 2 model (AutoModel)
        processor:   Loaded SigLIP 2 processor (AutoProcessor)
        fps:         Frame sampling rate (default 1.0 fps)
        window_sec:  Half-width of the predicted span in seconds (default ±5s)
        device:      "cuda" or "cpu"
        batch_size:  How many frames to encode at once (reduce if OOM)

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

    # ---- 1. Extract frames -----------------------------------------
    logger.info(f"Extracting frames from '{video_path.name}' at {fps}fps...")
    frames = extract_frames(video_path, fps=fps)

    if not frames:
        raise RuntimeError(f"No frames extracted from '{video_path.name}'")

    meta      = get_video_metadata(video_path)
    all_ts    = [ts for ts, _ in frames]
    pil_imgs  = [img for _, img in frames]

    # ---- 2. Encode text query (just once) --------------------------
    logger.info(f"Encoding query: '{query}'")
    text_inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        text_feat = model.get_text_features(**text_inputs)   # (1, 1152)
    text_feat = F.normalize(text_feat, dim=-1)

    # ---- 3. Encode frames in batches ------------------------------
    # ⚠️  BOTTLENECK: this re-runs SigLIP 2 visual tower on every call,
    #  even if the same video was already encoded for a previous query.
    #  frame_feats is a local variable — it gets thrown away when this
    #  function returns, so the next query recomputes it from scratch.
    #
    #  TODO (Phase 7): move frame encoding outside this function and
    #  pass frame_feats in as a parameter (or use the cache dict above).
    #  Only the query encoding (Step 2) and similarity (Step 4) need
    #  to re-run per query — frame encoding should run once per video.
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
