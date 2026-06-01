"""
augmentation.py — Training-time data augmentation for GroundZero.
Phase 4.2

All functions operate on:
    frames     : List[PIL.Image]  — frames in time order
    timestamps : List[float]      — seconds, parallel to frames
    gt_start_sec, gt_end_sec      — ground truth boundaries in seconds

None of these run during inference. Only the Phase 5 training loop uses them.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

from PIL import Image


# ---------------------------------------------------------------------------
# 1. Temporal Jitter
# ---------------------------------------------------------------------------

def temporal_jitter(
    gt_start_sec: float,
    gt_end_sec: float,
    duration_sec: float,
    jitter_frac: float = 0.1,
) -> Tuple[float, float]:
    """
    Shift GT boundaries by a random amount up to ±jitter_frac × event_duration.
    Both boundaries shift by the same amount so the span length is preserved.
    Result is clamped to [0, duration_sec].

    Why: prevents the model from memorising exact boundary positions and adds
    robustness to annotation noise (human annotators are off by ~1-2 seconds).
    """
    event_dur = gt_end_sec - gt_start_sec
    max_shift = event_dur * jitter_frac
    shift = random.uniform(-max_shift, max_shift)
    new_start = max(0.0, gt_start_sec + shift)
    new_end   = min(duration_sec, gt_end_sec + shift)
    return new_start, new_end


# ---------------------------------------------------------------------------
# 2. Random Temporal Crop
# ---------------------------------------------------------------------------

# DISABLED for first training run.
# Why: this crop makes videos in the same batch start at different seconds
# (e.g. Video A starts at second 0, Video B starts at second 15). That means
# each video needs its own timestamps list — complicates the training loop.
# All QVHighlights videos are already 150 frames at 1fps, so every batch has
# the same length anyway. No benefit to enabling this yet.
# Re-enable after baseline is confirmed working (Phase 6+).
#
# def random_temporal_crop(
#     frames: List[Image.Image],
#     timestamps: List[float],
#     gt_start_sec: float,
#     gt_end_sec: float,
#     min_frac: float = 0.6,
#     margin_sec: float = 5.0,
# ) -> Tuple[List[Image.Image], List[float], float, float]:
#     """
#     Crop the frame list to a random 60-100% window of the original video.
#     The GT segment is always fully inside the crop (guaranteed by margin_sec buffer).
#     """
#     if not timestamps:
#         return frames, timestamps, gt_start_sec, gt_end_sec
#     duration = timestamps[-1]
#     latest_crop_start  = max(0.0, gt_start_sec - margin_sec)
#     earliest_crop_end  = min(duration, gt_end_sec + margin_sec)
#     min_crop_dur = duration * min_frac
#     crop_start = random.uniform(0.0, latest_crop_start)
#     crop_end_lower = max(earliest_crop_end, crop_start + min_crop_dur)
#     crop_end = random.uniform(crop_end_lower, duration) if crop_end_lower <= duration else duration
#     cropped = [(ts, img) for ts, img in zip(timestamps, frames) if crop_start <= ts <= crop_end]
#     if not cropped:
#         return frames, timestamps, gt_start_sec, gt_end_sec
#     c_frames     = [img for _, img in cropped]
#     c_timestamps = [ts  for ts, _ in cropped]
#     return c_frames, c_timestamps, gt_start_sec, gt_end_sec


# ---------------------------------------------------------------------------
# 3. Speed Perturbation
# ---------------------------------------------------------------------------

def speed_perturbation(
    frames: List[Image.Image],
    timestamps: List[float],
    drop_frac: float = 0.1,
    apply_prob: float = 0.3,
) -> Tuple[List[Image.Image], List[float]]:
    """
    Randomly drop or duplicate ~drop_frac of frames.
    Only applied apply_prob of the time (skipped 70% of calls).

    Drop  → simulates faster playback / skipped frames
    Dup   → simulates slower playback / duplicate frames

    Why: makes the model robust to variable frame rates and encoding artifacts.
    GT boundaries (in seconds) are unaffected — only the frame list changes.
    """
    if random.random() > apply_prob or len(frames) < 3:
        return frames, timestamps

    n = len(frames)
    n_modify = max(1, int(n * drop_frac))

    if random.random() < 0.5:
        # Drop: remove n_modify frames at random positions (keep at least 1)
        indices_to_drop = set(random.sample(range(n), min(n_modify, n - 1)))
        new_frames = [f for i, f in enumerate(frames)     if i not in indices_to_drop]
        new_ts     = [t for i, t in enumerate(timestamps) if i not in indices_to_drop]
    else:
        # Duplicate: insert a copy of n_modify frames right after the original
        indices_to_dup = sorted(random.sample(range(n), min(n_modify, n)), reverse=True)
        new_frames = list(frames)
        new_ts     = list(timestamps)
        for idx in indices_to_dup:
            new_frames.insert(idx + 1, frames[idx])
            new_ts.insert(idx + 1, timestamps[idx])

    return new_frames, new_ts


def speed_perturbation_embeddings(
    embeddings,
    timestamps: List[float],
    drop_frac: float = 0.1,
    apply_prob: float = 0.3,
):
    """
    Embedding-space version of speed_perturbation for the cached-feature pipeline
    (Phase 2). Drops or duplicates ~drop_frac of frame EMBEDDINGS (rows of an
    (N, D) tensor) instead of PIL frames — identical fire rate and semantics.

    Args:
        embeddings : (N, D) tensor of cached frame embeddings
        timestamps : List[float] parallel to embeddings

    Returns:
        (embeddings, timestamps) — possibly shortened (drop) or lengthened (dup).
        GT boundaries (in seconds) are unaffected; only the frame sequence changes.
    """
    n = embeddings.shape[0]
    if random.random() > apply_prob or n < 3:
        return embeddings, timestamps

    n_modify = max(1, int(n * drop_frac))

    if random.random() < 0.5:
        # Drop: remove n_modify rows at random positions (keep at least 1)
        drop = set(random.sample(range(n), min(n_modify, n - 1)))
        keep = [i for i in range(n) if i not in drop]
        new_emb = embeddings[keep]
        new_ts  = [timestamps[i] for i in keep]
    else:
        # Duplicate: insert a copy of n_modify rows right after the original
        dup = sorted(random.sample(range(n), min(n_modify, n)), reverse=True)
        idx = list(range(n))
        new_ts = list(timestamps)
        for i in dup:
            idx.insert(i + 1, i)                 # duplicate original row i
            new_ts.insert(i + 1, timestamps[i])
        new_emb = embeddings[idx]

    return new_emb, new_ts


# ---------------------------------------------------------------------------
# 4. Query Paraphrase Sampling
# ---------------------------------------------------------------------------

def load_paraphrases(path: str) -> dict:
    """
    Load paraphrases.json from disk. Call once at Dataset.__init__, not per sample.
    Returns empty dict if file doesn't exist yet (paraphrases are generated offline).
    """
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sample_paraphrase(query: str, paraphrases: dict) -> str:
    """
    Randomly return the original query or one of its paraphrases.
    Falls back to the original query if no paraphrases exist for it.

    Why: expands the effective training vocabulary so the model learns to match
    semantically equivalent queries, not just specific phrasings.
    """
    candidates = paraphrases.get(query, [])
    if not candidates:
        return query
    return random.choice([query] + candidates)


# ---------------------------------------------------------------------------
# 5. Negative Query Injection
# ---------------------------------------------------------------------------

def maybe_inject_negative_query(
    query: str,
    all_queries: List[str],
    inject_prob: float = 0.2,
) -> Tuple[str, bool]:
    """
    With inject_prob probability, replace the query with one from a different video.
    Returns (query, is_negative).

    When is_negative=True, the training loop should set the confidence target to 0.0
    so the model learns to say 'event not found' when the query doesn't match the video.

    Why: without negative examples the confidence head never learns to output low scores.
    20% injection rate means ~1 in 5 training samples actively teaches rejection.
    """
    if random.random() < inject_prob:
        candidates = [q for q in all_queries if q != query]
        if candidates:
            return random.choice(candidates), True
    return query, False
