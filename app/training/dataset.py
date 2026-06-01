"""
dataset.py — PyTorch Dataset for temporal video grounding.
Phase 5.1 (Chunk K)

Reads the output of scripts/preprocess_qvhighlights.py:
    annotations.jsonl          — one JSON object per video
    frames/{vid}/{ts}.jpg      — 384×384 JPEGs named by timestamp

__getitem__ returns a dict with PIL frames, timestamps, query, and
ground truth frame indices. collate_fn handles variable-length sequences
(different videos have different N frames) by keeping frames and timestamps
as lists — the training loop encodes and pads per sample.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from app.training.augmentation import (
    temporal_jitter,
    # random_temporal_crop,  # disabled — see augmentation.py for why
    speed_perturbation,
    speed_perturbation_embeddings,
    load_paraphrases,
    sample_paraphrase,
    maybe_inject_negative_query,
)


class GroundingDataset(Dataset):
    """
    Args:
        jsonl_path:       path to annotations.jsonl
        frames_root:      parent directory containing frames/{vid}/*.jpg
        augment:          apply augmentations (True=train, False=val/test)
        paraphrases_path: optional path to paraphrases.json
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        frames_root: str | Path,
        augment: bool = True,
        paraphrases_path: Optional[str] = None,
    ):
        self.frames_root = Path(frames_root)
        self.augment     = augment

        with open(jsonl_path, encoding="utf-8") as f:
            all_samples = [json.loads(line) for line in f if line.strip()]

        # Keep only samples whose frame dir exists and holds at least one JPEG.
        # Lets training run on whatever frame batches are on disk (e.g. skipping
        # the loose first-1000 vids) without crashing in __getitem__.
        self.samples = [
            s for s in all_samples
            if (self.frames_root / s["vid"]).is_dir()
            and next((self.frames_root / s["vid"]).glob("*.jpg"), None) is not None
        ]
        dropped = len(all_samples) - len(self.samples)
        if dropped:
            print(f"GroundingDataset: kept {len(self.samples)} / {len(all_samples)} "
                  f"annotations ({dropped} skipped — frames not on disk)")

        self.paraphrases  = load_paraphrases(paraphrases_path) if paraphrases_path else {}
        self.all_queries  = [s["query"] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        vid          = s["vid"]
        query        = s["query"]
        duration     = float(s["duration"])
        gt_start_sec = float(s["gt_start_sec"])
        gt_end_sec   = float(s["gt_end_sec"])

        # ── Load JPEG frames ──────────────────────────────────────────────
        frame_dir = self.frames_root / vid
        jpg_files = sorted(frame_dir.glob("*.jpg"))

        if not jpg_files:
            raise RuntimeError(f"No frames found for vid '{vid}' at {frame_dir}")

        frames     = [Image.open(p).convert("RGB") for p in jpg_files]
        timestamps = [float(p.stem) for p in jpg_files]  # filename = timestamp

        # ── Augmentations (training only) ─────────────────────────────────
        is_negative = False
        if self.augment:
            gt_start_sec, gt_end_sec = temporal_jitter(
                gt_start_sec, gt_end_sec, duration
            )
            # random_temporal_crop disabled — see augmentation.py for why
            # frames, timestamps, gt_start_sec, gt_end_sec = random_temporal_crop(
            #     frames, timestamps, gt_start_sec, gt_end_sec
            # )
            frames, timestamps = speed_perturbation(frames, timestamps)
            query = sample_paraphrase(query, self.paraphrases)
            query, is_negative = maybe_inject_negative_query(query, self.all_queries)

        # ── GT seconds → nearest frame index ─────────────────────────────
        N = len(timestamps)
        gt_start_idx = min(range(N), key=lambda i: abs(timestamps[i] - gt_start_sec))
        gt_end_idx   = min(range(N), key=lambda i: abs(timestamps[i] - gt_end_sec))
        gt_start_idx = max(0, min(gt_start_idx, N - 1))
        gt_end_idx   = max(gt_start_idx, min(gt_end_idx, N - 1))

        # Negative query: span target is irrelevant — zero it out.
        # Training loop uses is_negative flag to skip span loss and set
        # confidence target to 0.
        if is_negative:
            gt_start_idx = 0
            gt_end_idx   = 0

        return {
            "frames":       frames,        # List[PIL.Image]  — length N
            "timestamps":   timestamps,    # List[float]      — length N
            "query":        query,         # str
            "gt_start_idx": gt_start_idx,  # int
            "gt_end_idx":   gt_end_idx,    # int
            "is_negative":  is_negative,   # bool
            "duration":     duration,      # float
            "vid":          vid,           # str (for logging)
        }


def collate_fn(batch: list) -> dict:
    """
    Variable-length collate. frames/timestamps stay as lists-of-lists
    because each video has a different N. The training loop handles
    per-sample encoding and pads to the batch maximum.
    """
    return {
        "frames":       [s["frames"]     for s in batch],
        "timestamps":   [s["timestamps"] for s in batch],
        "query":        [s["query"]      for s in batch],
        "gt_start_idx": torch.tensor([s["gt_start_idx"] for s in batch], dtype=torch.long),
        "gt_end_idx":   torch.tensor([s["gt_end_idx"]   for s in batch], dtype=torch.long),
        "is_negative":  torch.tensor([s["is_negative"]  for s in batch], dtype=torch.bool),
        "duration":     torch.tensor([s["duration"]     for s in batch], dtype=torch.float32),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Cached-feature variant (Phase 2 — feature caching)
# ──────────────────────────────────────────────────────────────────────────────

class CachedGroundingDataset(Dataset):
    """
    Cached-feature counterpart to GroundingDataset.

    Loads pre-extracted FROZEN SigLIP embeddings from cache/{vid}.pt (written by
    scripts/precompute_embeddings.py) instead of decoding JPEGs and running the
    vision tower. __getitem__ returns the (N, 1152) embedding tensor in place of
    PIL frames; GT indexing, augmentations, and negatives are identical to
    GroundingDataset, so the training loop differs only in the encode step.

    Args:
        jsonl_path:       path to annotations.jsonl
        cache_dir:        dir containing {vid}.pt files
        augment:          apply augmentations (True=train, False=val/test)
        paraphrases_path: optional path to paraphrases.json
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        cache_dir: str | Path,
        augment: bool = True,
        paraphrases_path: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.augment   = augment

        with open(jsonl_path, encoding="utf-8") as f:
            all_samples = [json.loads(line) for line in f if line.strip()]

        # Keep only samples whose cache file exists — mirrors GroundingDataset's
        # on-disk frame filter, so training runs on whatever has been cached so far.
        self.samples = [
            s for s in all_samples
            if (self.cache_dir / f"{s['vid']}.pt").exists()
        ]
        dropped = len(all_samples) - len(self.samples)
        if dropped:
            print(f"CachedGroundingDataset: kept {len(self.samples)} / {len(all_samples)} "
                  f"annotations ({dropped} skipped — no cache file)")

        self.paraphrases = load_paraphrases(paraphrases_path) if paraphrases_path else {}
        self.all_queries = [s["query"] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        vid          = s["vid"]
        query        = s["query"]
        duration     = float(s["duration"])
        gt_start_sec = float(s["gt_start_sec"])
        gt_end_sec   = float(s["gt_end_sec"])

        # ── Load cached embeddings (no SigLIP, no JPEG decode) ────────────
        blob       = torch.load(self.cache_dir / f"{vid}.pt", map_location="cpu")
        embeddings = blob["embeddings"]            # (N, 1152) fp16
        timestamps = blob["timestamps"].tolist()   # List[float] — seconds

        # ── Augmentations (training only) ─────────────────────────────────
        is_negative = False
        if self.augment:
            gt_start_sec, gt_end_sec = temporal_jitter(gt_start_sec, gt_end_sec, duration)
            embeddings, timestamps   = speed_perturbation_embeddings(embeddings, timestamps)
            query = sample_paraphrase(query, self.paraphrases)
            query, is_negative = maybe_inject_negative_query(query, self.all_queries)

        # ── GT seconds → nearest frame index ─────────────────────────────
        N = len(timestamps)
        gt_start_idx = min(range(N), key=lambda i: abs(timestamps[i] - gt_start_sec))
        gt_end_idx   = min(range(N), key=lambda i: abs(timestamps[i] - gt_end_sec))
        gt_start_idx = max(0, min(gt_start_idx, N - 1))
        gt_end_idx   = max(gt_start_idx, min(gt_end_idx, N - 1))

        if is_negative:
            gt_start_idx = 0
            gt_end_idx   = 0

        return {
            "frame_embs":   embeddings,    # (N, 1152) tensor (fp16) — replaces "frames"
            "timestamps":   timestamps,    # List[float]
            "query":        query,         # str
            "gt_start_idx": gt_start_idx,  # int
            "gt_end_idx":   gt_end_idx,    # int
            "is_negative":  is_negative,   # bool
            "duration":     duration,      # float
            "vid":          vid,           # str
        }


def cached_collate_fn(batch: list) -> dict:
    """
    Collate for CachedGroundingDataset. frame_embs/timestamps stay as lists
    (variable N per video); the training loop pads to the batch maximum in
    encode_batch_cached.
    """
    return {
        "frame_embs":   [s["frame_embs"] for s in batch],
        "timestamps":   [s["timestamps"] for s in batch],
        "query":        [s["query"]      for s in batch],
        "gt_start_idx": torch.tensor([s["gt_start_idx"] for s in batch], dtype=torch.long),
        "gt_end_idx":   torch.tensor([s["gt_end_idx"]   for s in batch], dtype=torch.long),
        "is_negative":  torch.tensor([s["is_negative"]  for s in batch], dtype=torch.bool),
        "duration":     torch.tensor([s["duration"]     for s in batch], dtype=torch.float32),
    }
