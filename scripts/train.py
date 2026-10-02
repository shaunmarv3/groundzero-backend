#!/usr/bin/env python3
"""
train.py — Standalone GroundZero training script.
Works on Lightning.ai Studios, Vast.ai, RunPod, Lambda Labs, or any SSH GPU machine.

──────────────────────────────────────────────────────────────────────────────
SETUP (run once in the terminal before training):

    pip install "transformers>=4.49.0" "peft>=0.18.0" einops wandb \
                huggingface_hub datasets tqdm

──────────────────────────────────────────────────────────────────────────────
EXAMPLE RUN (Lightning.ai / any SSH machine):

    # Option A — pass tokens as args
    python scripts/train.py \
        --hf_token   YOUR_HF_TOKEN \
        --wandb_key  YOUR_WANDB_KEY \
        --data_dir   ~/data/qvhighlights \
        --ckpt_dir   ~/checkpoints \
        --batch_size 8 \
        --epochs     20 \
        --max_hours  5.5

    # Option B — export as env vars, then run
    export HF_TOKEN=xxx
    export WANDB_API_KEY=xxx
    python scripts/train.py --batch_size 8 --max_hours 5.5

──────────────────────────────────────────────────────────────────────────────
SESSION-35 FIXES (all off by default → no flags = the original shipped run):

    --text_cache       queries from text_pooled*/text_tokens*.pt → NO SigLIP loaded
    --fix_contrastive  Fix 1: contrastive loss on post-cross-modal features
    --gelu             Fix 2: GELU between the temporal convs
    --word_level       Fix 3: frames attend over query word tokens (QD-DETR-style)
    --mask_padding     Fix 4: mask batch padding in attention / span head / losses
    --all_fixes        = fixes 1-4      --lowercase   lowercase queries (*_lower.pt)
    --run_name X       checkpoints go to <hf_repo>/X/ (parallel runs never collide)
    --init_from P      fine-tune from P's trainable weights (fresh optimizer)
    --resume_from_hf   new session, empty disk → pull <hf_repo>/X/latest.pt first

    python scripts/train.py --use_cache --text_cache --all_fixes \
        --data_dir /kaggle/working/data --skip_download \
        --hf_repo shaunmarvell/qvhighlights-model --run_name s35-all-fixes \
        --batch_size 32 --lr 1e-4 --epochs 200 --push_every 10 --resume_from_hf

──────────────────────────────────────────────────────────────────────────────
RESUME (auto-detects latest local checkpoint):

    python scripts/train.py --batch_size 8 --skip_download
    # or point explicitly:
    python scripts/train.py --resume ~/checkpoints/latest.pt --skip_download

──────────────────────────────────────────────────────────────────────────────
WHAT GETS PUSHED TO HUGGINGFACE EACH EPOCH:

    best.pt   — only trainable params (~310 MB).  Updated when val R@1 improves.
    latest.pt — trainable params + optimizer/scheduler states (~930 MB).
                Overwritten every epoch. Use this to resume.

    Full frozen SigLIP 2 weights are NOT saved — they are re-loaded from
    HuggingFace every run (identical across runs, no need to store them).
"""

import argparse
import os
import sys
import tarfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from huggingface_hub import login, HfApi, hf_hub_download
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.groundzero_model import GroundZeroModel, DEFAULT_MODEL_CFG
from app.training.batch_encoding import TextCache, encode_cached_batch
from app.training.dataset import (
    GroundingDataset, collate_fn,
    CachedGroundingDataset, cached_collate_fn,
)
from app.training.losses import combined_loss
from app.pipeline.span_extraction import decode_best_span, to_seconds


# ──────────────────────────────────────────────────────────────────────────────
# Dataset download
# ──────────────────────────────────────────────────────────────────────────────

DATASET_ID = "shaunmarvell/qvhighlights-1fps"

TRAIN_TARS = [
    "frames_000000_001000.tar",   # 880 recovered vids (loose frames re-packed; see Session 20)
    "frames_001000_002000.tar",
    "frames_002000_003000.tar",
    "frames_003000_004000.tar",
    "frames_004000_005000.tar",
    "frames_005000_006000.tar",
    "frames_006000_007241.tar",
    "frames_007242_007445.tar",
]


def download_dataset(data_dir: Path, cache_only: bool = False,
                     cache_tar: str = "cache_embeddings.tar") -> tuple:
    """
    Download QVHighlights from HuggingFace if not already present.
    Uses sentinel files so interrupted downloads can be resumed safely.
    Returns (train_jsonl, val_jsonl, frames_dir).

    cache_only=True  → cached-feature training: download annotations + the
    precomputed-embedding tar (`cache_tar`) ONLY, skipping the ~30 GB of raw
    frame tars (which cache-mode training never reads). This lets you extract on
    one machine, push the cache (precompute_embeddings.py --push_cache_repo), then
    train on a fresh/cheaper machine by pulling just the ~2.6 GB cache tar.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = data_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Annotations (small — a few MB)
    for fname in ["annotations_train.jsonl", "annotations_val.jsonl"]:
        dest = data_dir / fname
        if dest.exists():
            print(f"  {fname}: already present, skipping")
        else:
            print(f"  Downloading {fname}...")
            hf_hub_download(DATASET_ID, fname, repo_type="dataset", local_dir=str(data_dir))

    # Cached-feature path: pull only the embedding tar, not the raw frames.
    if cache_only:
        flag = data_dir / f".{cache_tar}.done"
        if flag.exists():
            print(f"  {cache_tar}: already extracted, skipping")
        else:
            print(f"  Downloading {cache_tar} (precomputed embeddings)...")
            p = hf_hub_download(DATASET_ID, cache_tar, repo_type="dataset", local_dir=str(data_dir))
            print(f"  Extracting {cache_tar}...")
            with tarfile.open(p) as t:
                t.extractall(str(data_dir))   # members are cache/{vid}.pt → data_dir/cache/
            os.remove(p)
            flag.touch()
        cache_path = data_dir / "cache"
        cache_n = sum(1 for _ in cache_path.glob("*.pt")) if cache_path.is_dir() else 0
        train_jsonl = data_dir / "annotations_train.jsonl"
        val_jsonl   = data_dir / "annotations_val.jsonl"
        print(f"Cache ready — {cache_n} embedding files / "
              f"{sum(1 for _ in open(train_jsonl))} train / {sum(1 for _ in open(val_jsonl))} val annotations")
        return train_jsonl, val_jsonl, frames_dir

    # The first-chunk vids used to be skipped (150k loose JPEGs downloaded one
    # HTTP-request-each, far too slow). They were recovered via `git clone` of the
    # dataset (git-LFS batch API sidesteps the per-file resolver rate limit), then
    # re-packed into frames_000000_001000.tar (880 vids — the repo only ever held
    # 880 loose-frame dirs, not 1000). So ALL chunks now come via fast tar extract.
    # GroundingDataset self-filters annotations to whatever frame dirs are on disk.

    # Train tar files (full train split, ~7430 vids)
    for tar_name in TRAIN_TARS:
        done_flag = data_dir / f".{tar_name}.done"
        if done_flag.exists():
            print(f"  {tar_name}: already extracted, skipping")
            continue
        print(f"  Downloading {tar_name}...")
        p = hf_hub_download(DATASET_ID, tar_name, repo_type="dataset", local_dir=str(data_dir))
        print(f"  Extracting {tar_name}...")
        with tarfile.open(p) as t:
            t.extractall(str(data_dir))
        os.remove(p)
        done_flag.touch()
        n = sum(1 for x in frames_dir.iterdir() if x.is_dir())
        print(f"  Done — {n} vid dirs so far")

    # Val tar
    val_flag = data_dir / ".val_tar.done"
    if val_flag.exists():
        print("  Val frames: already present, skipping")
    else:
        print("  Downloading val tar...")
        p = hf_hub_download(
            DATASET_ID, "val_frames_000000_end.tar", repo_type="dataset", local_dir=str(data_dir)
        )
        with tarfile.open(p) as t:
            t.extractall(str(data_dir))
        os.remove(p)
        val_flag.touch()
        print("  Val frames extracted.")

    train_jsonl = data_dir / "annotations_train.jsonl"
    val_jsonl   = data_dir / "annotations_val.jsonl"
    n_train = sum(1 for _ in open(train_jsonl))
    n_val   = sum(1 for _ in open(val_jsonl))
    print(f"Dataset ready — {n_train} train / {n_val} val annotations")
    return train_jsonl, val_jsonl, frames_dir


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def encode_batch(model, batch, device, d_model=1152, chunk_size=8, lowercase=False):
    """
    Encode one batch of PIL frame lists + query strings → tensors.
    Must be called inside torch.cuda.amp.autocast() so SigLIP runs in fp16.

    All frames across the B videos are flattened into ONE list and encoded
    together in chunks of `chunk_size`. This feeds the GPU big kernels instead
    of B separate per-video passes of 8 frames each — far better SM occupancy
    on an H100/A100. Embeddings are then split back per-video using frame counts.

    Returns (same 6-tuple as encode_cached_batch):
        frame_embs   : (B, N_max, d_model)  float32, padded with zeros
        timestamps   : (B, N_max)           fractional t/duration, padded 1.0
        frame_mask   : (B, N_max) bool      True = real frame
        query        : (B, 1, d_model)      float32 (pooled — word-level needs the text cache)
        query_mask   : None
        query_pooled : (B, 1, d_model)      same tensor as query
    """
    B = len(batch["frames"])

    # Flatten all frames across the batch, remember each video's frame count
    all_frames, counts = [], []
    for i in range(B):
        frames_i = batch["frames"][i]
        all_frames.extend(frames_i)
        counts.append(len(frames_i))

    # One batched encode for the whole batch (sum(counts) frames)
    all_embs = model.visual_encoder.encode_frames(all_frames, chunk_size=chunk_size)

    # Split back per-video and build the fractional-timestamp tensors
    emb_list, ts_list = [], []
    offset = 0
    for i in range(B):
        n = counts[i]
        emb_list.append(all_embs[offset:offset + n].float())
        offset += n
        dur_i = batch["duration"][i].item()
        frac_i = torch.tensor(
            [t / dur_i for t in batch["timestamps"][i]],
            dtype=torch.float32,
            device=device,
        )
        ts_list.append(frac_i)

    N_max      = max(counts)
    frame_embs = torch.zeros(B, N_max, d_model, device=device)
    timestamps = torch.ones(B, N_max, device=device)   # 1.0 = end-of-video pad position
    frame_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

    for i, (emb, ts) in enumerate(zip(emb_list, ts_list)):
        n = emb.shape[0]
        frame_embs[i, :n] = emb
        timestamps[i, :n] = ts
        frame_mask[i, :n] = True

    texts = [q.lower() for q in batch["query"]] if lowercase else batch["query"]
    query_emb = model.text_encoder.encode_queries(texts).float().unsqueeze(1)  # (B,1,D)
    return frame_embs, timestamps, frame_mask, query_emb, None, query_emb


# encode_batch_cached moved to app/training/batch_encoding.py (encode_cached_batch),
# shared with evaluate.py so the two can't drift apart.


def temporal_iou(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def trainable_state_dict(model):
    """Return only the trainable parameter keys from model.state_dict()."""
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable_names}


def save_and_push(
    model, optimizer, scheduler, scaler,
    epoch, global_step, best_r1_05, r1_05,
    ckpt_dir, hf_repo, api, is_best,
    push_latest=True, model_cfg=None, run_name=None,
):
    """
    Save two checkpoints every epoch:
      latest.pt — full training state (for resume). Overwrites each epoch.
      best.pt   — trainable params only (for inference). Overwrites on improvement.

    latest.pt is ALWAYS written to disk every epoch (so resume never loses progress).
    It is only *uploaded* to HuggingFace when push_latest=True — gated by --push_every,
    because the ~1.9GB upload can cost more wall-clock than a cached epoch. best.pt is
    smaller (trainable params only) and is pushed on every improvement.
    """
    t_state = trainable_state_dict(model)

    # latest.pt — always save for resume capability
    latest_path = ckpt_dir / "latest.pt"
    torch.save(
        {
            "epoch":           epoch,
            "step":            global_step,
            "trainable_state": t_state,
            "optimizer":       optimizer.state_dict(),
            "scheduler":       scheduler.state_dict(),
            "scaler":          scaler.state_dict(),
            "best_r1_05":      best_r1_05,
            "r1_iou05":        r1_05,
            "model_cfg":       model_cfg or dict(DEFAULT_MODEL_CFG),
        },
        latest_path,
    )
    print(f"  Saved latest.pt (ep {epoch}, R@1={r1_05:.4f})")
    if push_latest:
        _push(latest_path, hf_repo, api, run_name)
    else:
        print(f"  (skipped HF push of latest.pt — next push per --push_every)")

    # best.pt — only when val improved
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(
            {
                "epoch":           epoch,
                "r1_iou05":        best_r1_05,
                "trainable_state": t_state,
                "model_cfg":       model_cfg or dict(DEFAULT_MODEL_CFG),
            },
            best_path,
        )
        print(f"  Saved best.pt (R@1={best_r1_05:.4f}  ← new best)")
        _push(best_path, hf_repo, api, run_name)


def _hf_path(name: str, run_name: str | None) -> str:
    """Each run gets its own folder in the HF repo, so parallel runs never overwrite each other."""
    return f"{run_name}/{name}" if run_name else name


def _push(ckpt_path: Path, hf_repo: str, api: HfApi, run_name: str | None = None):
    try:
        api.create_repo(hf_repo, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=_hf_path(ckpt_path.name, run_name),
            repo_id=hf_repo,
            repo_type="model",
        )
        print(f"  Pushed {ckpt_path.name} → hf:{hf_repo}/{_hf_path(ckpt_path.name, run_name)}")
    except Exception as exc:
        print(f"  HF push failed (non-fatal): {exc}")


def forward_batch(model, batch, device, args, text_cache=None) -> dict:
    """
    Encode one batch + run the model. Call inside autocast. Shared by the training
    loop and validate() so both always run the identical path.
    """
    if args.use_cache:
        fe, ts, fmask, q, qmask, qpool = encode_cached_batch(
            model, batch, device, text_cache=text_cache, lowercase=args.lowercase)
    else:
        fe, ts, fmask, q, qmask, qpool = encode_batch(
            model, batch, device, chunk_size=args.chunk_size, lowercase=args.lowercase)
    start_logits, end_logits, confidence, grounded = model(
        fe, ts, q,
        frame_mask=fmask if args.mask_padding else None,   # Fix 4
        query_mask=qmask,
        return_features=True,
    )
    return {"start": start_logits, "end": end_logits, "conf": confidence,
            "grounded": grounded, "frame_embs": fe, "frame_mask": fmask, "query_pooled": qpool}


def contrastive_inputs(out: dict, pos: torch.Tensor, args) -> tuple:
    """
    What the contrastive term sees, for the positive (non-negative-query) samples.
    Fix 1: the post-cross-modal `grounded` features — a function of trainable weights.
    Without --fix_contrastive: the raw cached `frame_embs` (the original bug, kept so
    the old run is reproducible) — no grad path → the term is a constant.
    """
    feats   = out["grounded"] if args.fix_contrastive else out["frame_embs"]
    lengths = out["frame_mask"].sum(dim=1) if args.mask_padding else None
    return (out["query_pooled"][pos], feats[pos],
            lengths[pos] if lengths is not None else None)


@torch.no_grad()
def validate(model, val_loader, device, args, text_cache=None):
    model.eval()
    hits_05 = hits_07 = total = 0
    val_loss_sum = 0.0

    for batch in val_loader:
        with torch.cuda.amp.autocast():
            out = forward_batch(model, batch, device, args, text_cache)
        start_logits, end_logits = out["start"], out["end"]

        for i in range(start_logits.shape[0]):
            # timestamps is present in both the JPEG and cached collates; frames is not
            n_frames = len(batch["timestamps"][i])
            # decode over real frames only — same as evaluate.py
            s_idx, e_idx = decode_best_span(start_logits[i, :n_frames], end_logits[i, :n_frames])
            s_idx = min(s_idx, n_frames - 1)
            e_idx = min(e_idx, n_frames - 1)
            dur_i  = batch["duration"][i].item()
            pred_s, pred_e = to_seconds(s_idx, e_idx, n_frames, dur_i)

            ts_i = batch["timestamps"][i]
            gt_s = ts_i[min(batch["gt_start_idx"][i].item(), len(ts_i) - 1)]
            gt_e = ts_i[min(batch["gt_end_idx"][i].item(),   len(ts_i) - 1)]
            iou  = temporal_iou(pred_s, pred_e, gt_s, gt_e)
            hits_05 += iou >= 0.5
            hits_07 += iou >= 0.7
            total   += 1

        pos = ~batch["is_negative"].to(device)
        if pos.any():
            q_pool, feats, lengths = contrastive_inputs(out, pos, args)
            with torch.cuda.amp.autocast():
                loss, *_ = combined_loss(
                    start_logits[pos], end_logits[pos],
                    batch["gt_start_idx"].to(device)[pos],
                    batch["gt_end_idx"].to(device)[pos],
                    q_pool, feats, lengths=lengths,
                )
            val_loss_sum += loss.item()

    return {
        "val/r1_iou05": hits_05 / max(total, 1),
        "val/r1_iou07": hits_07 / max(total, 1),
        "val/loss":     val_loss_sum / max(len(val_loader), 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train GroundZero")

    # Auth (can also be set via HF_TOKEN / WANDB_API_KEY env vars)
    p.add_argument("--hf_token",  default=os.environ.get("HF_TOKEN", ""))
    p.add_argument("--wandb_key", default=os.environ.get("WANDB_API_KEY", ""))

    # Paths
    p.add_argument("--data_dir",      default="~/data/qvhighlights")
    p.add_argument("--ckpt_dir",      default="./checkpoints")
    p.add_argument("--hf_repo",       default="shaunmarvell/groundzero-weights")
    p.add_argument("--resume",        default=None,
                   help="Checkpoint path to resume from. Defaults to auto-detect latest.pt")
    p.add_argument("--skip_download", action="store_true",
                   help="Skip dataset download check (data already on disk)")

    # Feature caching (Phase 2)
    p.add_argument("--use_cache", action="store_true",
                   help="Train on pre-extracted SigLIP embeddings (cache/{vid}.pt from "
                        "scripts/precompute_embeddings.py). Skips the vision tower + LoRA "
                        "entirely — frame embeddings come from disk.")
    p.add_argument("--cache_dir", default=None,
                   help="Dir of cached {vid}.pt files (default: <data_dir>/cache)")
    p.add_argument("--cache_tar", default="cache_embeddings.tar",
                   help="With --use_cache (and without --skip_download): name of the precomputed-"
                        "embedding tar to pull from HF instead of the raw frame tars. Must match "
                        "precompute_embeddings.py --cache_tar_name.")

    # Training
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--grad_accum",   type=int,   default=1,
                   help="Gradient accumulation steps. effective_batch = batch_size × grad_accum")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--log_every",    type=int,   default=50)
    p.add_argument("--push_every",   type=int,   default=1,
                   help="Push latest.pt to HF every N epochs (it is ALWAYS saved locally "
                        "every epoch for resume). latest.pt is ~1.9GB (params+optimizer); "
                        "pushing every epoch can cost more wall-clock than the epoch itself. "
                        "Set e.g. 10 for long runs. best.pt still pushes on every improvement. "
                        "The final epoch and any time-limit exit always push.")
    p.add_argument("--max_hours",    type=float, default=None,
                   help="Stop cleanly after this many hours (e.g. 5.5 for a 6-hour session)")
    p.add_argument("--chunk_size",   type=int,   default=8,
                   help="Frames per SigLIP forward pass. 8 for T4/Colab, 64 for H100/A100")
    p.add_argument("--num_workers",  type=int,   default=4,
                   help="DataLoader workers (JPEG decode). 2 on Colab, 8-12 on H100")

    # Model
    p.add_argument("--lora_rank",  type=int,   default=8)
    p.add_argument("--lora_alpha", type=int,   default=16)
    p.add_argument("--n_layers",   type=int,   default=4)
    p.add_argument("--n_heads",    type=int,   default=8)
    p.add_argument("--dropout",    type=float, default=0.1)

    # Session-35 fixes — every one is OFF by default, so with no flags this script
    # reproduces the original (shipped) training run exactly.
    p.add_argument("--text_cache", action="store_true",
                   help="Read queries from text_pooled*.pt / text_tokens*.pt "
                        "(scripts/precompute_text_embeddings.py) — no SigLIP loaded at all. "
                        "Needs --use_cache. Downloaded from HF into --data_dir if missing.")
    p.add_argument("--fix_contrastive", action="store_true",
                   help="Fix 1: contrastive loss on the post-cross-modal features "
                        "(original passed the frozen cached input → zero gradient).")
    p.add_argument("--gelu",         action="store_true", help="Fix 2: GELU between the temporal convs.")
    p.add_argument("--word_level",   action="store_true",
                   help="Fix 3: frames cross-attend over query word tokens (QD-DETR-style). "
                        "Needs --text_cache.")
    p.add_argument("--mask_padding", action="store_true",
                   help="Fix 4: mask batch-padding frames in attention, span head and losses.")
    p.add_argument("--all_fixes",    action="store_true",
                   help="Shorthand for --fix_contrastive --gelu --word_level --mask_padding.")
    p.add_argument("--lowercase",    action="store_true",
                   help="Lowercase queries (SigLIP 2's pretraining format; uses *_lower.pt).")

    # Run management
    p.add_argument("--run_name", default=None,
                   help="W&B run name AND the folder in --hf_repo this run's checkpoints go to "
                        "(so two runs in parallel never overwrite each other).")
    p.add_argument("--init_from", default=None,
                   help="Load ONLY the trainable weights from this checkpoint (e.g. the shipped "
                        "best.pt) and start a fresh optimizer/schedule — a fine-tune, not a resume.")
    p.add_argument("--resume_from_hf", action="store_true",
                   help="If no local latest.pt, pull <run_name>/latest.pt from --hf_repo first "
                        "(a new Kaggle session starts with an empty disk).")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging entirely.")

    args = p.parse_args()
    if args.all_fixes:
        args.fix_contrastive = args.gelu = args.word_level = args.mask_padding = True
    if args.text_cache and not args.use_cache:
        p.error("--text_cache needs --use_cache (frames from cache, text from cache)")
    if args.word_level and not args.text_cache:
        p.error("--word_level needs --text_cache (word tokens come from text_tokens*.pt)")
    return args


def main():
    args = parse_args()

    # ── Auth ─────────────────────────────────────────────────────────────────
    if args.hf_token:
        login(token=args.hf_token)
    if args.wandb_key and not args.no_wandb:
        wandb.login(key=args.wandb_key)

    # ── Device ───────────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {DEVICE}")
    if DEVICE == "cuda":
        # TF32 gives free ~2x speedup on matmul on H100/A100 with no precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.set_float32_matmul_precision("high")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}  {props.total_memory / 1e9:.1f} GB")

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir).expanduser().resolve()
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nDataset:")
    if args.skip_download:
        train_jsonl = data_dir / "annotations_train.jsonl"
        val_jsonl   = data_dir / "annotations_val.jsonl"
        frames_dir  = data_dir / "frames"
        print("  Skipping download check (--skip_download)")
    elif args.use_cache:
        # Cache-mode: pull only the ~2.6 GB embedding tar, not the 30 GB of frames.
        train_jsonl, val_jsonl, frames_dir = download_dataset(
            data_dir, cache_only=True, cache_tar=args.cache_tar)
    else:
        train_jsonl, val_jsonl, frames_dir = download_dataset(data_dir)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    if args.use_cache:
        cache_dir = (Path(args.cache_dir).expanduser().resolve()
                     if args.cache_dir else (data_dir / "cache"))
        print(f"  Cache mode (--use_cache): loading embeddings from {cache_dir}")
        train_ds = CachedGroundingDataset(train_jsonl, cache_dir, augment=True)
        val_ds   = CachedGroundingDataset(val_jsonl,   cache_dir, augment=False)
        collate  = cached_collate_fn
    else:
        train_ds = GroundingDataset(train_jsonl, frames_dir, augment=True)
        val_ds   = GroundingDataset(val_jsonl,   frames_dir, augment=False)
        collate  = collate_fn

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"\nTrain: {len(train_ds):,} samples / {len(train_loader):,} batches")
    print(f"Val:   {len(val_ds):,} samples / {len(val_loader):,} batches")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nLoading model...")
    model = GroundZeroModel(
        model_id="google/siglip2-so400m-patch14-384",
        d_model=1152,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_layers=[23, 24, 25, 26],
        dropout=args.dropout,
        device=DEVICE,
        use_cache=args.use_cache,
        use_gelu=args.gelu,
        word_level=args.word_level,
        load_text_encoder=not args.text_cache,     # text cache → no SigLIP in memory at all
    )
    model_cfg = {
        "use_gelu":     args.gelu,
        "word_level":   args.word_level,
        "mask_padding": args.mask_padding,
        "lowercase":    args.lowercase,
    }
    print(f"model_cfg: {model_cfg}  |  fix_contrastive={args.fix_contrastive}")
    text_cache = (TextCache(data_dir, lowercase=args.lowercase, word_level=args.word_level,
                            token=args.hf_token or None)
                  if args.text_cache else None)
    counts = model.count_params()
    print(f"Trainable: {counts['total_trainable'] / 1e6:.1f} M  |  "
          f"Frozen: {counts['total_frozen'] / 1e6:.1f} M")

    # ── Optimizer + Scheduler + GradScaler ────────────────────────────────────
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay_params if ("lora_" in name or "bias" in name) else decay_params).append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )
    # total_steps = optimizer steps (not batch steps) — grad_accum reduces update frequency
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler      = torch.cuda.amp.GradScaler()

    print(f"AdamW lr={args.lr}  |  cosine warmup={args.warmup_steps} / {total_steps} steps")
    print(f"batch_size={args.batch_size}  grad_accum={args.grad_accum}  "
          f"effective_batch={args.batch_size * args.grad_accum}  chunk_size={args.chunk_size}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    best_r1_05  = 0.0

    resume_path = args.resume
    if resume_path is None:
        candidate = ckpt_dir / "latest.pt"
        if not candidate.exists() and args.resume_from_hf:
            try:
                print(f"\nNo local latest.pt — trying hf:{args.hf_repo}/{_hf_path('latest.pt', args.run_name)}")
                got = hf_hub_download(args.hf_repo, _hf_path("latest.pt", args.run_name),
                                      repo_type="model", local_dir=str(ckpt_dir))
                os.replace(got, candidate)
            except Exception as exc:
                print(f"  nothing to resume on HF ({type(exc).__name__}) — starting fresh")
        if candidate.exists():
            resume_path = str(candidate)
            print(f"\nAuto-resume: found {candidate}")

    if resume_path and Path(resume_path).exists():
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
        # strict=False: frozen SigLIP params not in checkpoint — they stay as initialized
        model.load_state_dict(ckpt["trainable_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_r1_05  = ckpt.get("best_r1_05", 0.0)
        print(f"Resumed from epoch {ckpt['epoch']}  step {global_step}  "
              f"best R@1={best_r1_05:.4f}")
    elif args.init_from:
        src = torch.load(args.init_from, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(src["trainable_state"], strict=False)
        trainable  = {n for n, p_ in model.named_parameters() if p_.requires_grad}
        not_loaded = sorted(trainable & set(missing))
        print(f"\nInit from {args.init_from}: loaded {len(src['trainable_state'])} tensors | "
              f"{len(unexpected)} unexpected | {len(not_loaded)} trainable tensors NOT in it")
        if unexpected or not_loaded:
            print(f"  WARNING arch mismatch — unexpected {list(unexpected)[:3]} / not loaded {not_loaded[:3]}")
        print("  Fresh optimizer + LR schedule (fine-tune, not resume).")
    else:
        print("\nStarting fresh (no checkpoint found)")

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_mode = ("disabled" if args.no_wandb else
                  "online" if (args.wandb_key or os.environ.get("WANDB_API_KEY")) else "offline")
    run = wandb.init(
        project="groundzero",
        name=args.run_name or f"train-siglip2-lora-r{args.lora_rank}-bs{args.batch_size}",
        config=vars(args),
        resume="allow",
        mode=wandb_mode,
    )

    api         = HfApi()
    train_start = time.time()

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining {args.epochs} epochs  (batch_size={args.batch_size}"
          + (f"  max_hours={args.max_hours}" if args.max_hours else "") + ")")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs + 1):

        # Time-limit check before starting each epoch
        if args.max_hours and (time.time() - train_start) / 3600 >= args.max_hours:
            print(f"\nTime limit ({args.max_hours}h) reached before epoch {epoch}. Stopping.")
            break

        model.train()
        epoch_loss  = 0.0
        n_updates   = 0
        epoch_start = time.time()

        # accumulators for logging — track the unscaled loss across accum steps
        accum_loss = accum_span = accum_iou = accum_cont = accum_conf = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):

            # Intra-epoch time check: stop with 3% buffer so we have time to save
            if args.max_hours:
                elapsed_h = (time.time() - train_start) / 3600
                if elapsed_h >= args.max_hours * 0.97:
                    print(f"\nApproaching time limit — saving at step {global_step} and exiting.")
                    save_and_push(
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, best_r1_05, 0.0,
                        ckpt_dir, args.hf_repo, api, is_best=False,
                        model_cfg=model_cfg, run_name=args.run_name,
                    )
                    run.finish()
                    return

            is_last_batch = (batch_idx + 1 == len(train_loader))
            do_update     = ((batch_idx + 1) % args.grad_accum == 0) or is_last_batch

            with torch.cuda.amp.autocast():
                out = forward_batch(model, batch, DEVICE, args, text_cache)
                start_logits, end_logits, confidence = out["start"], out["end"], out["conf"]

                gt_start = batch["gt_start_idx"].to(DEVICE)
                gt_end   = batch["gt_end_idx"].to(DEVICE)
                is_neg   = batch["is_negative"].to(DEVICE)
                pos_mask = ~is_neg

                if pos_mask.any():
                    q_pool, feats, lengths = contrastive_inputs(out, pos_mask, args)
                    total_l, span_l, iou_l, cont_l = combined_loss(
                        start_logits[pos_mask], end_logits[pos_mask],
                        gt_start[pos_mask], gt_end[pos_mask],
                        q_pool, feats, lengths=lengths,
                    )
                else:
                    total_l = span_l = iou_l = cont_l = torch.tensor(0.0, device=DEVICE)

                conf_targets = torch.where(
                    is_neg, torch.zeros_like(confidence), torch.ones_like(confidence)
                )
                # binary_cross_entropy is unsafe under autocast (fp16) and PyTorch blocks
                # it outright. Run it with autocast disabled so it executes in fp32.
                # confidence is already a sigmoid probability, so plain BCE is correct here.
                with torch.cuda.amp.autocast(enabled=False):
                    conf_l = F.binary_cross_entropy(confidence.float(), conf_targets.float())
                # divide by grad_accum so gradients average (not sum) across accum steps
                loss   = (total_l + conf_l) / args.grad_accum

            scaler.scale(loss).backward()

            # accumulate for logging (store unscaled values)
            accum_loss += (total_l + conf_l).item()
            accum_span += span_l.item()
            accum_iou  += iou_l.item()
            accum_cont += cont_l.item()
            accum_conf += conf_l.item()

            if do_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.clip_grad
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                n_steps_this_window = min(args.grad_accum, batch_idx + 1 - n_updates * args.grad_accum)
                epoch_loss  += accum_loss / max(n_steps_this_window, 1)
                n_updates   += 1
                global_step += 1

                if global_step % args.log_every == 0:
                    metrics = {
                        "train/loss":      accum_loss / max(n_steps_this_window, 1),
                        "train/span_loss": accum_span / max(n_steps_this_window, 1),
                        "train/iou_loss":  accum_iou  / max(n_steps_this_window, 1),
                        "train/cont_loss": accum_cont / max(n_steps_this_window, 1),
                        "train/conf_loss": accum_conf / max(n_steps_this_window, 1),
                        "train/lr":        scheduler.get_last_lr()[0],
                    }
                    wandb.log(metrics, step=global_step)

                    # ETA: time-per-update × (updates left this epoch + remaining epochs)
                    elapsed_epoch    = time.time() - epoch_start
                    time_per_update  = elapsed_epoch / max(n_updates, 1)
                    updates_per_epoch = max(len(train_loader) // args.grad_accum, 1)
                    updates_left_epoch = updates_per_epoch - n_updates
                    epochs_left      = args.epochs - epoch           # complete future epochs
                    eta_secs = (updates_left_epoch + epochs_left * updates_per_epoch) * time_per_update
                    eta_h, eta_rem = divmod(int(eta_secs), 3600)
                    eta_m = eta_rem // 60

                    print(
                        f"ep {epoch:02d}/{args.epochs}  step {global_step:05d}  "
                        f"loss={metrics['train/loss']:.4f}  "
                        f"span={metrics['train/span_loss']:.4f}  "
                        f"iou={metrics['train/iou_loss']:.4f}  "
                        f"lr={metrics['train/lr']:.2e}  "
                        f"ETA {eta_h}h{eta_m:02d}m"
                    )

                accum_loss = accum_span = accum_iou = accum_cont = accum_conf = 0.0

        epoch_mins = (time.time() - epoch_start) / 60
        avg_loss   = epoch_loss / max(n_updates, 1)
        print(f"\n--- Epoch {epoch:02d}  avg_loss={avg_loss:.4f}  time={epoch_mins:.1f} min ---")

        # ── Validate ──────────────────────────────────────────────────────────
        print("  Validating...")
        vm = validate(model, val_loader, DEVICE, args, text_cache)
        wandb.log({**vm, "epoch": epoch}, step=global_step)
        print(
            f"  Val  R@1 IoU=0.5: {vm['val/r1_iou05']:.4f}  "
            f"IoU=0.7: {vm['val/r1_iou07']:.4f}  "
            f"loss={vm['val/loss']:.4f}"
        )

        # ── Save + push ───────────────────────────────────────────────────────
        is_best = vm["val/r1_iou05"] > best_r1_05
        if is_best:
            best_r1_05 = vm["val/r1_iou05"]

        push_latest = (epoch % args.push_every == 0) or (epoch == args.epochs)
        save_and_push(
            model, optimizer, scheduler, scaler,
            epoch, global_step, best_r1_05, vm["val/r1_iou05"],
            ckpt_dir, args.hf_repo, api, is_best=is_best,
            push_latest=push_latest, model_cfg=model_cfg, run_name=args.run_name,
        )
        print()

    run.finish()
    print(f"\nDone. Best R@1 IoU=0.5: {best_r1_05:.4f}")


if __name__ == "__main__":
    main()
