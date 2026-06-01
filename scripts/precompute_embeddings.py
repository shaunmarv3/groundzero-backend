#!/usr/bin/env python3
"""
precompute_embeddings.py — Phase 1 of the feature-caching pipeline (Chunk: caching).

WHY THIS EXISTS
───────────────
The Moment-DETR / QD-DETR / CG-DETR / UniVTG recipe (and every QVHighlights
method) runs the visual backbone ONCE, offline, freezes it, and trains the
grounding head on the cached features. That is why they fit 200 epochs in ~4h
on a weak GPU. GroundZero currently re-encodes 150 frames through SigLIP So400m
*every step, every epoch* — paying the vision cost ~20x instead of 1x.

This script does the "1x" part: encode every frame of every video through a
FROZEN SigLIP So400m vision tower a single time and write the embeddings to
disk. Training (Phase 2, --use_cache) then loads these tensors instead of
running SigLIP, so all compute goes into the from-scratch temporal / cross-attn
/ span head — the part that actually needs many epochs to converge.

KEY INVARIANT
─────────────
Frame embeddings are DETERMINISTIC: they depend only on the raw JPEGs on disk.
All augmentations (temporal_jitter, speed_perturbation, paraphrase, negative
queries) are applied later in the dataset/training loop, NOT here. speed_perturbation
will operate on the cached embedding sequence instead of PIL frames.

So we encode each `vid` ONCE (many annotations share a vid — embeddings are
per-video, independent of the query) and store:
    cache/{vid}.pt = {"embeddings": (N, 1152) fp16, "timestamps": (N,) fp32}

This mirrors VisualEncoder.encode_frames EXACTLY (same processor, same
get_image_features → pooler_output extraction) but with NO LoRA and NO gradient
checkpointing — the cached features are pure frozen-backbone outputs, which is
the configuration the cached-feature training path uses.

RESUMABLE
─────────
A vid is skipped if its .pt already exists. Each file is written to a .tmp and
atomically renamed, so a stopped Kaggle/Lightning cell never leaves a corrupt
cache entry. Just re-run to continue.

EXAMPLE
───────
    # frames already on disk (typical: one tar downloaded for a smoke test)
    python scripts/precompute_embeddings.py \
        --data_dir ~/data/qvhighlights \
        --cache_dir ~/data/qvhighlights/cache \
        --skip_download

    # download the dataset first (reuses train.py's resumable downloader)
    python scripts/precompute_embeddings.py \
        --hf_token YOUR_HF_TOKEN \
        --data_dir ~/data/qvhighlights \
        --cache_dir ~/data/qvhighlights/cache
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "google/siglip2-so400m-patch14-384"


# ──────────────────────────────────────────────────────────────────────────────
# Collect the set of vids to encode
# ──────────────────────────────────────────────────────────────────────────────

def collect_vids(jsonl_paths, frames_root: Path) -> list:
    """
    Read one or more annotations.jsonl files and return the UNIQUE list of vids
    whose frame dir exists on disk and holds at least one JPEG.

    Dedup by vid: many annotations share a video, but embeddings are per-video
    and query-independent, so each vid is encoded exactly once. Filtering to
    on-disk dirs mirrors GroundingDataset, so the cache matches what training
    would have seen (e.g. when only some tar batches are present).
    """
    seen = {}
    for jp in jsonl_paths:
        if not Path(jp).exists():
            print(f"  (skip) {jp} not found")
            continue
        with open(jp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                vid = json.loads(line)["vid"]
                if vid in seen:
                    continue
                frame_dir = frames_root / vid
                if frame_dir.is_dir() and next(frame_dir.glob("*.jpg"), None) is not None:
                    seen[vid] = True
    return list(seen.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Encode one video
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_video(model, processor, frame_dir: Path, device: str, chunk_size: int):
    """
    Load all JPEGs for one video (sorted by timestamp filename) and encode them
    through the frozen SigLIP vision tower in chunks.

    Returns:
        embeddings : (N, 1152) fp16 on CPU
        timestamps : (N,)      fp32 on CPU   (filename stems = seconds)
    """
    jpg_files = sorted(frame_dir.glob("*.jpg"))
    timestamps = torch.tensor([float(p.stem) for p in jpg_files], dtype=torch.float32)

    feats = []
    for i in range(0, len(jpg_files), chunk_size):
        chunk = [Image.open(p).convert("RGB") for p in jpg_files[i : i + chunk_size]]
        inputs = processor(images=chunk, return_tensors="pt").to(device)
        out = model.get_image_features(**inputs)
        # transformers >=4.50 returns BaseModelOutputWithPooling, not a bare tensor —
        # pull the pooled (chunk, 1152) embedding out either way (same as encode_frames).
        emb = out if isinstance(out, torch.Tensor) else out.pooler_output
        feats.append(emb.half().cpu())

    embeddings = torch.cat(feats, dim=0)  # (N, 1152) fp16
    return embeddings, timestamps


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pre-extract frozen SigLIP frame embeddings.")
    ap.add_argument("--data_dir",   type=str, required=True,
                    help="Dir containing frames/{vid}/*.jpg and annotations_*.jsonl")
    ap.add_argument("--cache_dir",  type=str, default=None,
                    help="Where to write {vid}.pt (default: <data_dir>/cache)")
    ap.add_argument("--model_id",   type=str, default=MODEL_ID)
    ap.add_argument("--chunk_size", type=int, default=16,
                    help="Frames encoded per forward. Higher is fine here (no backward, "
                         "no grad-ckpt) — raise it on big GPUs to go faster.")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both",
                    help="Which annotation file(s) to source vids from.")
    ap.add_argument("--hf_token",     type=str, default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--skip_download", action="store_true",
                    help="Frames already on disk — don't download the dataset.")
    ap.add_argument("--max_hours", type=float, default=None,
                    help="Soft wall-clock budget. Stop cleanly after this many hours "
                         "(resumable — already-cached vids are skipped on rerun).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only encode the first N vids (smoke test).")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (data_dir / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional dataset download (reuse train.py's resumable downloader) ──────
    if not args.skip_download:
        if args.hf_token:
            from huggingface_hub import login
            login(token=args.hf_token)
        from train import download_dataset  # scripts/ is on sys.path[0] at runtime
        download_dataset(data_dir)

    frames_root = data_dir / "frames"
    if not frames_root.is_dir():
        sys.exit(f"ERROR: {frames_root} does not exist. Run without --skip_download "
                 f"or point --data_dir at the extracted frames.")

    # ── Which vids to encode ──────────────────────────────────────────────────
    jsonls = []
    if args.split in ("train", "both"):
        jsonls.append(data_dir / "annotations_train.jsonl")
    if args.split in ("val", "both"):
        jsonls.append(data_dir / "annotations_val.jsonl")

    vids = collect_vids(jsonls, frames_root)
    if args.limit:
        vids = vids[: args.limit]
    print(f"Found {len(vids)} unique vids on disk to consider.")

    # ── Load FROZEN SigLIP vision tower (no LoRA, no grad-ckpt, fp16) ─────────
    print(f"Loading {args.model_id} (frozen, fp16)...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model.eval().to(device)

    # ── Encode loop ────────────────────────────────────────────────────────────
    t0 = time.time()
    done = skipped = 0
    total_frames = 0      # frames encoded this run
    total_enc_s  = 0.0    # pure GPU encode seconds (excludes disk/io)
    warm_frames  = 0      # same, but excluding the 1st video (CUDA init/compile warmup)
    warm_enc_s   = 0.0
    # On small/limited runs print a timing line for every video; on the full run
    # only every 25th, so the log stays readable.
    verbose = bool(args.limit) or len(vids) <= 20

    for idx, vid in enumerate(vids):
        out_path = cache_dir / f"{vid}.pt"
        if out_path.exists():
            skipped += 1
            continue

        # Time ONLY the encode, with cuda.synchronize() on both sides so we measure
        # real GPU compute (CUDA is async — without sync we'd just time kernel launches).
        if device == "cuda":
            torch.cuda.synchronize()
        t_enc = time.time()
        embeddings, timestamps = encode_video(
            model, processor, frames_root / vid, device, args.chunk_size
        )
        if device == "cuda":
            torch.cuda.synchronize()
        enc_s = time.time() - t_enc

        n = embeddings.shape[0]
        total_frames += n
        total_enc_s  += enc_s
        if done >= 1:                 # skip the 1st video: it absorbs CUDA init/compile
            warm_frames += n
            warm_enc_s  += enc_s

        # Atomic write: tmp then rename, so a killed cell never leaves a corrupt .pt
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save({"embeddings": embeddings, "timestamps": timestamps, "vid": vid}, tmp)
        os.replace(tmp, out_path)
        done += 1

        # ── speed readout ─────────────────────────────────────────────────
        this_ms  = enc_s / max(n, 1) * 1000          # ms/frame for THIS video
        this_fps = n / max(enc_s, 1e-9)              # frames/sec for THIS video
        avg_ms   = (warm_enc_s / warm_frames * 1000) if warm_frames else this_ms
        avg_fps  = (warm_frames / warm_enc_s) if warm_enc_s else this_fps

        if verbose or done % 25 == 0 or idx == len(vids) - 1:
            elapsed   = time.time() - t0
            vid_rate  = done / max(elapsed, 1e-9)
            remaining = len(vids) - skipped - done
            eta_min   = (remaining / vid_rate) / 60 if vid_rate > 0 else float("inf")
            print(f"  [{idx + 1}/{len(vids)}] {n:3d} frames | "
                  f"this {this_ms:6.1f} ms/frame ({this_fps:5.1f} fps) | "
                  f"avg {avg_ms:6.1f} ms/frame ({avg_fps:5.1f} fps) | "
                  f"cached={done} ETA {eta_min:.1f} min")

        if args.max_hours is not None and (time.time() - t0) > args.max_hours * 3600:
            print(f"Hit --max_hours={args.max_hours}. Stopping cleanly "
                  f"({done} cached this run). Re-run to continue.")
            break

    total = done + skipped
    if total_frames:
        ov_ms  = total_enc_s / total_frames * 1000
        ov_fps = total_frames / total_enc_s
        wm_ms  = (warm_enc_s / warm_frames * 1000) if warm_frames else ov_ms
        wm_fps = (warm_frames / warm_enc_s) if warm_enc_s else ov_fps
        print(f"\nSpeed: encoded {total_frames} frames in {total_enc_s:.1f}s GPU time")
        print(f"  overall          : {ov_ms:6.1f} ms/frame  ({ov_fps:6.1f} fps)")
        print(f"  excluding warmup : {wm_ms:6.1f} ms/frame  ({wm_fps:6.1f} fps)  "
              f"← use this to extrapolate to the H100")
    print(f"\nDone. {done} newly cached, {skipped} already present "
          f"({total} total) → {cache_dir}")


if __name__ == "__main__":
    main()
