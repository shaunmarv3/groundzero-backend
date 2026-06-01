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
from torch.utils.data import Dataset, DataLoader
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
# Frame loading (workers) + GPU encode (main thread)
# ──────────────────────────────────────────────────────────────────────────────

class VideoFrameDataset(Dataset):
    """
    One item = one video, fully decoded + preprocessed on a DataLoader WORKER.

    The JPEG read + PIL decode + processor resize/normalise are pure CPU work and,
    in the naive serial loop, the GPU sat idle (~40% util) waiting for them. Doing
    this here lets num_workers decode videos N+1, N+2… in parallel while the GPU
    encodes video N — overlapping I/O with compute and keeping the GPU fed.

    Returns (vid, pixel_values (N,3,384,384) fp32, timestamps (N,) fp32).
    """

    def __init__(self, vids, frames_root, processor):
        self.vids        = vids
        self.frames_root = Path(frames_root)
        self.processor   = processor

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, i):
        vid        = self.vids[i]
        jpg_files  = sorted((self.frames_root / vid).glob("*.jpg"))
        timestamps = torch.tensor([float(p.stem) for p in jpg_files], dtype=torch.float32)
        imgs       = [Image.open(p).convert("RGB") for p in jpg_files]
        # processor → resize 384 + normalise → (N, 3, 384, 384) pixel tensor (CPU)
        pixel_values = self.processor(images=imgs, return_tensors="pt")["pixel_values"]
        return vid, pixel_values, timestamps


@torch.no_grad()
def encode_pixel_values(model, pixel_values, device, chunk_size):
    """
    Encode preprocessed (N, 3, 384, 384) pixel tensors → (N, 1152) fp16 on CPU.
    Runs on the main thread / GPU; chunked to bound activation memory.
    """
    feats = []
    for i in range(0, pixel_values.shape[0], chunk_size):
        pv  = pixel_values[i : i + chunk_size].to(device, non_blocking=True).half()
        out = model.get_image_features(pixel_values=pv)
        # transformers >=4.50 returns BaseModelOutputWithPooling, not a bare tensor —
        # pull the pooled (chunk, 1152) embedding out either way (same as encode_frames).
        emb = out if isinstance(out, torch.Tensor) else out.pooler_output
        feats.append(emb.half().cpu())
    return torch.cat(feats, dim=0)  # (N, 1152) fp16


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
    ap.add_argument("--chunk_size", type=int, default=32,
                    help="Frames encoded per forward. Higher is fine here (no backward, "
                         "no grad-ckpt) — raise it on big GPUs to go faster.")
    ap.add_argument("--num_workers", type=int, default=3,
                    help="DataLoader workers decoding JPEGs in parallel with GPU encode. "
                         "2-3 hides decode behind compute on T4; 0 = serial (debug).")
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

    # Skip already-cached vids UP FRONT so workers never waste time decoding them
    # (resumable: a stopped run just re-runs and continues from here).
    todo    = [v for v in vids if not (cache_dir / f"{v}.pt").exists()]
    skipped = len(vids) - len(todo)
    print(f"Found {len(vids)} unique vids on disk — {skipped} already cached, "
          f"{len(todo)} to encode.")
    if not todo:
        print(f"Nothing to do → {cache_dir}")
        return

    # ── Load FROZEN SigLIP vision tower (no LoRA, no grad-ckpt, fp16) ─────────
    print(f"Loading {args.model_id} (frozen, fp16)...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.float16)
    model.eval().to(device)

    # ── DataLoader: workers decode + preprocess while the GPU encodes ─────────
    loader = DataLoader(
        VideoFrameDataset(todo, frames_root, processor),
        batch_size=1,                                   # one video per item (variable N)
        num_workers=args.num_workers,
        collate_fn=lambda b: b[0],                      # unwrap the single-item batch
        prefetch_factor=(2 if args.num_workers > 0 else None),
        pin_memory=True,
    )

    # ── Encode loop ────────────────────────────────────────────────────────────
    t0 = time.time()
    done = 0
    total_frames = 0      # frames encoded this run
    total_enc_s  = 0.0    # GPU-only encode seconds (sync'd) — the compute floor
    verbose = bool(args.limit) or len(todo) <= 20

    for vid, pixel_values, timestamps in loader:
        # Time the GPU encode separately (sync'd both sides) so we can compare it to
        # wall-clock and SEE how much of the time the GPU is actually busy.
        if device == "cuda":
            torch.cuda.synchronize()
        t_enc = time.time()
        embeddings = encode_pixel_values(model, pixel_values, device, args.chunk_size)
        if device == "cuda":
            torch.cuda.synchronize()
        total_enc_s += time.time() - t_enc

        n = embeddings.shape[0]
        total_frames += n

        # Atomic write: tmp then rename, so a killed cell never leaves a corrupt .pt
        out_path = cache_dir / f"{vid}.pt"
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save({"embeddings": embeddings, "timestamps": timestamps, "vid": vid}, tmp)
        os.replace(tmp, out_path)
        done += 1

        # ── speed readout ──────────────────────────────────────────────────
        # wall = real throughput (decode overlapped); gpu = compute floor;
        # busy% = total_enc / wall = how well the workers keep the GPU fed.
        elapsed  = time.time() - t0
        wall_ms  = elapsed / max(total_frames, 1) * 1000
        wall_fps = total_frames / max(elapsed, 1e-9)
        gpu_ms   = total_enc_s / max(total_frames, 1) * 1000
        gpu_busy = total_enc_s / max(elapsed, 1e-9) * 100

        if verbose or done % 25 == 0 or done == len(todo):
            eta_min = (len(todo) - done) / max(done / max(elapsed, 1e-9), 1e-9) / 60
            print(f"  [{done}/{len(todo)}] {n:3d} frames | "
                  f"wall {wall_ms:5.1f} ms/frame ({wall_fps:5.1f} fps) | "
                  f"gpu {gpu_ms:5.1f} ms/frame busy {gpu_busy:3.0f}% | "
                  f"ETA {eta_min:.1f} min")

        if args.max_hours is not None and (time.time() - t0) > args.max_hours * 3600:
            print(f"Hit --max_hours={args.max_hours}. Stopping cleanly "
                  f"({done} cached this run). Re-run to continue.")
            break

    if total_frames:
        elapsed  = time.time() - t0
        wall_ms  = elapsed / total_frames * 1000
        wall_fps = total_frames / elapsed
        gpu_ms   = total_enc_s / total_frames * 1000
        gpu_busy = total_enc_s / max(elapsed, 1e-9) * 100
        print(f"\nSpeed: {total_frames} frames in {elapsed:.1f}s wall ({total_enc_s:.1f}s on GPU)")
        print(f"  wall-clock : {wall_ms:6.1f} ms/frame  ({wall_fps:6.1f} fps)  "
              f"← real throughput; use this to extrapolate")
        print(f"  gpu-only   : {gpu_ms:6.1f} ms/frame   (compute floor, GPU busy {gpu_busy:.0f}% "
              f"of wall — closer to 100% = decode fully hidden)")
    print(f"\nDone. {done} newly cached, {skipped} already present → {cache_dir}")


if __name__ == "__main__":
    main()
