#!/usr/bin/env python3
"""
precompute_text_embeddings.py — text-side counterpart of precompute_embeddings.py.

WHY THIS EXISTS
───────────────
In --use_cache training the frame embeddings come from disk, but every step still
runs the frozen SigLIP 2 text tower live (encode_batch_cached → encode_queries).
TextEncoder loads the WHOLE SigLIP AutoModel in fp32 (~4.5 GB) for that, which
does not fit a 4 GB RTX 3050.

The query set is finite and fixed: train.py never passes a paraphrase file, and
negative-query injection only swaps in queries from the same annotation set. The
text tower is frozen, so a query's embedding is identical every epoch — the same
argument that justified caching the frames. Encode every unique query ONCE here,
and training needs no SigLIP at all.

WHAT IT WRITES
──────────────
    text_pooled{sfx}.pt = {"queries": [str] (Q,), "pooled": (Q, 1152) fp16, ...}
        One vector per query — exactly what the current model consumes.
        Used for: contrastive-wiring fix, GELU, padding mask (fixes 1, 2, 4).

    text_tokens{sfx}.pt = {"queries": [str] (Q,), "tokens": (Q, L, 1152) fp16,
                           "mask": (Q, L) bool, ...}
        One vector per TOKEN (L = 64 for SigLIP 2) + which positions are real
        tokens vs padding. Used for: word-level query (fix 3, QD-DETR style).

    sfx = "" by default, "_lower" with --lowercase.

`queries` holds the ORIGINAL annotation strings, so the dataset can look a query
up by the exact string it reads from the jsonl, whichever variant was encoded.

MIRRORS TRAINING EXACTLY (default, no --lowercase)
──────────────────────────────────────────────────
Same processor call as TextEncoder.encode_queries (padding="max_length",
truncation=True), fp32 weights, forward under fp16 autocast — the way
encode_batch_cached ran inside the autocast block of train.py / evaluate.py.
So text_pooled.pt reproduces the inputs best.pt was trained on, and fix 1 can
fine-tune from best.pt.

--lowercase: SigLIP 2 was trained on lowercased text (HF Siglip2 docs), but the
QVHighlights queries are cased ("A woman is looking...") and the shipped model
never lowercased them. --lowercase encodes q.lower() instead — for a from-scratch
retrain, to measure whether matching the pretraining text format helps.

MULTI-GPU
─────────
One process per visible GPU (torch.multiprocessing.spawn). Query i goes to GPU
i % world_size, so each GPU gets Q / world_size queries (±1). Every query is
padded to the same length, so equal counts = equal work. Each worker writes a
part file; the main process merges them back into the original order.

EXAMPLE (Kaggle, accelerator "GPU T4 x2")
─────────────────────────────────────────
    python scripts/precompute_text_embeddings.py \
        --out_dir /kaggle/working/text_cache \
        --push_repo shaunmarvell/qvhighlights-1fps

    # smoke test first: 64 queries, no upload
    python scripts/precompute_text_embeddings.py --out_dir /kaggle/working/tc_test --limit 64
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

MODEL_ID     = "google/siglip2-so400m-patch14-384"
DATASET_REPO = "shaunmarvell/qvhighlights-1fps"


# ──────────────────────────────────────────────────────────────────────────────
# Collect the unique queries
# ──────────────────────────────────────────────────────────────────────────────

def load_unique_queries(data_dir: Path | None, token: str | None) -> list:
    """
    Unique query strings from annotations_train.jsonl + annotations_val.jsonl,
    in first-appearance order (deterministic). Pulls the two small jsonl files
    from the HF dataset repo when --data_dir isn't given.
    """
    paths = []
    for name in ("annotations_train.jsonl", "annotations_val.jsonl"):
        if data_dir is not None:
            paths.append(data_dir / name)
        else:
            from huggingface_hub import hf_hub_download
            paths.append(Path(hf_hub_download(DATASET_REPO, name, repo_type="dataset", token=token)))

    seen = {}                                   # dict keeps insertion order
    n_rows = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.setdefault(json.loads(line)["query"], None)
                    n_rows += 1
    queries = list(seen)
    print(f"Annotations: {n_rows} rows → {len(queries)} unique queries")
    return queries


# ──────────────────────────────────────────────────────────────────────────────
# Worker — one per GPU
# ──────────────────────────────────────────────────────────────────────────────

def encode_shard(rank: int, world_size: int, queries: list, args) -> None:
    """
    Encode queries[rank::world_size] on cuda:rank and save a part file.
    Called by mp.spawn (rank = 0..world_size-1) or directly with rank=0 when
    there is at most one GPU.
    """
    from transformers import AutoModel, AutoProcessor
    sys.stdout.reconfigure(line_buffering=True)   # spawned workers: print lines immediately

    use_cuda = torch.cuda.is_available()
    device   = f"cuda:{rank}" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.set_device(device)

    idx   = list(range(rank, len(queries), world_size))   # this GPU's share
    texts = [queries[i].lower() if args.lowercase else queries[i] for i in idx]

    # Same load as TextEncoder (cache mode): full AutoModel, fp32, frozen.
    processor = AutoProcessor.from_pretrained(args.model_id)
    model     = AutoModel.from_pretrained(args.model_id).eval().to(device)
    pad_id    = processor.tokenizer.pad_token_id

    pooled, tokens, masks = [], [], []
    n_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    print(f"  [rank {rank}] model loaded on {device} — {len(texts)} queries / {n_batches} batches", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for bi, b in enumerate(range(0, len(texts), args.batch_size), start=1):
            chunk  = texts[b : b + args.batch_size]
            # identical to TextEncoder.encode_queries
            inputs = processor(
                text=chunk, return_tensors="pt", padding="max_length", truncation=True,
            ).to(device)

            ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else torch.autocast("cpu", enabled=False)
            with ctx:
                out = model.get_text_features(**inputs)
                p   = out if isinstance(out, torch.Tensor) else out.pooler_output    # (b, 1152)
                h   = model.text_model(**inputs).last_hidden_state                    # (b, L, 1152)

            # real-token mask: prefer the tokenizer's own mask, else compare to pad id
            if "attention_mask" in inputs:
                m = inputs["attention_mask"].bool()
            else:
                m = inputs["input_ids"] != pad_id                                     # (b, L)

            pooled.append(p.half().cpu())
            tokens.append(h.half().cpu())
            masks.append(m.cpu())

            # progress + time left for this GPU (avg time per batch so far × batches left)
            if bi % args.log_every == 0 or bi == n_batches:
                elapsed = time.time() - t0
                eta     = elapsed / bi * (n_batches - bi)
                print(f"  [rank {rank}] batch {bi}/{n_batches} ({bi / n_batches:5.1%}) | "
                      f"{min(b + args.batch_size, len(texts))}/{len(texts)} queries | "
                      f"elapsed {elapsed / 60:4.1f} min | ETA {eta / 60:4.1f} min", flush=True)

    part = {
        "idx":    torch.tensor(idx, dtype=torch.long),
        "pooled": torch.cat(pooled),
        "tokens": torch.cat(tokens),
        "mask":   torch.cat(masks),
    }
    torch.save(part, Path(args.out_dir) / f"part_{rank}.pt")

    peak = torch.cuda.max_memory_allocated(device) / 1e9 if use_cuda else 0.0
    name = torch.cuda.get_device_name(device) if use_cuda else "cpu"
    print(f"  [rank {rank} | {name}] {len(idx)} queries in {time.time() - t0:.1f}s | "
          f"peak VRAM {peak:.2f} GB | tokens {tuple(part['tokens'].shape)}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Merge + verify + save
# ──────────────────────────────────────────────────────────────────────────────

def merge_parts(out_dir: Path, world_size: int, n_queries: int):
    parts = [torch.load(out_dir / f"part_{r}.pt") for r in range(world_size)]
    d     = parts[0]["pooled"].shape[1]
    L     = parts[0]["tokens"].shape[1]

    pooled = torch.zeros(n_queries, d, dtype=torch.float16)
    tokens = torch.zeros(n_queries, L, d, dtype=torch.float16)
    mask   = torch.zeros(n_queries, L, dtype=torch.bool)
    filled = torch.zeros(n_queries, dtype=torch.bool)

    for p in parts:
        pooled[p["idx"]] = p["pooled"]
        tokens[p["idx"]] = p["tokens"]
        mask[p["idx"]]   = p["mask"]
        filled[p["idx"]] = True

    # ── verify before anything gets uploaded ─────────────────────────────────
    n_missing = int((~filled).sum())
    n_nonfin  = int((~torch.isfinite(pooled.float())).any(dim=1).sum()
                    + (~torch.isfinite(tokens.float())).any(dim=(1, 2)).sum())
    n_empty   = int((mask.sum(dim=1) == 0).sum())
    real_len  = mask.sum(dim=1).float()
    print(f"\nVerify: pooled {tuple(pooled.shape)} | tokens {tuple(tokens.shape)} | mask {tuple(mask.shape)}")
    print(f"  missing rows: {n_missing} | non-finite rows: {n_nonfin} | queries with 0 real tokens: {n_empty}")
    print(f"  real tokens per query: min {int(real_len.min())} / mean {real_len.mean():.1f} / "
          f"max {int(real_len.max())} of L={L}  "
          f"({int((real_len == L).sum())} queries hit the limit → truncated)")
    if n_missing or n_nonfin or n_empty:
        raise SystemExit("Verification FAILED — not saving / not uploading.")
    return pooled, tokens, mask


def atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    print(f"  wrote {path.name}  {path.stat().st_size / 1e9:.2f} GB")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pre-encode every query with the frozen SigLIP 2 text tower.")
    ap.add_argument("--out_dir",    type=str, required=True)
    ap.add_argument("--data_dir",   type=str, default=None,
                    help="Dir with annotations_{train,val}.jsonl. Default: download them from HF.")
    ap.add_argument("--model_id",   type=str, default=MODEL_ID)
    ap.add_argument("--batch_size", type=int, default=64, help="Queries per forward, per GPU.")
    ap.add_argument("--lowercase",  action="store_true",
                    help="Encode q.lower() (SigLIP 2's training format). Writes *_lower.pt.")
    ap.add_argument("--limit",      type=int, default=None, help="Only the first N queries (smoke test).")
    ap.add_argument("--log_every",  type=int, default=5, help="Print progress + ETA every N batches per GPU.")
    ap.add_argument("--hf_token",   type=str, default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--push_repo",  type=str, default=None,
                    help=f"Upload both files to this HF dataset repo (e.g. {DATASET_REPO}).")
    args = ap.parse_args()
    # Kaggle's `!python` pipes stdout (block-buffered) — flush every line so progress shows live.
    sys.stdout.reconfigure(line_buffering=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = load_unique_queries(Path(args.data_dir) if args.data_dir else None, args.hf_token)
    if args.limit:
        queries = queries[: args.limit]

    # Download the model once here, so the GPU workers don't race on the HF cache.
    from huggingface_hub import snapshot_download
    snapshot_download(args.model_id, token=args.hf_token)

    world_size = max(torch.cuda.device_count(), 1)
    print(f"Encoding {len(queries)} queries on {world_size} device(s) "
          f"(lowercase={args.lowercase}, batch {args.batch_size}/GPU)...")
    t0 = time.time()
    if world_size > 1:
        mp.spawn(encode_shard, args=(world_size, queries, args), nprocs=world_size, join=True)
    else:
        encode_shard(0, 1, queries, args)
    print(f"All shards done in {time.time() - t0:.1f}s")

    pooled, tokens, mask = merge_parts(out_dir, world_size, len(queries))

    sfx  = "_lower" if args.lowercase else ""
    meta = {"model_id": args.model_id, "lowercase": args.lowercase,
            "padding": "max_length", "autocast": "fp16", "dtype": "fp16"}
    pooled_path = out_dir / f"text_pooled{sfx}.pt"
    tokens_path = out_dir / f"text_tokens{sfx}.pt"
    print()
    atomic_save({"queries": queries, "pooled": pooled, **meta}, pooled_path)
    atomic_save({"queries": queries, "tokens": tokens, "mask": mask, **meta}, tokens_path)
    for r in range(world_size):
        (out_dir / f"part_{r}.pt").unlink(missing_ok=True)

    if args.push_repo:
        from huggingface_hub import HfApi
        api = HfApi(token=args.hf_token)
        for path in (pooled_path, tokens_path):
            print(f"  uploading {path.name} → {args.push_repo} (dataset) ...")
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                            repo_id=args.push_repo, repo_type="dataset")
        print("Upload done.")


if __name__ == "__main__":
    main()
