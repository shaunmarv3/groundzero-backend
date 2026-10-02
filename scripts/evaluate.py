#!/usr/bin/env python3
"""
evaluate.py — Phase 6 evaluation suite (metrics computable from the cached val set).

Runs INFERENCE ONLY on an already-trained checkpoint (best.pt). No training, no
SigLIP vision tower (uses --use_cache-style cached embeddings). Everything here
scores the existing model — nothing is retrained.

Computes:
  6.1 Primary metrics : R@1 IoU=0.5, R@1 IoU=0.7, R@5 IoU=0.5, median center displacement (s)
  6.3 Query analysis  : R@1 IoU=0.5 bucketed by query token-length
  6.7 Calibration     : ECE (10-bin) of the confidence head, + R@1@0.5 in the low-conf (<0.4) bucket

WHY no train.py import: train.py does a top-level `import wandb`, which isn't always
installed on a fresh eval box. We import the model/dataset/decoders straight from app/
(batch building is shared with train.py via app/training/batch_encoding.py) and
re-implement only temporal_iou locally.

The architecture is read from the checkpoint's "model_cfg" (GELU / word-level /
padding mask / lowercase). The shipped best.pt has none → original design.
--text_cache uses the pre-encoded queries → no SigLIP loaded (fits a 4 GB GPU).

EXAMPLE (Kaggle / any GPU box, after pulling the cache tar + best.pt):
    python scripts/evaluate.py \
        --data_dir ~/data/qvhighlights \
        --hf_repo  shaunmarvell/qvhighlights-model \
        --ckpt_name best.pt
"""

import argparse
import os
import sys
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.groundzero_model import GroundZeroModel, resolve_model_cfg
from app.training.batch_encoding import TextCache, encode_cached_batch
from app.pipeline.span_extraction import decode_best_span, to_seconds
from app.training.dataset import CachedGroundingDataset, cached_collate_fn


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers (re-implemented here to avoid importing train.py / wandb)
# ──────────────────────────────────────────────────────────────────────────────

def temporal_iou(s1, e1, s2, e2):
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0


def decode_topk_spans(start_logits, end_logits, k=5, nms_iou=0.5):
    """
    Top-k valid spans (end>=start) ranked by start+end score, with greedy NMS so the
    k spans aren't near-duplicates. Returns list of (start_idx, end_idx) ints.
    Used for R@5 (a hit if ANY of the top-k clears the IoU bar).
    """
    N = start_logits.shape[0]
    score = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)          # (N,N)
    mask  = torch.triu(torch.ones(N, N, device=start_logits.device))     # j >= i
    score = score * mask + (1 - mask) * (-1e9)

    flat = score.flatten()
    order = torch.argsort(flat, descending=True)                          # best first
    kept = []
    for idx in order.tolist():
        i, j = idx // N, idx % N
        if score[i, j].item() <= -1e8:
            break
        # NMS against already-kept spans (frame-index IoU)
        if all(temporal_iou(i, j, ks, ke) <= nms_iou for ks, ke in kept):
            kept.append((i, j))
            if len(kept) == k:
                break
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GroundZero Phase-6 eval (inference only).")
    ap.add_argument("--data_dir",  required=True, help="Dir with annotations_val.jsonl and cache/")
    ap.add_argument("--cache_dir", default=None,  help="Cached {vid}.pt dir (default <data_dir>/cache)")
    ap.add_argument("--val_jsonl", default=None,  help="(default <data_dir>/annotations_val.jsonl)")
    ap.add_argument("--ckpt",      default=None,  help="Local path to best.pt (overrides --hf_repo)")
    ap.add_argument("--hf_repo",   default=None,  help="HF model repo to pull --ckpt_name from")
    ap.add_argument("--ckpt_name", default="best.pt")
    ap.add_argument("--hf_token",  default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=2)
    # Architecture — MUST match training (train.py defaults; the run used defaults)
    ap.add_argument("--d_model",  type=int, default=1152)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads",  type=int, default=8)
    ap.add_argument("--dropout",  type=float, default=0.1)
    ap.add_argument("--lora_rank",  type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--text_cache", action="store_true",
                    help="Use text_pooled*/text_tokens*.pt instead of the live SigLIP text tower "
                         "(no SigLIP loaded). Required for word-level checkpoints.")
    args = ap.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir  = Path(args.data_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (data_dir / "cache")
    val_jsonl = Path(args.val_jsonl) if args.val_jsonl else (data_dir / "annotations_val.jsonl")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    ckpt_path = args.ckpt
    if ckpt_path is None:
        if not args.hf_repo:
            sys.exit("ERROR: pass --ckpt <path> or --hf_repo <repo> (+ --ckpt_name).")
        from huggingface_hub import hf_hub_download, login
        if args.hf_token:
            login(token=args.hf_token)
        print(f"Downloading {args.ckpt_name} from {args.hf_repo} ...")
        ckpt_path = hf_hub_download(args.hf_repo, args.ckpt_name, repo_type="model")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint: {ckpt_path}  (epoch={ckpt.get('epoch')}, "
          f"saved R@1@0.5={ckpt.get('r1_iou05')})")
    cfg = resolve_model_cfg(ckpt)
    print(f"model_cfg: {cfg}")
    if cfg["word_level"] and not args.text_cache:
        sys.exit("ERROR: word-level checkpoint — pass --text_cache (word tokens come from text_tokens*.pt).")

    # ── Model (cache mode → no vision tower) ────────────────────────────────────
    print("Building model + loading trainable weights...")
    model = GroundZeroModel(
        model_id="google/siglip2-so400m-patch14-384",
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_layers=[23, 24, 25, 26], dropout=args.dropout,
        device=device, use_cache=True,
        use_gelu=cfg["use_gelu"], word_level=cfg["word_level"],
        load_text_encoder=not args.text_cache,
    )
    text_cache = (TextCache(data_dir, lowercase=cfg["lowercase"], word_level=cfg["word_level"],
                            token=args.hf_token or None)
                  if args.text_cache else None)
    missing, unexpected = model.load_state_dict(ckpt["trainable_state"], strict=False)
    # 'missing' will list the frozen text-tower keys (loaded from pretrained) — expected.
    loaded = len(ckpt["trainable_state"])
    print(f"  Loaded {loaded} trainable tensors "
          f"({len(unexpected)} unexpected — should be 0).")
    if unexpected:
        print(f"  WARNING unexpected keys (arch mismatch?): {list(unexpected)[:5]}")
    model.eval().to(device)

    # ── Data ────────────────────────────────────────────────────────────────────
    ds = CachedGroundingDataset(val_jsonl, cache_dir, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=cached_collate_fn, num_workers=args.num_workers)
    print(f"Val samples: {len(ds)}  /  {len(loader)} batches\n")

    # ── Accumulators ─────────────────────────────────────────────────────────────
    hit1_05 = hit1_07 = hit5_05 = total = 0
    center_err = []                       # |pred_center - gt_center| seconds
    by_len = {}                           # token-length bin → [hits, total] at IoU 0.5
    conf_correct = []                     # (confidence, correct@0.5) for ECE
    LEN_BINS = [(1, 5), (6, 10), (11, 15), (16, 999)]

    def len_bin(q):
        L = len(q.split())
        for lo, hi in LEN_BINS:
            if lo <= L <= hi:
                return f"{lo}-{hi if hi < 999 else '+'}"
        return "?"

    # ── Eval loop ────────────────────────────────────────────────────────────────
    with torch.no_grad():
        for batch in loader:
            with torch.cuda.amp.autocast():
                frame_embs, timestamps, frame_mask, query, query_mask, _ = encode_cached_batch(
                    model, batch, device, text_cache=text_cache, lowercase=cfg["lowercase"])
                start_logits, end_logits, confidence = model(
                    frame_embs, timestamps, query,
                    frame_mask=frame_mask if cfg["mask_padding"] else None,
                    query_mask=query_mask,
                )

            for i in range(start_logits.shape[0]):
                ts_i   = batch["timestamps"][i]
                n      = len(ts_i)
                dur_i  = batch["duration"][i].item()
                query  = batch["query"][i]

                # GT seconds (from GT frame indices stored by the dataset)
                gt_s = ts_i[min(batch["gt_start_idx"][i].item(), n - 1)]
                gt_e = ts_i[min(batch["gt_end_idx"][i].item(),   n - 1)]

                sl = start_logits[i, :n]
                el = end_logits[i, :n]

                # top-1
                s_idx, e_idx = decode_best_span(sl, el)
                s_idx, e_idx = min(s_idx, n - 1), min(e_idx, n - 1)
                pred_s, pred_e = to_seconds(s_idx, e_idx, n, dur_i)
                iou1 = temporal_iou(pred_s, pred_e, gt_s, gt_e)

                # top-5 (R@5: best IoU among top-k)
                topk = decode_topk_spans(sl, el, k=5, nms_iou=0.5)
                iou5 = max(
                    (temporal_iou(*to_seconds(ks, ke, n, dur_i), gt_s, gt_e) for ks, ke in topk),
                    default=0.0,
                )

                correct_05 = iou1 >= 0.5
                hit1_05 += correct_05
                hit1_07 += iou1 >= 0.7
                hit5_05 += iou5 >= 0.5
                total   += 1

                # center displacement (seconds)
                pred_c = (pred_s + pred_e) / 2
                gt_c   = (gt_s + gt_e) / 2
                center_err.append(abs(pred_c - gt_c))

                # query-length bucket
                b = len_bin(query)
                by_len.setdefault(b, [0, 0])
                by_len[b][0] += int(correct_05)
                by_len[b][1] += 1

                # calibration
                conf_correct.append((float(confidence[i].item()), int(correct_05)))

    # ── 6.1 Primary metrics ───────────────────────────────────────────────────────
    print("=" * 60)
    print("6.1  PRIMARY METRICS (QVHighlights val)")
    print("-" * 60)
    print(f"  R@1 IoU=0.5 : {hit1_05 / total:.4f}")
    print(f"  R@1 IoU=0.7 : {hit1_07 / total:.4f}")
    print(f"  R@5 IoU=0.5 : {hit5_05 / total:.4f}")
    print(f"  Median center displacement : {statistics.median(center_err):.2f} s  (target <8s)")
    print(f"  Mean   center displacement : {statistics.mean(center_err):.2f} s")
    print(f"  (n = {total} queries)")

    # ── 6.3 Query-length analysis ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6.3  R@1 IoU=0.5 BY QUERY TOKEN-LENGTH")
    print("-" * 60)
    for lo, hi in LEN_BINS:
        b = f"{lo}-{hi if hi < 999 else '+'}"
        if b in by_len:
            h, t = by_len[b]
            print(f"  {b:>5} tokens : {h / t:.4f}  (n={t})")

    # ── 6.7 Confidence calibration (ECE) ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6.7  CONFIDENCE CALIBRATION")
    print("-" * 60)
    n_bins = 10
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bucket = [(c, ok) for c, ok in conf_correct if (lo <= c < hi or (b == n_bins - 1 and c == 1.0))]
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc      = sum(ok for _, ok in bucket) / len(bucket)
        ece     += (len(bucket) / total) * abs(avg_conf - acc)
    print(f"  ECE (10-bin) : {ece:.4f}  (target <0.10)")

    low = [(c, ok) for c, ok in conf_correct if c < 0.4]
    if low:
        acc_low = sum(ok for _, ok in low) / len(low)
        print(f"  R@1 IoU=0.5 for confidence<0.4 : {acc_low:.4f}  (n={len(low)}; "
              f"target <0.20 → justifies a rejection threshold)")
    else:
        print("  (no predictions with confidence<0.4)")
    print("=" * 60)


if __name__ == "__main__":
    main()
