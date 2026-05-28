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
from huggingface_hub import login, HfApi, hf_hub_download, snapshot_download
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.pipeline.groundzero_model import GroundZeroModel
from app.training.dataset import GroundingDataset, collate_fn
from app.training.losses import combined_loss
from app.pipeline.span_extraction import decode_best_span, to_seconds


# ──────────────────────────────────────────────────────────────────────────────
# Dataset download
# ──────────────────────────────────────────────────────────────────────────────

DATASET_ID = "shaunmarvell/qvhighlights-1fps"

TRAIN_TARS = [
    "frames_001000_002000.tar",
    "frames_002000_003000.tar",
    "frames_003000_004000.tar",
    "frames_004000_005000.tar",
    "frames_005000_006000.tar",
    "frames_006000_007241.tar",
    "frames_007242_007445.tar",
]


def download_dataset(data_dir: Path) -> tuple:
    """
    Download QVHighlights from HuggingFace if not already present.
    Uses sentinel files so interrupted downloads can be resumed safely.
    Returns (train_jsonl, val_jsonl, frames_dir).
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

    # First 1 000 train vids stored as individual JPEGs (not tarred)
    sentinel = data_dir / ".first_batch_done"
    if sentinel.exists():
        print("  First 1 000 train vids: already present, skipping")
    else:
        print("  Downloading first 1 000 train vids (individual JPEGs)...")
        snapshot_download(
            DATASET_ID,
            repo_type="dataset",
            allow_patterns=["frames/**"],
            local_dir=str(data_dir),
            local_dir_use_symlinks=False,
        )
        sentinel.touch()
        print("  First batch done.")

    # Train tar files (vids 1 000–7 445)
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

def encode_batch(model, batch, device, d_model=1152):
    """
    Encode one batch of PIL frame lists + query strings → tensors.
    Must be called inside torch.cuda.amp.autocast() so SigLIP runs in fp16.

    Returns:
        frame_embs : (B, N_max, d_model)  float32, padded with zeros
        timestamps : (B, N_max)           fractional t/duration, padded 1.0
        query_emb  : (B, 1, d_model)      float32
    """
    B = len(batch["frames"])
    emb_list, ts_list = [], []

    for i in range(B):
        emb_i = model.visual_encoder.encode_frames(batch["frames"][i])   # (N_i, D)
        dur_i = batch["duration"][i].item()
        frac_i = torch.tensor(
            [t / dur_i for t in batch["timestamps"][i]],
            dtype=torch.float32,
            device=device,
        )
        emb_list.append(emb_i.float())
        ts_list.append(frac_i)

    N_max      = max(e.shape[0] for e in emb_list)
    frame_embs = torch.zeros(B, N_max, d_model, device=device)
    timestamps = torch.ones(B, N_max, device=device)   # 1.0 = end-of-video pad position

    for i, (emb, ts) in enumerate(zip(emb_list, ts_list)):
        n = emb.shape[0]
        frame_embs[i, :n] = emb
        timestamps[i, :n] = ts

    query_emb = model.text_encoder.encode_queries(batch["query"]).float().unsqueeze(1)  # (B,1,D)
    return frame_embs, timestamps, query_emb


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
):
    """
    Save two checkpoints every epoch:
      latest.pt — full training state (for resume). Overwrites each epoch.
      best.pt   — trainable params only (for inference). Overwrites on improvement.
    Both are pushed to HuggingFace.
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
        },
        latest_path,
    )
    print(f"  Saved latest.pt (ep {epoch}, R@1={r1_05:.4f})")
    _push(latest_path, hf_repo, api)

    # best.pt — only when val improved
    if is_best:
        best_path = ckpt_dir / "best.pt"
        torch.save(
            {
                "epoch":           epoch,
                "r1_iou05":        best_r1_05,
                "trainable_state": t_state,
            },
            best_path,
        )
        print(f"  Saved best.pt (R@1={best_r1_05:.4f}  ← new best)")
        _push(best_path, hf_repo, api)


def _push(ckpt_path: Path, hf_repo: str, api: HfApi):
    try:
        api.create_repo(hf_repo, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(ckpt_path),
            path_in_repo=ckpt_path.name,
            repo_id=hf_repo,
            repo_type="model",
        )
        print(f"  Pushed {ckpt_path.name} → hf:{hf_repo}")
    except Exception as exc:
        print(f"  HF push failed (non-fatal): {exc}")


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    hits_05 = hits_07 = total = 0
    val_loss_sum = 0.0

    for batch in val_loader:
        with torch.cuda.amp.autocast():
            frame_embs, timestamps, query_emb = encode_batch(model, batch, device)
            start_logits, end_logits, confidence = model(frame_embs, timestamps, query_emb)

        for i in range(start_logits.shape[0]):
            n_frames = len(batch["frames"][i])
            s_idx, e_idx = decode_best_span(start_logits[i], end_logits[i])
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
            with torch.cuda.amp.autocast():
                loss, *_ = combined_loss(
                    start_logits[pos], end_logits[pos],
                    batch["gt_start_idx"].to(device)[pos],
                    batch["gt_end_idx"].to(device)[pos],
                    query_emb[pos], frame_embs[pos],
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

    # Training
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--warmup_steps", type=int,   default=500)
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--log_every",    type=int,   default=50)
    p.add_argument("--max_hours",    type=float, default=None,
                   help="Stop cleanly after this many hours (e.g. 5.5 for a 6-hour session)")

    # Model
    p.add_argument("--lora_rank",  type=int,   default=8)
    p.add_argument("--lora_alpha", type=int,   default=16)
    p.add_argument("--n_layers",   type=int,   default=4)
    p.add_argument("--n_heads",    type=int,   default=8)
    p.add_argument("--dropout",    type=float, default=0.1)

    return p.parse_args()


def main():
    args = parse_args()

    # ── Auth ─────────────────────────────────────────────────────────────────
    if args.hf_token:
        login(token=args.hf_token)
    if args.wandb_key:
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
    else:
        train_jsonl, val_jsonl, frames_dir = download_dataset(data_dir)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = GroundingDataset(train_jsonl, frames_dir, augment=True)
    val_ds   = GroundingDataset(val_jsonl,   frames_dir, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True, persistent_workers=True,
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
    )
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
    total_steps = len(train_loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    scaler      = torch.cuda.amp.GradScaler()

    print(f"AdamW lr={args.lr}  |  cosine warmup={args.warmup_steps} / {total_steps} steps")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    global_step = 0
    best_r1_05  = 0.0

    resume_path = args.resume
    if resume_path is None:
        candidate = ckpt_dir / "latest.pt"
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
    else:
        print("\nStarting fresh (no checkpoint found)")

    # ── W&B ───────────────────────────────────────────────────────────────────
    run = wandb.init(
        project="groundzero",
        name=f"train-siglip2-lora-r{args.lora_rank}-bs{args.batch_size}",
        config=vars(args),
        resume="allow",
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
        epoch_loss = 0.0
        n_batches  = 0
        epoch_start = time.time()

        for batch in train_loader:

            # Intra-epoch time check: stop with 3% buffer so we have time to save
            if args.max_hours:
                elapsed_h = (time.time() - train_start) / 3600
                if elapsed_h >= args.max_hours * 0.97:
                    print(f"\nApproaching time limit — saving at step {global_step} and exiting.")
                    save_and_push(
                        model, optimizer, scheduler, scaler,
                        epoch, global_step, best_r1_05, 0.0,
                        ckpt_dir, args.hf_repo, api, is_best=False,
                    )
                    run.finish()
                    return

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                frame_embs, timestamps, query_emb = encode_batch(model, batch, DEVICE)
                start_logits, end_logits, confidence = model(frame_embs, timestamps, query_emb)

                gt_start = batch["gt_start_idx"].to(DEVICE)
                gt_end   = batch["gt_end_idx"].to(DEVICE)
                is_neg   = batch["is_negative"].to(DEVICE)
                pos_mask = ~is_neg

                if pos_mask.any():
                    total_l, span_l, iou_l, cont_l = combined_loss(
                        start_logits[pos_mask], end_logits[pos_mask],
                        gt_start[pos_mask], gt_end[pos_mask],
                        query_emb[pos_mask], frame_embs[pos_mask],
                    )
                else:
                    total_l = span_l = iou_l = cont_l = torch.tensor(0.0, device=DEVICE)

                conf_targets = torch.where(
                    is_neg, torch.zeros_like(confidence), torch.ones_like(confidence)
                )
                conf_l = F.binary_cross_entropy(confidence, conf_targets)
                loss   = total_l + conf_l

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.clip_grad
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

            if global_step % args.log_every == 0:
                metrics = {
                    "train/loss":      loss.item(),
                    "train/span_loss": span_l.item(),
                    "train/iou_loss":  iou_l.item(),
                    "train/cont_loss": cont_l.item(),
                    "train/conf_loss": conf_l.item(),
                    "train/lr":        scheduler.get_last_lr()[0],
                }
                wandb.log(metrics, step=global_step)
                print(
                    f"ep {epoch:02d}  step {global_step:05d}  "
                    f"loss={loss.item():.4f}  span={span_l.item():.4f}  "
                    f"iou={iou_l.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )

        epoch_mins = (time.time() - epoch_start) / 60
        avg_loss   = epoch_loss / max(n_batches, 1)
        print(f"\n--- Epoch {epoch:02d}  avg_loss={avg_loss:.4f}  time={epoch_mins:.1f} min ---")

        # ── Validate ──────────────────────────────────────────────────────────
        print("  Validating...")
        vm = validate(model, val_loader, DEVICE)
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

        save_and_push(
            model, optimizer, scheduler, scaler,
            epoch, global_step, best_r1_05, vm["val/r1_iou05"],
            ckpt_dir, args.hf_repo, api, is_best=is_best,
        )
        print()

    run.finish()
    print(f"\nDone. Best R@1 IoU=0.5: {best_r1_05:.4f}")


if __name__ == "__main__":
    main()
