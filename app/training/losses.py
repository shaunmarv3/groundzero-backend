"""
losses.py — Training loss functions for GroundZero.
Phase 4.1

Three losses combined during training:
    total = span_loss + 0.5 * iou_loss + 0.1 * contrastive_loss

Each is returned separately so the training loop can log them to W&B.
"""

import random
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Loss 1 — Span Extraction Loss (primary)
# ---------------------------------------------------------------------------

def span_extraction_loss(
    start_logits: torch.Tensor,  # (B, N)
    end_logits: torch.Tensor,    # (B, N)
    gt_start_idx: torch.Tensor,  # (B,) long — ground truth start frame index
    gt_end_idx: torch.Tensor,    # (B,) long — ground truth end frame index
) -> torch.Tensor:
    """
    Cross-entropy over frame positions for start and end boundaries.
    Same formulation as extractive QA (BERT SQuAD): 'which frame is the start?'
    is a classification problem over N frames, not a regression.
    """
    start_loss = F.cross_entropy(start_logits, gt_start_idx)
    end_loss   = F.cross_entropy(end_logits,   gt_end_idx)
    return start_loss + end_loss


# ---------------------------------------------------------------------------
# Loss 2 — Temporal IoU Loss (auxiliary)
# ---------------------------------------------------------------------------

def _soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Differentiable position estimate: softmax-weighted average of frame indices.

    Why not hard argmax: argmax has zero gradient everywhere — no learning signal.
    Soft argmax produces a continuous expected position that backprop can flow through.

    Input:  (B, N)
    Output: (B,) float — expected frame index for each sample
    """
    B, N = logits.shape
    indices = torch.arange(N, dtype=logits.dtype, device=logits.device)  # (N,)
    probs = torch.softmax(logits, dim=-1)                                  # (B, N)
    return (probs * indices).sum(dim=-1)                                   # (B,)


def temporal_iou_loss(
    start_logits: torch.Tensor,  # (B, N)
    end_logits: torch.Tensor,    # (B, N)
    gt_start_idx: torch.Tensor,  # (B,) long
    gt_end_idx: torch.Tensor,    # (B,) long
) -> torch.Tensor:
    """
    1 - IoU between predicted span and ground truth span. Always in [0, 1].
    Differentiable via soft argmax — gradients flow back through the logits.
    Complements span_extraction_loss by directly optimising the evaluation metric.
    """
    pred_start = _soft_argmax(start_logits)  # (B,) float
    pred_end   = _soft_argmax(end_logits)    # (B,) float
    gt_start   = gt_start_idx.float()        # (B,) float
    gt_end     = gt_end_idx.float()          # (B,) float

    intersection = torch.clamp(
        torch.min(pred_end, gt_end) - torch.max(pred_start, gt_start), min=0.0
    )
    union = torch.max(pred_end, gt_end) - torch.min(pred_start, gt_start)
    iou   = intersection / (union + 1e-8)
    return (1.0 - iou).mean()


# ---------------------------------------------------------------------------
# Loss 3 — Intra-Video Contrastive Loss
# ---------------------------------------------------------------------------

def _sample_non_overlapping_segments(
    n_frames: int,
    gt_start: int,
    gt_end: int,
    n_negatives: int = 8,
    min_len: int = 3,
) -> list:
    """
    Randomly sample n_negatives (start, end) frame index pairs that do NOT
    overlap with [gt_start, gt_end]. Falls back gracefully when GT spans most
    of the video.

    Why intra-video negatives: with batch_size=2 we only get 1 in-batch negative
    (too few for InfoNCE to learn from). Mining 8 hard negatives from the same
    video works regardless of batch size and provides harder negatives
    (same visual domain, wrong temporal location).
    """
    segments = []
    max_attempts = n_negatives * 100
    attempts = 0

    while len(segments) < n_negatives and attempts < max_attempts:
        attempts += 1
        max_start = max(0, n_frames - min_len)
        s = random.randint(0, max_start)
        max_length = max(min_len, n_frames // 4)
        e = random.randint(s + min_len - 1, min(s + max_length, n_frames - 1))
        if e < gt_start or s > gt_end:  # no overlap with GT
            segments.append((s, e))

    # Fallback: pad from video boundaries if random search found too few
    while len(segments) < n_negatives:
        if gt_start >= min_len:
            s = random.randint(0, max(0, gt_start - min_len))
            segments.append((s, s + min_len - 1))
        elif gt_end + min_len < n_frames:
            s = gt_end + 1
            segments.append((s, min(s + min_len - 1, n_frames - 1)))
        else:
            # GT covers the whole video — degenerate fallback
            segments.append((0, min(min_len - 1, n_frames - 1)))

    return segments[:n_negatives]


def contrastive_loss_intra_video(
    query_emb: torch.Tensor,          # (B, 1, d)
    grounded_features: torch.Tensor,  # (B, N, d)
    gt_start_idx: torch.Tensor,       # (B,) long
    gt_end_idx: torch.Tensor,         # (B,) long
    temperature: float = 0.07,
    n_negatives: int = 8,
    lengths: torch.Tensor | None = None,  # (B,) real frame count per sample (None = all N)
) -> torch.Tensor:
    """
    InfoNCE loss with intra-video hard negatives.

    grounded_features must be the POST-cross-modal features (model output). Fed the
    cached input embeddings instead — which have no grad path to any trainable
    weight — this term is a constant and trains nothing (the shipped-model bug).
    lengths: negatives are sampled only from real frames, never from batch padding.

    For each sample:
      Positive  = mean-pool of frames in [gt_start, gt_end]
      Negatives = mean-pool of 8 non-overlapping segments from the same video

    Loss = cross_entropy([pos_sim, neg_sim_1, ..., neg_sim_8], label=0)
    where label=0 means 'the first entry (positive) is correct'.
    """
    B = query_emb.shape[0]
    losses = []

    for b in range(B):
        q    = query_emb[b]          # (1, d)
        n_b  = int(lengths[b]) if lengths is not None else grounded_features.shape[1]
        feats = grounded_features[b, :n_b]  # (n_b, d) — real frames only
        gs   = int(gt_start_idx[b].item())
        ge   = int(gt_end_idx[b].item())

        # Positive: mean-pool the GT segment
        pos_emb = feats[gs : ge + 1].mean(dim=0, keepdim=True)  # (1, d)

        # Negatives: 8 non-overlapping segments
        neg_segs = _sample_non_overlapping_segments(
            n_frames=feats.shape[0],
            gt_start=gs,
            gt_end=ge,
            n_negatives=n_negatives,
        )
        neg_embs = torch.stack(
            [feats[s : e + 1].mean(dim=0) for s, e in neg_segs]
        )  # (n_negatives, d)

        pos_sim  = F.cosine_similarity(q, pos_emb, dim=-1) / temperature              # (1,)
        neg_sims = F.cosine_similarity(q.expand(n_negatives, -1), neg_embs, dim=-1) / temperature  # (n_negatives,)

        logits = torch.cat([pos_sim, neg_sims]).unsqueeze(0)         # (1, 1+n_negatives)
        label  = torch.zeros(1, dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits, label))

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

def combined_loss(
    start_logits: torch.Tensor,       # (B, N)
    end_logits: torch.Tensor,         # (B, N)
    gt_start_idx: torch.Tensor,       # (B,) long
    gt_end_idx: torch.Tensor,         # (B,) long
    query_emb: torch.Tensor,          # (B, 1, d)
    grounded_features: torch.Tensor,  # (B, N, d)
    lengths: torch.Tensor | None = None,  # (B,) real frames per sample
) -> tuple:
    """
    total = span + 0.5 * iou + 0.1 * contrastive

    Returns (total, span, iou, contrastive) so the training loop can log each
    component to W&B separately for debugging.
    """
    span = span_extraction_loss(start_logits, end_logits, gt_start_idx, gt_end_idx)
    iou  = temporal_iou_loss(start_logits, end_logits, gt_start_idx, gt_end_idx)
    cont = contrastive_loss_intra_video(
        query_emb, grounded_features, gt_start_idx, gt_end_idx, lengths=lengths
    )
    total = span + 0.5 * iou + 0.1 * cont
    return total, span, iou, cont
