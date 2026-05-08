"""
span_extraction.py — Per-frame Span Extraction Head.
Phase 3.6 (Chunk G).

Architecture:
  start_scorer:      MLP(1152 → 256 → 1) applied to every frame → N start logits
  end_scorer:        MLP(1152 → 256 → 1) applied to every frame → N end logits
  confidence_head:   MLP(1152 → 64 → 1) applied to mean-pooled frames → scalar

  decode_best_span:  argmax(start_logits) → start_idx
                     argmax(end_logits, where end >= start_idx) → end_idx
                     enforces valid span: end always >= start

  to_seconds:        start_idx / N * duration_sec → start_sec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpanExtractionHead(nn.Module):
    """
    Reads query-aware frame embeddings and scores each frame for
    start and end boundary likelihood.

    Input:  (B, N, d_model) — output of CrossModalTransformer
    Output: start_logits (B, N), end_logits (B, N), confidence (B,)
    """

    def __init__(self, d_model: int = 1152, dropout: float = 0.1):
        super().__init__()

        # per-frame start scorer: is this frame the start of the event?
        self.start_scorer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # per-frame end scorer: is this frame the end of the event?
        self.end_scorer = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        # confidence head: is the described event present in this video at all?
        # operates on mean-pooled frame features → single scalar per sample
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, N, d_model) — query-aware frame embeddings

        Returns:
            start_logits:  (B, N) — raw start scores per frame (use with CrossEntropy)
            end_logits:    (B, N) — raw end scores per frame
            confidence:    (B,)  — event presence score in [0, 1]
        """
        start_logits = self.start_scorer(x).squeeze(-1)   # (B, N)
        end_logits   = self.end_scorer(x).squeeze(-1)     # (B, N)

        pooled     = x.mean(dim=1)                        # (B, d_model)
        confidence = self.confidence_head(pooled).squeeze(-1)  # (B,)

        return start_logits, end_logits, confidence


def decode_best_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
) -> tuple:
    """
    Find the highest-scoring valid span where end_idx >= start_idx.

    Uses the same decoding algorithm as extractive QA (BERT SQuAD):
    score(i, j) = start_logits[i] + end_logits[j]  for all j >= i
    Pick (i, j) with the highest combined score.

    Args:
        start_logits: (N,) — per-frame start scores for one sample
        end_logits:   (N,) — per-frame end scores for one sample

    Returns:
        (start_idx, end_idx) as Python ints
    """
    N = start_logits.shape[0]

    # outer sum: score_matrix[i, j] = start_logits[i] + end_logits[j]
    score_matrix = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)  # (N, N)

    # keep only valid spans where end >= start (upper triangle including diagonal)
    mask = torch.triu(torch.ones(N, N, device=start_logits.device))     # upper triangle: col >= row
    score_matrix = score_matrix * mask + (1 - mask) * (-1e9)            # mask out j < i

    # find global argmax
    best_idx   = score_matrix.argmax()
    start_idx  = (best_idx // N).item()
    end_idx    = (best_idx  % N).item()

    return start_idx, end_idx


def to_seconds(
    start_idx: int,
    end_idx: int,
    n_frames: int,
    duration_sec: float,
) -> tuple:
    """
    Convert frame indices to timestamps in seconds.

    Args:
        start_idx:    predicted start frame index
        end_idx:      predicted end frame index
        n_frames:     total number of frames in the video
        duration_sec: video duration in seconds

    Returns:
        (start_sec, end_sec) as floats
    """
    start_sec = start_idx / n_frames * duration_sec
    end_sec   = end_idx   / n_frames * duration_sec
    return start_sec, end_sec
