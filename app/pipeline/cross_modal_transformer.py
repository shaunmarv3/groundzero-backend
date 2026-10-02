"""
cross_modal_transformer.py — Cross-Modal Transformer for query-frame fusion.
Phase 3.5 (Chunk F).

Architecture:
  4 layers, each layer:
    1. CrossAttentionBlock  — Q=query, K=frames, V=frames
                              query attends over all frames → relevance map (B,1,N)
                              broadcast back: each frame absorbs query signal weighted by its score
                              output: (B, N, d_model) frames with query context injected

    2. SelfAttentionBlock   — Q=K=V=frames
                              frames spread relevance signal globally
                              output: (B, N, d_model) frames aware of each other

  Each block follows standard pre-norm transformer structure:
    Attention → Add&Norm → FFN → Add&Norm

  word_level=True (QD-DETR-style, Fix 3) swaps CrossAttentionBlock for
  WordCrossAttentionBlock: Q=frames, K=V=query WORD tokens (B, L, d_model).
  The pooled single-vector query gives every frame the SAME vector times one
  scalar (a gate); with L word tokens each frame gets its OWN mix of words.
"""

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """
    Query attends over frames: Q=query, K=frames, V=frames.

    The query is a single token — using it as K/V (frames=Q) gives only one key,
    so softmax is trivially 1.0 for every frame regardless of relevance. That
    reduces to a learned projection, not attention.

    Flipping the direction (query=Q, frames=K/V) produces a meaningful attention
    distribution over N frames — a relevance map of the timeline. The enriched
    query is then broadcast back to each frame weighted by its attention score,
    so relevant frames absorb a strong query signal and irrelevant frames get almost none.
    """

    def __init__(self, d_model: int = 1152, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, frames: torch.Tensor, query: torch.Tensor,
                frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        # frames:     (B, N, d_model) — video frame embeddings
        # query:      (B, 1, d_model) — text query embedding
        # frame_mask: (B, N) bool, True = real frame (optional)
        # returns (B, N, d_model)

        # query attends over frames → relevance map over the timeline
        # enriched_query: (B, 1, d_model)
        # attn_weights:   (B, 1, N) — how much each frame matched the query
        # key_padding_mask: True = IGNORE → pad frames get attention weight 0
        enriched_query, attn_weights = self.attn(
            query=query, key=frames, value=frames,
            key_padding_mask=None if frame_mask is None else ~frame_mask,
        )

        # broadcast back: scale each frame by its attention weight
        # relevant frames (high weight) absorb strong query signal
        # irrelevant frames (weight ≈ 0) get almost no update
        # attn_weights: (B, 1, N) → (B, N, 1) to broadcast over d_model
        frame_update = attn_weights.transpose(1, 2) * enriched_query  # (B, N, d_model)
        frames = self.norm1(frames + frame_update)
        frames = self.norm2(frames + self.ffn(frames))
        return frames


class WordCrossAttentionBlock(nn.Module):
    """
    Frames attend over query WORDS: Q=frames, K=V=word tokens (Fix 3, QD-DETR-style).

    This is the "natural" direction the pooled version could not use: with a single
    query token there was only one key, so softmax was always 1.0. With L word tokens
    each frame gets a real distribution over the words — frame 40 ("opens the
    fridge") can weight "opens" + "fridge", frame 80 ("closes it") can weight
    "fridge" but not "opens" — and absorbs a DIFFERENT vector per frame.
    Pad tokens (~82% of the 64 SigLIP positions) are masked out of the softmax.
    """

    def __init__(self, d_model: int = 1152, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, frames: torch.Tensor, words: torch.Tensor,
                word_mask: torch.Tensor | None = None) -> torch.Tensor:
        # frames:    (B, N, d_model)
        # words:     (B, L, d_model) — per-token SigLIP text states
        # word_mask: (B, L) bool, True = real token
        # returns (B, N, d_model)
        attn_out, _ = self.attn(
            query=frames, key=words, value=words,
            key_padding_mask=None if word_mask is None else ~word_mask,
            need_weights=False,
        )                                                # (B, N, d_model) — one mix per frame
        frames = self.norm1(frames + attn_out)
        frames = self.norm2(frames + self.ffn(frames))
        return frames


class SelfAttentionBlock(nn.Module):
    """
    Frames attend to all other frames: Q=K=V=frames.

    After cross-attention each frame knows its own query-relevance.
    Self-attention lets that signal spread globally — a frame whose
    neighbours are query-relevant also becomes relevant.
    """

    def __init__(self, d_model: int = 1152, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, frames: torch.Tensor,
                frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        # frames:     (B, N, d_model)
        # frame_mask: (B, N) bool, True = real frame (optional) — real frames never attend to pads
        # returns (B, N, d_model)
        attn_out, _ = self.attn(
            query=frames, key=frames, value=frames,
            key_padding_mask=None if frame_mask is None else ~frame_mask,
        )
        frames = self.norm1(frames + attn_out)
        frames = self.norm2(frames + self.ffn(frames))
        return frames


class CrossModalTransformer(nn.Module):
    """
    4 layers of alternating cross-attention (frames ← query) and
    self-attention (frames ← frames).

    Input:  frames (B, N, d_model) + query (B, 1, d_model)
    Output: (B, N, d_model) — each frame embedding is now query-aware
            and globally context-aware, ready for span extraction.
    """

    def __init__(
        self,
        d_model: int = 1152,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        word_level: bool = False,
    ):
        super().__init__()
        self.word_level = word_level
        cross_block = WordCrossAttentionBlock if word_level else CrossAttentionBlock
        self.cross_layers = nn.ModuleList([
            cross_block(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.self_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, frames: torch.Tensor, query: torch.Tensor,
                frame_mask: torch.Tensor | None = None,
                query_mask: torch.Tensor | None = None) -> torch.Tensor:
        # frames: (B, N, d_model)
        # query:  (B, 1, d_model) pooled, or (B, L, d_model) word tokens when word_level
        # frame_mask: (B, N) bool / query_mask: (B, L) bool — True = real (both optional)
        # returns (B, N, d_model)
        for cross_attn, self_attn in zip(self.cross_layers, self.self_layers):
            if self.word_level:
                frames = cross_attn(frames, query, query_mask)   # each frame reads the words
            else:
                frames = cross_attn(frames, query, frame_mask)   # inject query context
            frames = self_attn(frames, frame_mask)               # spread relevance globally
        return frames
