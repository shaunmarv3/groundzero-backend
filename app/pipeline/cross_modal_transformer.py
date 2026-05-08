"""
cross_modal_transformer.py — Cross-Modal Transformer for query-frame fusion.
Phase 3.5 (Chunk F).

Architecture:
  4 layers, each layer:
    1. CrossAttentionBlock  — Q=frames, K=query, V=query
                              each frame asks "how relevant am I to the query?"
                              output: (B, N, d_model) frames with query context injected

    2. SelfAttentionBlock   — Q=K=V=frames
                              frames spread relevance signal globally
                              output: (B, N, d_model) frames aware of each other

  Each block follows standard pre-norm transformer structure:
    Attention → Add&Norm → FFN → Add&Norm
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

    def forward(self, frames: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # frames: (B, N, d_model) — video frame embeddings
        # query:  (B, 1, d_model) — text query embedding
        # returns (B, N, d_model)

        # query attends over frames → relevance map over the timeline
        # enriched_query: (B, 1, d_model)
        # attn_weights:   (B, 1, N) — how much each frame matched the query
        enriched_query, attn_weights = self.attn(
            query=query, key=frames, value=frames
        )

        # broadcast back: scale each frame by its attention weight
        # relevant frames (high weight) absorb strong query signal
        # irrelevant frames (weight ≈ 0) get almost no update
        # attn_weights: (B, 1, N) → (B, N, 1) to broadcast over d_model
        frame_update = attn_weights.transpose(1, 2) * enriched_query  # (B, N, d_model)
        frames = self.norm1(frames + frame_update)
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

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B, N, d_model)
        # returns (B, N, d_model)
        attn_out, _ = self.attn(query=frames, key=frames, value=frames)
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
    ):
        super().__init__()
        self.cross_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.self_layers = nn.ModuleList([
            SelfAttentionBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

    def forward(self, frames: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # frames: (B, N, d_model), query: (B, 1, d_model)
        # returns (B, N, d_model)
        for cross_attn, self_attn in zip(self.cross_layers, self.self_layers):
            frames = cross_attn(frames, query)  # inject query context
            frames = self_attn(frames)           # spread relevance globally
        return frames
