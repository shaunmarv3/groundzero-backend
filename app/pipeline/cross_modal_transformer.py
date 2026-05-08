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
    Each frame attends to the query: Q=frames, K=query, V=query.

    Output shape matches Q — so output is (B, N, d_model).
    Each frame gets query context injected in proportion to how well it matches.
    """

    def __init__(self, d_model: int = 1152, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # batch_first=True so shapes are (B, seq, d_model) not (seq, B, d_model)
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
        attn_out, _ = self.attn(query=frames, key=query, value=query)
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
