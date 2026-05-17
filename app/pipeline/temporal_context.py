import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Encodes fractional timestamps (t/T in [0,1]) into sinusoidal vectors.
    Using t/T instead of raw frame indices makes the encoding length-agnostic —
    a 3-minute and a 30-minute video both map to the same [0,1] range.
    """

    def __init__(self, d_model: int = 1152):
        super().__init__()
        self.d_model = d_model

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        # timestamps: (N,) — fractional positions in [0, 1]
        # returns:    (N, d_model)
        N = timestamps.shape[0]
        device = timestamps.device

        i = torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
        div_term = torch.pow(10000.0, i / self.d_model)  # (d_model/2,)

        t = timestamps.unsqueeze(1).float()  # (N, 1)

        pe = torch.zeros(N, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(t / div_term)
        pe[:, 1::2] = torch.cos(t / div_term)

        return pe  # (N, d_model)


class TemporalContextModule(nn.Module):
    """
    Adds temporal context to per-frame embeddings via 4-layer dilated 1D convolutions.

    Dilations [1, 2, 4, 8] with kernel_size=5 give a receptive field of ~61 frames.
    At 1 fps that's ~61 seconds of context — each output frame "sees" what happened
    30 seconds before and after it.

    Input/output shape: (B, N, d_model) — sequence length N is preserved.
    """

    def __init__(self, d_model: int = 1152, kernel_size: int = 5):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        dilations = [1, 2, 4, 8]
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                dilation=d,
                padding="same",
            )
            for d in dilations
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in dilations
        ])

    def forward(self, x: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        # x:          (B, N, d_model) — frame embeddings from visual encoder
        # timestamps: (N,) or (B, N)  — fractional positions t/T in [0, 1]
        #             (N,)   → same timestamps for all samples (backward-compat / inference)
        #             (B, N) → per-sample timestamps (training with variable-length videos)
        # returns:    (B, N, d_model)

        if timestamps.dim() == 1:
            pe = self.pos_enc(timestamps)  # (N, d_model) — broadcasts over B
        else:
            # per-sample: loop is tiny (B ≤ 4 in training)
            pe = torch.stack([self.pos_enc(timestamps[i]) for i in range(timestamps.shape[0])])

        x = x + pe  # inject position info

        for conv, norm in zip(self.convs, self.norms):
            residual = x                      # (B, N, d_model)
            x_t = x.transpose(1, 2)          # (B, d_model, N) — Conv1d format
            x_t = conv(x_t)                  # (B, d_model, N)
            x_t = x_t.transpose(1, 2)        # (B, N, d_model)
            x = norm(x_t + residual)          # residual connection + LayerNorm

        return x

    @staticmethod
    def receptive_field(kernel_size: int = 5) -> int:
        dilations = [1, 2, 4, 8]
        print(f"Receptive field  (kernel_size={kernel_size}, dilations={dilations})")
        print("Layer | Dilation | Cumulative RF")
        print("------+----------+--------------")
        rf = 1
        for i, d in enumerate(dilations):
            rf += d * (kernel_size - 1)
            print(f"  {i + 1}   |    {d:2d}    |    {rf}")
        print(f"\nTotal: {rf} frames  ≈  {rf}s at 1 fps")
        return rf
