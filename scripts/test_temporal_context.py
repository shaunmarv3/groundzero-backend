"""
test_temporal_context.py — Unit tests for temporal_context.py

Run from groundzero-backend/:
    python scripts/test_temporal_context.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.pipeline.temporal_context import SinusoidalPositionalEncoding, TemporalContextModule


def test_receptive_field():
    print("=== Receptive Field ===")
    rf = TemporalContextModule.receptive_field(kernel_size=5)
    assert rf == 61, f"Expected RF=61, got {rf}"
    print("PASS\n")


def test_positional_encoding_shape():
    print("=== Positional Encoding — shape & finite ===")
    pe = SinusoidalPositionalEncoding(d_model=1152)
    timestamps = torch.linspace(0, 1, 10)
    out = pe(timestamps)

    assert out.shape == (10, 1152), f"Expected (10, 1152), got {out.shape}"
    assert torch.isfinite(out).all(), "NaNs or Infs in positional encoding"
    assert not torch.allclose(out[0], out[-1]), "All positions encoded identically"

    print(f"  Shape:          {out.shape} ✓")
    print(f"  All finite:     True ✓")
    print(f"  Positions differ: True ✓")
    print("PASS\n")


def test_forward_shape():
    print("=== TemporalContextModule — output shape unchanged ===")
    B, N, d_model = 1, 100, 1152
    module = TemporalContextModule(d_model=d_model, kernel_size=5)
    module.eval()

    x = torch.randn(B, N, d_model)
    timestamps = torch.linspace(0, 1, N)

    with torch.no_grad():
        out = module(x, timestamps)

    assert out.shape == (B, N, d_model), f"Expected {(B, N, d_model)}, got {out.shape}"
    assert torch.isfinite(out).all(), "NaNs or Infs in output"

    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape} ✓")
    print(f"  All finite: True ✓")
    print("PASS\n")


def test_batch_sizes():
    print("=== Batch Size Invariance ===")
    module = TemporalContextModule(d_model=1152, kernel_size=5)
    module.eval()

    N = 50
    timestamps = torch.linspace(0, 1, N)
    x = torch.randn(1, N, 1152)

    with torch.no_grad():
        for B in [1, 2, 4]:
            out = module(x.expand(B, -1, -1), timestamps)
            assert out.shape == (B, N, 1152), f"Failed for B={B}: got {out.shape}"
            print(f"  B={B}: {out.shape} ✓")
    print("PASS\n")


def test_output_changes_with_input():
    print("=== Output changes when input changes ===")
    module = TemporalContextModule(d_model=1152, kernel_size=5)
    module.eval()

    timestamps = torch.linspace(0, 1, 20)
    x1 = torch.randn(1, 20, 1152)
    x2 = torch.randn(1, 20, 1152)

    with torch.no_grad():
        out1 = module(x1, timestamps)
        out2 = module(x2, timestamps)

    assert not torch.allclose(out1, out2), "Different inputs produced identical outputs"
    print("  Different inputs → different outputs ✓")
    print("PASS\n")


if __name__ == "__main__":
    print("Running temporal_context.py unit tests\n")
    test_receptive_field()
    test_positional_encoding_shape()
    test_forward_shape()
    test_batch_sizes()
    test_output_changes_with_input()
    print("=" * 40)
    print("All tests passed!")
