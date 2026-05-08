"""
test_cross_modal_transformer.py — Unit tests for cross_modal_transformer.py

Run from groundzero-backend/:
    python scripts/test_cross_modal_transformer.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.pipeline.cross_modal_transformer import (
    CrossAttentionBlock,
    SelfAttentionBlock,
    CrossModalTransformer,
)

D_MODEL = 1152
N_HEADS = 8
B, N = 1, 100  # batch size, num frames


def test_output_shape():
    print("=== Output shape unchanged ===")
    model = CrossModalTransformer(d_model=D_MODEL, n_heads=N_HEADS, n_layers=4)
    model.eval()

    frames = torch.randn(B, N, D_MODEL)
    query  = torch.randn(B, 1, D_MODEL)

    with torch.no_grad():
        out = model(frames, query)

    assert out.shape == (B, N, D_MODEL), f"Expected {(B, N, D_MODEL)}, got {out.shape}"
    assert torch.isfinite(out).all(), "NaNs or Infs in output"

    print(f"  frames {frames.shape} + query {query.shape} → {out.shape} ✓")
    print(f"  All finite: True ✓")
    print("PASS\n")


def test_different_queries_give_different_outputs():
    print("=== Different queries → different frame embeddings ===")
    model = CrossModalTransformer(d_model=D_MODEL, n_heads=N_HEADS, n_layers=4)
    model.eval()

    frames  = torch.randn(B, N, D_MODEL)
    query_a = torch.randn(B, 1, D_MODEL)
    query_b = torch.randn(B, 1, D_MODEL)

    with torch.no_grad():
        out_a = model(frames, query_a)
        out_b = model(frames, query_b)

    assert not torch.allclose(out_a, out_b), "Different queries gave identical outputs"
    print("  Different queries → different frame embeddings ✓")
    print("PASS\n")


def test_different_frames_give_different_outputs():
    print("=== Different frames → different outputs ===")
    model = CrossModalTransformer(d_model=D_MODEL, n_heads=N_HEADS, n_layers=4)
    model.eval()

    query    = torch.randn(B, 1, D_MODEL)
    frames_a = torch.randn(B, N, D_MODEL)
    frames_b = torch.randn(B, N, D_MODEL)

    with torch.no_grad():
        out_a = model(frames_a, query)
        out_b = model(frames_b, query)

    assert not torch.allclose(out_a, out_b), "Different frames gave identical outputs"
    print("  Different frames → different outputs ✓")
    print("PASS\n")


def test_batch_size_invariance():
    print("=== Batch size invariance ===")
    model = CrossModalTransformer(d_model=D_MODEL, n_heads=N_HEADS, n_layers=4)
    model.eval()

    with torch.no_grad():
        for b in [1, 2, 4]:
            frames = torch.randn(b, N, D_MODEL)
            query  = torch.randn(b, 1, D_MODEL)
            out    = model(frames, query)
            assert out.shape == (b, N, D_MODEL), f"Failed for B={b}: got {out.shape}"
            print(f"  B={b}: frames {frames.shape} → {out.shape} ✓")
    print("PASS\n")


def test_output_changes_from_input():
    print("=== Output differs from raw input (model is doing something) ===")
    model = CrossModalTransformer(d_model=D_MODEL, n_heads=N_HEADS, n_layers=4)
    model.eval()

    frames = torch.randn(B, N, D_MODEL)
    query  = torch.randn(B, 1, D_MODEL)

    with torch.no_grad():
        out = model(frames, query)

    assert not torch.allclose(frames, out), "Output identical to input — model did nothing"
    print("  Output ≠ input — cross-modal fusion is active ✓")
    print("PASS\n")


def test_param_count():
    print("=== Parameter count ===")
    model = CrossModalTransformer(d_model=D_MODEL, n_heads=N_HEADS, n_layers=4)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())

    assert trainable > 0, "No trainable params"
    assert trainable == total, "Some params unexpectedly frozen"

    print(f"  Total:     {total / 1e6:.1f}M")
    print(f"  Trainable: {trainable / 1e6:.1f}M ✓  (all params trained from scratch)")
    print("PASS\n")


if __name__ == "__main__":
    print("Running cross_modal_transformer.py unit tests\n")
    test_output_shape()
    test_different_queries_give_different_outputs()
    test_different_frames_give_different_outputs()
    test_batch_size_invariance()
    test_output_changes_from_input()
    test_param_count()
    print("=" * 40)
    print("All tests passed!")


# Running cross_modal_transformer.py unit tests

# === Output shape unchanged ===
#   frames torch.Size([1, 100, 1152]) + query torch.Size([1, 1, 1152]) → torch.Size([1, 100, 1152]) ✓
#   All finite: True ✓
# PASS

# === Different queries → different frame embeddings ===
#   Different queries → different frame embeddings ✓
# PASS

# === Different frames → different outputs ===
#   Different frames → different outputs ✓
# PASS

# === Batch size invariance ===
#   B=1: frames torch.Size([1, 100, 1152]) → torch.Size([1, 100, 1152]) ✓
#   B=2: frames torch.Size([2, 100, 1152]) → torch.Size([2, 100, 1152]) ✓
#   B=4: frames torch.Size([4, 100, 1152]) → torch.Size([4, 100, 1152]) ✓
# PASS

# === Output differs from raw input (model is doing something) ===
#   Output ≠ input — cross-modal fusion is active ✓
# PASS

# === Parameter count ===
#   Total:     127.5M
#   Trainable: 127.5M ✓  (all params trained from scratch)
# PASS

# ========================================
# All tests passed!