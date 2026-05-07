"""
test_text_encoder.py — Unit tests for text_encoder.py

Run from groundzero-backend/:
    python scripts/test_text_encoder.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.pipeline.text_encoder import TextEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_single_query_shape(encoder: TextEncoder):
    print("=== Single query shape ===")
    out = encoder.encode_query("a person opens a door")

    assert out.shape == (1, 1152), f"Expected (1, 1152), got {out.shape}"
    assert torch.isfinite(out).all(), "NaNs or Infs in output"

    print(f"  Output: {out.shape} ✓")
    print(f"  All finite: True ✓")
    print("PASS\n")


def test_batch_queries_shape(encoder: TextEncoder):
    print("=== Batch queries shape ===")
    queries = [
        "a person opens a door",
        "someone is cooking in the kitchen",
        "a dog runs across the field",
        "two people shake hands",
        "a car drives past a building",
    ]
    out = encoder.encode_queries(queries)

    assert out.shape == (5, 1152), f"Expected (5, 1152), got {out.shape}"
    assert torch.isfinite(out).all(), "NaNs or Infs in output"

    print(f"  5 queries → {out.shape} ✓")
    print(f"  All finite: True ✓")
    print("PASS\n")


def test_different_queries_give_different_embeddings(encoder: TextEncoder):
    print("=== Different queries → different embeddings ===")
    out_a = encoder.encode_query("a person opens a door")
    out_b = encoder.encode_query("a dog runs across the field")

    assert not torch.allclose(out_a, out_b), "Different queries gave identical embeddings"
    print("  Different queries → different embeddings ✓")
    print("PASS\n")


def test_all_params_frozen(encoder: TextEncoder):
    print("=== All parameters frozen ===")
    counts = encoder.count_params()

    trainable_m = counts["trainable"] / 1e6
    total_m     = counts["total"]     / 1e6

    assert counts["trainable"] == 0, f"Expected 0 trainable params, got {counts['trainable']}"

    print(f"  Total:     {total_m:.1f}M")
    print(f"  Trainable: {trainable_m:.3f}M ✓  (fully frozen)")
    print("PASS\n")


def test_same_query_same_embedding(encoder: TextEncoder):
    print("=== Same query → same embedding (deterministic) ===")
    q = "a person opens a door"
    out_1 = encoder.encode_query(q)
    out_2 = encoder.encode_query(q)

    assert torch.allclose(out_1, out_2), "Same query gave different embeddings — not deterministic"
    print("  Same query → identical embedding ✓")
    print("PASS\n")


if __name__ == "__main__":
    print(f"Device: {DEVICE.upper()}")
    print("Loading SigLIP 2 So400m (from cache)...")
    encoder = TextEncoder(device=DEVICE)
    print("Loaded.\n")

    test_single_query_shape(encoder)
    test_batch_queries_shape(encoder)
    test_different_queries_give_different_embeddings(encoder)
    test_all_params_frozen(encoder)
    test_same_query_same_embedding(encoder)

    print("=" * 40)
    print("All tests passed!")
