"""
test_span_extraction.py — Unit tests for span_extraction.py

Run from groundzero-backend/:
    python scripts/test_span_extraction.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.pipeline.span_extraction import SpanExtractionHead, decode_best_span, to_seconds

D_MODEL = 1152
B, N = 2, 100


def test_output_shapes():
    print("=== Output shapes ===")
    head = SpanExtractionHead(d_model=D_MODEL)
    head.eval()

    x = torch.randn(B, N, D_MODEL)
    with torch.no_grad():
        start_logits, end_logits, confidence = head(x)

    assert start_logits.shape == (B, N), f"Expected ({B},{N}), got {start_logits.shape}"
    assert end_logits.shape   == (B, N), f"Expected ({B},{N}), got {end_logits.shape}"
    assert confidence.shape   == (B,),   f"Expected ({B},), got {confidence.shape}"
    assert torch.isfinite(start_logits).all(), "NaNs in start_logits"
    assert torch.isfinite(end_logits).all(),   "NaNs in end_logits"

    print(f"  start_logits: {start_logits.shape} ✓")
    print(f"  end_logits:   {end_logits.shape} ✓")
    print(f"  confidence:   {confidence.shape}, values in [0,1]: "
          f"{(confidence >= 0).all() and (confidence <= 1).all()} ✓")
    print("PASS\n")


def test_decode_best_span_valid():
    print("=== decode_best_span always returns end >= start ===")
    for _ in range(50):
        start_logits = torch.randn(N)
        end_logits   = torch.randn(N)
        s, e = decode_best_span(start_logits, end_logits)
        assert e >= s, f"Invalid span: start={s}, end={e}"

    print("  50 random decode calls — end >= start always ✓")
    print("PASS\n")


def test_decode_best_span_finds_peak():
    print("=== decode_best_span finds the planted peak ===")
    start_logits = torch.full((N,), -10.0)
    end_logits   = torch.full((N,), -10.0)

    # plant a clear peak at (30, 45)
    start_logits[30] = 10.0
    end_logits[45]   = 10.0

    s, e = decode_best_span(start_logits, end_logits)
    assert s == 30, f"Expected start=30, got {s}"
    assert e == 45, f"Expected end=45, got {e}"

    print(f"  Planted peak at (30, 45) → decoded ({s}, {e}) ✓")
    print("PASS\n")


def test_to_seconds():
    print("=== to_seconds conversion ===")
    start_sec, end_sec = to_seconds(
        start_idx=30, end_idx=45, n_frames=100, duration_sec=200.0
    )
    assert abs(start_sec - 60.0) < 1e-5, f"Expected 60.0, got {start_sec}"
    assert abs(end_sec   - 90.0) < 1e-5, f"Expected 90.0, got {end_sec}"

    print(f"  frame 30/100 of 200s video → {start_sec}s ✓")
    print(f"  frame 45/100 of 200s video → {end_sec}s ✓")
    print("PASS\n")


def test_param_count():
    print("=== Parameter count ===")
    head = SpanExtractionHead(d_model=D_MODEL)
    total = sum(p.numel() for p in head.parameters())

    print(f"  Total params: {total:,}  (~{total/1e3:.1f}K) ✓")
    assert total < 1_000_000, f"SpanHead unexpectedly large: {total}"
    print("PASS\n")


def test_different_inputs_different_outputs():
    print("=== Different inputs → different outputs ===")
    head = SpanExtractionHead(d_model=D_MODEL)
    head.eval()

    x1 = torch.randn(1, N, D_MODEL)
    x2 = torch.randn(1, N, D_MODEL)

    with torch.no_grad():
        s1, e1, c1 = head(x1)
        s2, e2, c2 = head(x2)

    assert not torch.allclose(s1, s2), "Different inputs gave identical start logits"
    print("  Different inputs → different logits ✓")
    print("PASS\n")


if __name__ == "__main__":
    print("Running span_extraction.py unit tests\n")
    test_output_shapes()
    test_decode_best_span_valid()
    test_decode_best_span_finds_peak()
    test_to_seconds()
    test_param_count()
    test_different_inputs_different_outputs()
    print("=" * 40)
    print("All tests passed!")


# Running span_extraction.py unit tests

# === Output shapes ===
#   start_logits: torch.Size([2, 100]) ✓
#   end_logits:   torch.Size([2, 100]) ✓
#   confidence:   torch.Size([2]), values in [0,1]: True ✓
# PASS

# === decode_best_span always returns end >= start ===
#   50 random decode calls — end >= start always ✓
# PASS

# === decode_best_span finds the planted peak ===
#   Planted peak at (30, 45) → decoded (30, 45) ✓
# PASS

# === to_seconds conversion ===
#   frame 30/100 of 200s video → 60.0s ✓
#   frame 45/100 of 200s video → 90.0s ✓
# PASS

# === Parameter count ===
#   Total params: 664,707  (~664.7K) ✓
# PASS

# === Different inputs → different outputs ===
#   Different inputs → different logits ✓
# PASS

# ========================================
# All tests passed!
# PS D:\groundzero\groundzero-backend> 