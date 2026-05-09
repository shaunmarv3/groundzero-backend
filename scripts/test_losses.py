"""
test_losses.py — Unit tests for app/training/losses.py
Phase 4.1

Run from groundzero-backend/:
    python scripts/test_losses.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.training.losses import (
    span_extraction_loss,
    temporal_iou_loss,
    contrastive_loss_intra_video,
    combined_loss,
    _sample_non_overlapping_segments,
)

B, N, D = 2, 50, 1152


def test_span_loss_positive():
    print("=== span_extraction_loss: positive value ===")
    start_logits = torch.randn(B, N)
    end_logits   = torch.randn(B, N)
    gt_start = torch.tensor([5, 10], dtype=torch.long)
    gt_end   = torch.tensor([15, 25], dtype=torch.long)

    loss = span_extraction_loss(start_logits, end_logits, gt_start, gt_end)

    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    assert torch.isfinite(loss), "Loss is NaN/Inf"
    print(f"  loss = {loss.item():.4f} ✓")


def test_span_loss_gradients():
    print("=== span_extraction_loss: gradients flow ===")
    start_logits = torch.randn(B, N, requires_grad=True)
    end_logits   = torch.randn(B, N, requires_grad=True)
    gt_start = torch.tensor([5, 10], dtype=torch.long)
    gt_end   = torch.tensor([15, 25], dtype=torch.long)

    loss = span_extraction_loss(start_logits, end_logits, gt_start, gt_end)
    loss.backward()

    assert start_logits.grad is not None, "No grad on start_logits"
    assert end_logits.grad is not None,   "No grad on end_logits"
    assert torch.isfinite(start_logits.grad).all(), "NaN/Inf in start_logits.grad"
    assert torch.isfinite(end_logits.grad).all(),   "NaN/Inf in end_logits.grad"
    print("  gradients OK ✓")


def test_iou_loss_in_range():
    print("=== temporal_iou_loss: value in [0, 1] ===")
    start_logits = torch.randn(B, N)
    end_logits   = torch.randn(B, N)
    gt_start = torch.tensor([5, 10], dtype=torch.long)
    gt_end   = torch.tensor([15, 25], dtype=torch.long)

    loss = temporal_iou_loss(start_logits, end_logits, gt_start, gt_end)

    assert 0.0 <= loss.item() <= 1.0, f"IoU loss out of [0,1]: {loss.item()}"
    assert torch.isfinite(loss), "IoU loss is NaN/Inf"
    print(f"  iou_loss = {loss.item():.4f} ✓")


def test_iou_loss_gradients():
    print("=== temporal_iou_loss: gradients flow (soft argmax) ===")
    start_logits = torch.randn(B, N, requires_grad=True)
    end_logits   = torch.randn(B, N, requires_grad=True)
    gt_start = torch.tensor([5, 10], dtype=torch.long)
    gt_end   = torch.tensor([15, 25], dtype=torch.long)

    loss = temporal_iou_loss(start_logits, end_logits, gt_start, gt_end)
    loss.backward()

    assert start_logits.grad is not None, "No grad on start_logits"
    assert torch.isfinite(start_logits.grad).all(), "NaN/Inf in start_logits.grad"
    print("  gradients OK ✓")


def test_non_overlapping_segments_count():
    print("=== _sample_non_overlapping_segments: returns n_negatives ===")
    segs = _sample_non_overlapping_segments(n_frames=100, gt_start=30, gt_end=50, n_negatives=8)
    assert len(segs) == 8, f"Expected 8 segments, got {len(segs)}"
    print(f"  got {len(segs)} segments ✓")


def test_non_overlapping_segments_no_overlap():
    print("=== _sample_non_overlapping_segments: no overlap with GT ===")
    for _ in range(20):
        segs = _sample_non_overlapping_segments(n_frames=100, gt_start=30, gt_end=50, n_negatives=8)
        for s, e in segs:
            assert e < 30 or s > 50, f"Segment ({s},{e}) overlaps GT [30,50]"
            assert 0 <= s <= e < 100,  f"Segment ({s},{e}) out of bounds"
    print("  no overlaps in 20 repeated calls ✓")


def test_contrastive_loss_positive():
    print("=== contrastive_loss_intra_video: positive value ===")
    query_emb        = torch.randn(B, 1, D)
    grounded_features = torch.randn(B, N, D)
    gt_start = torch.tensor([5, 10], dtype=torch.long)
    gt_end   = torch.tensor([15, 25], dtype=torch.long)

    loss = contrastive_loss_intra_video(query_emb, grounded_features, gt_start, gt_end)

    assert loss.item() > 0, f"Expected positive loss, got {loss.item()}"
    assert torch.isfinite(loss), "Contrastive loss is NaN/Inf"
    print(f"  contrastive_loss = {loss.item():.4f} ✓")


def test_contrastive_loss_gradients():
    print("=== contrastive_loss_intra_video: gradients flow ===")
    query_emb        = torch.randn(B, 1, D, requires_grad=True)
    grounded_features = torch.randn(B, N, D, requires_grad=True)
    gt_start = torch.tensor([5, 10], dtype=torch.long)
    gt_end   = torch.tensor([15, 25], dtype=torch.long)

    loss = contrastive_loss_intra_video(query_emb, grounded_features, gt_start, gt_end)
    loss.backward()

    assert grounded_features.grad is not None, "No grad on grounded_features"
    assert torch.isfinite(grounded_features.grad).all(), "NaN/Inf in grounded_features.grad"
    print("  gradients OK ✓")


def test_combined_loss_formula():
    print("=== combined_loss: formula is span + 0.5*iou + 0.1*cont ===")
    start_logits     = torch.randn(B, N)
    end_logits       = torch.randn(B, N)
    gt_start = torch.tensor([5, 10], dtype=torch.long)
    gt_end   = torch.tensor([15, 25], dtype=torch.long)
    query_emb        = torch.randn(B, 1, D)
    grounded_features = torch.randn(B, N, D)

    total, span, iou, cont = combined_loss(
        start_logits, end_logits, gt_start, gt_end, query_emb, grounded_features
    )

    expected = span + 0.5 * iou + 0.1 * cont
    assert abs(total.item() - expected.item()) < 1e-4, \
        f"Formula mismatch: total={total.item():.4f}, expected={expected.item():.4f}"
    assert total.item() > 0, "Total loss should be positive"
    print(f"  total={total.item():.4f}  span={span.item():.4f}  "
          f"iou={iou.item():.4f}  cont={cont.item():.4f} ✓")


def test_combined_loss_all_gradients():
    print("=== combined_loss: gradients reach all inputs ===")
    start_logits     = torch.randn(B, N, requires_grad=True)
    end_logits       = torch.randn(B, N, requires_grad=True)
    gt_start = torch.tensor([5, 10], dtype=torch.long)
    gt_end   = torch.tensor([15, 25], dtype=torch.long)
    query_emb        = torch.randn(B, 1, D)
    grounded_features = torch.randn(B, N, D, requires_grad=True)

    total, _, _, _ = combined_loss(
        start_logits, end_logits, gt_start, gt_end, query_emb, grounded_features
    )
    total.backward()

    assert start_logits.grad is not None,     "No grad on start_logits"
    assert end_logits.grad is not None,       "No grad on end_logits"
    assert grounded_features.grad is not None, "No grad on grounded_features"
    print("  gradients reach start_logits, end_logits, grounded_features ✓")


if __name__ == "__main__":
    test_span_loss_positive()
    test_span_loss_gradients()
    test_iou_loss_in_range()
    test_iou_loss_gradients()
    test_non_overlapping_segments_count()
    test_non_overlapping_segments_no_overlap()
    test_contrastive_loss_positive()
    test_contrastive_loss_gradients()
    test_combined_loss_formula()
    test_combined_loss_all_gradients()
    print("\nAll loss tests passed ✓")

# === span_extraction_loss: positive value ===
#   loss = 7.9201 ✓
# === span_extraction_loss: gradients flow ===
#   gradients OK ✓
# === temporal_iou_loss: value in [0, 1] ===
#   iou_loss = 1.0000 ✓
# === temporal_iou_loss: gradients flow (soft argmax) ===
#   gradients OK ✓
# === _sample_non_overlapping_segments: returns n_negatives ===
#   got 8 segments ✓
# === _sample_non_overlapping_segments: no overlap with GT ===
#   no overlaps in 20 repeated calls ✓
# === contrastive_loss_intra_video: positive value ===
#   contrastive_loss = 2.1898 ✓
# === contrastive_loss_intra_video: gradients flow ===
#   gradients OK ✓
# === combined_loss: formula is span + 0.5*iou + 0.1*cont ===
#   total=8.5016  span=7.8092  iou=1.0000  cont=1.9236 ✓
# === combined_loss: gradients reach all inputs ===
#   gradients reach start_logits, end_logits, grounded_features ✓

# All loss tests passed ✓