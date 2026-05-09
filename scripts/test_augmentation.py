"""
test_augmentation.py — Unit tests for app/training/augmentation.py
Phase 4.2

Run from groundzero-backend/:
    python scripts/test_augmentation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from PIL import Image
from app.training.augmentation import (
    temporal_jitter,
    random_temporal_crop,
    speed_perturbation,
    sample_paraphrase,
    maybe_inject_negative_query,
)


def _make_frames(n=60, duration_sec=60.0):
    """Helper: n dummy 384x384 frames with evenly spaced timestamps."""
    frames     = [Image.new("RGB", (384, 384), color=(i % 256, i % 256, i % 256)) for i in range(n)]
    timestamps = [i * duration_sec / (n - 1) for i in range(n)]
    return frames, timestamps


# ---------------------------------------------------------------------------
# 1. temporal_jitter
# ---------------------------------------------------------------------------

def test_jitter_stays_in_bounds():
    print("=== temporal_jitter: output stays in [0, duration] ===")
    for _ in range(200):
        new_start, new_end = temporal_jitter(10.0, 30.0, 60.0)
        assert 0.0 <= new_start <= 60.0, f"new_start={new_start:.3f} out of bounds"
        assert 0.0 <= new_end   <= 60.0, f"new_end={new_end:.3f} out of bounds"
    print("  200 jitters all in [0, 60] ✓")


def test_jitter_actually_moves():
    print("=== temporal_jitter: boundaries actually shift ===")
    random.seed(42)
    moves = sum(
        1 for _ in range(100)
        if abs(temporal_jitter(10.0, 30.0, 60.0)[0] - 10.0) > 0.01
    )
    assert moves > 20, f"Expected jitter to move boundaries often, moved {moves}/100"
    print(f"  moved in {moves}/100 trials ✓")


# ---------------------------------------------------------------------------
# 2. random_temporal_crop
# ---------------------------------------------------------------------------

def test_crop_contains_gt():
    print("=== random_temporal_crop: GT always fully inside crop ===")
    frames, timestamps = _make_frames(n=100, duration_sec=100.0)
    gt_start, gt_end = 30.0, 60.0
    for _ in range(100):
        c_frames, c_ts, gs, ge = random_temporal_crop(frames, timestamps, gt_start, gt_end)
        assert len(c_frames) > 0,        "Crop returned 0 frames"
        assert c_ts[0]  <= gt_start,     f"Crop starts {c_ts[0]:.1f}s > GT start {gt_start}s"
        assert c_ts[-1] >= gt_end,       f"Crop ends {c_ts[-1]:.1f}s < GT end {gt_end}s"
        assert gs == gt_start,           "gt_start_sec should be unchanged"
        assert ge == gt_end,             "gt_end_sec should be unchanged"
    print("  GT inside crop in all 100 trials ✓")


def test_crop_minimum_length():
    print("=== random_temporal_crop: covers at least 60% of video ===")
    frames, timestamps = _make_frames(n=100, duration_sec=100.0)
    gt_start, gt_end = 30.0, 60.0
    for _ in range(50):
        _, c_ts, _, _ = random_temporal_crop(frames, timestamps, gt_start, gt_end, min_frac=0.6)
        crop_dur = c_ts[-1] - c_ts[0]
        assert crop_dur >= 100.0 * 0.55, f"Crop duration {crop_dur:.1f}s too short"
    print("  all crops >= ~60% of video ✓")


# ---------------------------------------------------------------------------
# 3. speed_perturbation
# ---------------------------------------------------------------------------

def test_speed_perturbation_changes_count():
    print("=== speed_perturbation: changes frame count ===")
    random.seed(0)
    frames, timestamps = _make_frames(n=20, duration_sec=20.0)
    changed = sum(
        1 for _ in range(100)
        if len(speed_perturbation(frames, timestamps, apply_prob=1.0)[0]) != len(frames)
    )
    assert changed > 0, "speed_perturbation never changed frame count in 100 trials"
    print(f"  frame count changed in {changed}/100 trials ✓")


def test_speed_perturbation_skips_at_zero_prob():
    print("=== speed_perturbation: skips when apply_prob=0 ===")
    frames, timestamps = _make_frames(n=20, duration_sec=20.0)
    new_frames, new_ts = speed_perturbation(frames, timestamps, apply_prob=0.0)
    assert len(new_frames) == len(frames), "Should not modify frames when apply_prob=0"
    print("  skipped correctly ✓")


def test_speed_perturbation_preserves_order():
    print("=== speed_perturbation: timestamps stay monotonically non-decreasing ===")
    random.seed(7)
    frames, timestamps = _make_frames(n=30, duration_sec=30.0)
    for _ in range(50):
        _, new_ts = speed_perturbation(frames, timestamps, apply_prob=1.0)
        for i in range(1, len(new_ts)):
            assert new_ts[i] >= new_ts[i - 1], f"Timestamps not ordered at index {i}"
    print("  timestamps ordered in all 50 trials ✓")


# ---------------------------------------------------------------------------
# 4. sample_paraphrase
# ---------------------------------------------------------------------------

def test_paraphrase_returns_variant():
    print("=== sample_paraphrase: returns paraphrase or original ===")
    query = "the speaker slams the table"
    paraphrases = {query: ["someone hits the table", "person banging on desk"]}
    results = {sample_paraphrase(query, paraphrases) for _ in range(50)}
    assert len(results) >= 2, f"Expected at least 2 distinct outputs, got {results}"
    assert all(r in [query] + paraphrases[query] for r in results), \
        f"Got unexpected result: {results}"
    print(f"  returned variants: {results} ✓")


def test_paraphrase_falls_back_to_original():
    print("=== sample_paraphrase: unknown query returns original ===")
    result = sample_paraphrase("unknown query", {})
    assert result == "unknown query", f"Expected original, got '{result}'"
    print("  falls back to original ✓")


# ---------------------------------------------------------------------------
# 5. maybe_inject_negative_query
# ---------------------------------------------------------------------------

def test_injection_rate():
    print("=== maybe_inject_negative_query: ~20% injection rate ===")
    all_queries = [f"query_{i}" for i in range(20)]
    n_injected = sum(
        1 for _ in range(500)
        if maybe_inject_negative_query("query_0", all_queries, inject_prob=0.2)[1]
    )
    assert 50 <= n_injected <= 150, \
        f"Expected ~100 injections/500, got {n_injected} (rate too far from 20%)"
    print(f"  {n_injected}/500 injected (expected ~100) ✓")


def test_injected_query_differs():
    print("=== maybe_inject_negative_query: injected != original ===")
    all_queries = ["query_A", "query_B", "query_C"]
    for _ in range(50):
        returned, is_neg = maybe_inject_negative_query("query_A", all_queries, inject_prob=1.0)
        assert returned != "query_A", f"Injected query same as original: {returned}"
        assert is_neg is True, "is_negative should be True when inject_prob=1.0"
    print("  injected always differs from original ✓")


def test_no_injection_returns_original():
    print("=== maybe_inject_negative_query: no injection returns original + False ===")
    q, is_neg = maybe_inject_negative_query("query_A", ["query_A", "query_B"], inject_prob=0.0)
    assert q == "query_A", f"Expected original query, got '{q}'"
    assert is_neg is False, "is_negative should be False when inject_prob=0.0"
    print("  original returned with is_negative=False ✓")


if __name__ == "__main__":
    test_jitter_stays_in_bounds()
    test_jitter_actually_moves()
    test_crop_contains_gt()
    test_crop_minimum_length()
    test_speed_perturbation_changes_count()
    test_speed_perturbation_skips_at_zero_prob()
    test_speed_perturbation_preserves_order()
    test_paraphrase_returns_variant()
    test_paraphrase_falls_back_to_original()
    test_injection_rate()
    test_injected_query_differs()
    test_no_injection_returns_original()
    print("\nAll augmentation tests passed ✓")

# === temporal_jitter: output stays in [0, duration] ===
#   200 jitters all in [0, 60] ✓
# === temporal_jitter: boundaries actually shift ===
#   moved in 100/100 trials ✓
# === random_temporal_crop: GT always fully inside crop ===
#   GT inside crop in all 100 trials ✓
# === random_temporal_crop: covers at least 60% of video ===
#   all crops >= ~60% of video ✓
# === speed_perturbation: changes frame count ===
#   frame count changed in 100/100 trials ✓
# === speed_perturbation: skips when apply_prob=0 ===
#   skipped correctly ✓
# === speed_perturbation: timestamps stay monotonically non-decreasing ===
#   timestamps ordered in all 50 trials ✓
# === sample_paraphrase: returns paraphrase or original ===
#   returned variants: {'the speaker slams the table', 'someone hits the table', 'person banging on desk'} ✓
# === sample_paraphrase: unknown query returns original ===
#   falls back to original ✓
# === maybe_inject_negative_query: ~20% injection rate ===
#   108/500 injected (expected ~100) ✓
# === maybe_inject_negative_query: injected != original ===
#   injected always differs from original ✓
# === maybe_inject_negative_query: no injection returns original + False ===
#   original returned with is_negative=False ✓

# All augmentation tests passed ✓