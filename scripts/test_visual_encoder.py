"""
test_visual_encoder.py — Unit tests for visual_encoder.py

Run from groundzero-backend/:
    python scripts/test_visual_encoder.py

SigLIP 2 So400m loads from HuggingFace cache (~1.6GB).
First run downloads it; subsequent runs are instant.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import torch
from PIL import Image
from app.pipeline.visual_encoder import VisualEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_dummy_frames(n: int, size: int = 384) -> list:
    frames = []
    for _ in range(n):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        frames.append(Image.new("RGB", (size, size), color=color))
    return frames


def test_output_shape(encoder: VisualEncoder):
    print("=== Output Shape ===")
    frames = make_dummy_frames(10)

    with torch.no_grad():
        out = encoder.encode_frames(frames)

    assert out.shape == (10, 1152), f"Expected (10, 1152), got {out.shape}"
    assert torch.isfinite(out).all(), "NaNs or Infs in output"

    print(f"  Input:      10 PIL images")
    print(f"  Output:     {out.shape} ✓")
    print(f"  All finite: True ✓")
    print("PASS\n")


def test_trainable_params(encoder: VisualEncoder):
    print("=== Trainable Parameters (LoRA) ===")
    counts = encoder.count_params()

    trainable_m = counts["trainable"] / 1e6
    total_m     = counts["total"]     / 1e6
    frozen_m    = counts["frozen"]    / 1e6

    assert counts["trainable"] > 0,               "No trainable params — LoRA not applied"
    assert counts["trainable"] < counts["total"],  "All params trainable — freeze failed"

    print(f"  Total:     {total_m:.1f}M")
    print(f"  Frozen:    {frozen_m:.1f}M  (SigLIP 2 backbone)")
    print(f"  Trainable: {trainable_m:.3f}M  (LoRA adapters only) ✓")
    print("PASS\n")


def test_different_inputs_give_different_outputs(encoder: VisualEncoder):
    print("=== Different frames → different embeddings ===")
    frames_a = make_dummy_frames(3)
    frames_b = make_dummy_frames(3)

    with torch.no_grad():
        out_a = encoder.encode_frames(frames_a)
        out_b = encoder.encode_frames(frames_b)

    assert not torch.allclose(out_a, out_b), "Different images gave identical embeddings"
    print("  Different images → different embeddings ✓")
    print("PASS\n")


def test_single_frame(encoder: VisualEncoder):
    print("=== Single frame edge case ===")
    frames = make_dummy_frames(1)

    with torch.no_grad():
        out = encoder.encode_frames(frames)

    assert out.shape == (1, 1152), f"Expected (1, 1152), got {out.shape}"
    print(f"  1 frame → {out.shape} ✓")
    print("PASS\n")


if __name__ == "__main__":
    print(f"Device: {DEVICE.upper()}")
    print("Loading SigLIP 2 So400m (from cache if already downloaded)...")
    encoder = VisualEncoder(device=DEVICE)
    print("Loaded.\n")

    test_output_shape(encoder)
    test_trainable_params(encoder)
    test_different_inputs_give_different_outputs(encoder)
    test_single_frame(encoder)

    print("=" * 40)
    print("All tests passed!")
