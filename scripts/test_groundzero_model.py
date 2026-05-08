"""
test_groundzero_model.py — Smoke tests for groundzero_model.py

Run from groundzero-backend/:
    python scripts/test_groundzero_model.py

Loads SigLIP 2 once (~30s from cache) then runs 3 tests.
No real video needed — uses random tensors for forward pass tests.
predict() test is skipped (needs a real video file).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from app.pipeline.groundzero_model import GroundZeroModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
B, N, D = 1, 50, 1152


def test_forward_pass(model: GroundZeroModel):
    print("=== Forward pass (training mode) ===")
    frames    = torch.randn(B, N, D, device=DEVICE)
    timestamps = torch.linspace(0, 1, N, device=DEVICE)
    query_emb = torch.randn(B, 1, D, device=DEVICE)

    with torch.no_grad():
        start_logits, end_logits, confidence = model(frames, timestamps, query_emb)

    assert start_logits.shape == (B, N), f"Expected ({B},{N}), got {start_logits.shape}"
    assert end_logits.shape   == (B, N), f"Expected ({B},{N}), got {end_logits.shape}"
    assert confidence.shape   == (B,),   f"Expected ({B},), got {confidence.shape}"
    assert torch.isfinite(start_logits).all(), "NaNs in start_logits"
    assert (confidence >= 0).all() and (confidence <= 1).all(), "confidence out of [0,1]"

    print(f"  frames {frames.shape} + query {query_emb.shape}")
    print(f"  start_logits: {start_logits.shape} ✓")
    print(f"  end_logits:   {end_logits.shape} ✓")
    print(f"  confidence:   {confidence.shape}, value={confidence[0].item():.3f} ✓")
    print("PASS\n")


def test_param_count(model: GroundZeroModel):
    print("=== Parameter count ===")
    counts = model.count_params()

    print(f"  {'Component':<22} {'Trainable':>12}  {'Total':>12}")
    print(f"  {'-'*22} {'-'*12}  {'-'*12}")
    for name, c in counts["components"].items():
        print(f"  {name:<22} {c['trainable']/1e6:>10.3f}M  {c['total']/1e6:>10.1f}M")
    print(f"  {'─'*22} {'─'*12}  {'─'*12}")
    print(f"  {'TOTAL':<22} {counts['total_trainable']/1e6:>10.3f}M  {counts['total_params']/1e6:>10.1f}M")
    print(f"  {'Frozen':<22} {counts['total_frozen']/1e6:>10.1f}M")

    assert counts["total_trainable"] > 0, "No trainable params"
    assert counts["total_frozen"] > counts["total_trainable"], "Frozen should dominate"
    print("PASS\n")


def test_different_queries_different_output(model: GroundZeroModel):
    print("=== Different queries → different logits ===")
    frames     = torch.randn(B, N, D, device=DEVICE)
    timestamps = torch.linspace(0, 1, N, device=DEVICE)
    query_a    = torch.randn(B, 1, D, device=DEVICE)
    query_b    = torch.randn(B, 1, D, device=DEVICE)

    with torch.no_grad():
        sl_a, _, _ = model(frames, timestamps, query_a)
        sl_b, _, _ = model(frames, timestamps, query_b)

    assert not torch.allclose(sl_a, sl_b), "Different queries gave identical start logits"
    print("  Different queries → different start logits ✓")
    print("PASS\n")


if __name__ == "__main__":
    print(f"Device: {DEVICE.upper()}")
    print("Loading GroundZeroModel (SigLIP 2 from cache)...")
    model = GroundZeroModel(device=DEVICE)
    model.eval()
    print("Loaded.\n")

    test_forward_pass(model)
    test_param_count(model)
    test_different_queries_different_output(model)

    print("=" * 40)
    print("All tests passed!")
    print("\nNote: predict() test requires a real video file.")
    print("Run scripts/run_baseline.py to test end-to-end with a real video.")

# PS D:\groundzero\groundzero-backend> python -u "d:\groundzero\groundzero-backend\scripts\test_groundzero_model.py"
# 2026-05-09 02:03:28.573485: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# 2026-05-09 02:03:47.825259: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
# Device: CUDA
# Loading GroundZeroModel (SigLIP 2 from cache)...
# Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
# `torch_dtype` is deprecated! Use `dtype` instead!
# Loaded.

# === Forward pass (training mode) ===
#   frames torch.Size([1, 50, 1152]) + query torch.Size([1, 1, 1152])
#   start_logits: torch.Size([1, 50]) ✓
#   end_logits:   torch.Size([1, 50]) ✓
#   confidence:   torch.Size([1]), value=0.503 ✓
# PASS

# === Parameter count ===
#   Component                 Trainable         Total
#   ---------------------- ------------  ------------
#   visual_encoder              0.147M      1136.2M
#   text_encoder                0.000M      1136.0M
#   temporal_context           26.556M        26.6M
#   cross_modal               127.522M       127.5M
#   span_head                   0.665M         0.7M
#   ────────────────────── ────────────  ────────────
#   TOTAL                     154.890M      2426.9M
#   Frozen                     2272.0M
# PASS

# === Different queries → different logits ===
#   Different queries → different start logits ✓
# PASS

# ========================================
# All tests passed!

# Note: predict() test requires a real video file.
# Run scripts/run_baseline.py to test end-to-end with a real video.
# PS D:\groundzero\groundzero-backend> 








#   Why LoRA is not inside forward() — the real explanation

#   PyTorch builds a computation graph dynamically as tensors
#   flow through operations. The graph doesn't care about
#   function boundaries — it only cares about which tensors
#   touched which parameters.

#   Training loop:

#   Step 1 — call encode_frames() OUTSIDE forward()
#       frame_embs =
#   model.visual_encoder.encode_frames(pil_images)

#       Internally: pil → SigLIP layers → LoRA adapters →
#   frame_embs

#       frame_embs now has a GRADIENT PATH back to LoRA
#   weights.
#       PyTorch is tracking this silently.

#   Step 2 — call forward() with that tensor
#       start_logits, end_logits, conf = model(frame_embs,
#   timestamps, query_emb)

#       Internally: frame_embs → temporal_context → cross_modal
#    → span_head → logits

#       The gradient path extends: logits → ... → frame_embs →
#   LoRA

#   Step 3 — backprop
#       loss.backward()

#       PyTorch walks the graph backwards:
#       loss → span_head → cross_modal → temporal_context →
#   frame_embs → encode_frames() → LoRA weights

#       Internally: frame_embs → temporal_context → cross_modal → span_head → logits

#       The gradient path extends: logits → ... → frame_embs → LoRA

#   Step 3 — backprop
#       loss.backward()

#       PyTorch walks the graph backwards:
#       loss → span_head → cross_modal → temporal_context → frame_embs → encode_frames() → LoRA weights

#       LoRA gets gradients. optimizer.step() updates LoRA. ✓

#   The key: frame_embs is the bridge. It came from LoRA, then got passed into forward(). The gradient path
#   is unbroken even though encode_frames() was called before forward().

#   Why not put encode_frames() inside forward()?

#   If forward() took PIL images directly:
#   def forward(self, pil_images, timestamps, query_str):
#       frame_embs = self.visual_encoder.encode_frames(pil_images)  # inside forward
#       ...

#   This would work for gradients — but now forward() takes PIL images (not tensors). PyTorch's DataLoader,
#   distributed training, torch.compile, gradient checkpointing — all of these expect tensors. PIL images
#   don't batch, don't go on GPU, don't serialize. You'd break the entire training infrastructure.

#   The separation — encode_frames() before forward(), tensors into forward() — is the standard PyTorch
#   pattern. It's not about hiding LoRA from training. It's about keeping forward() tensor-in → tensor-out
#   so all of PyTorch's tooling works.




# ● All 3 tests passed. A few things worth noting from the output:

#   - temporal_context: 26.6M — higher than the 10M estimate. Makes sense: 4×
#   Conv1d(1152, 1152, kernel_size=5) = 1152 × 1152 × 5 × 4 ≈ 26.5M. Correct.
#   - visual_encoder + text_encoder = 1136M + 1136M = SigLIP 2 loaded twice.
#   Both load the full model independently. That's ~2.3GB of frozen weights
#   duplicated on GPU. We'll fix this in Phase 5 by sharing a single SigLIP 2
#   backbone. Not blocking now.
#   - Total trainable: 154.9M — slightly more than the 138M estimate due to
#   temporal_context.