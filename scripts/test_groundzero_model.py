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






#  1. is empty _init__.py empty proper ???
#   2. ewhat do u mean by  dtype mismatch in inference path: encode_frames() and
#   encode_query()
#     return float16 (SigLIP 2 is loaded in float16), but
#     temporal_context/cross_modal/span_head are float32. nn.MultiheadAttention
#     will crash with mixed dtypes when predict() is actually called and   Bug: float16 vs
#   float32 dtype mismatch — encode_frames()/encode_query() return float16
#       (SigLIP), cross_modal/span_head are float32 → MultiheadAttention crash in predict()
#     File: groundzero_model.py
#     Fix: Added .float() after both encode calls in predict() and _run_pass() and   One thing
#    worth knowing: the dtype fix (.float()) means the training loop must also call
#     .float() after encode_frames() before passing tensors to forward(). Since the smoke
#   tests
#     use torch.randn() (already float32) they passed fine, but real training with cached
#   SigLIP
#     tensors would have hit the same crash. Ready for Phase 4 whenever you are.


#   2. The dtype mismatch — explained step by step

#   SigLIP 2 is 1.1 billion parameters. In float32, that's 1.1B × 4 bytes = 4.4 GB of GPU
#   memory. So we load it in float16 (torch_dtype=torch.float16) — half the memory, 2.2 GB
#   instead.

#   Our custom modules (TemporalContext, CrossModal, SpanHead) are small (154M params) and stay
#    in float32 because training is more stable in float32.

#   So the model has two halves with different dtypes:
#   SigLIP 2 (float16) → encode_frames() → float16 tensor
#                      → encode_query()  → float16 tensor

#   TemporalContext  (float32 weights)
#   CrossModal       (float32 weights)
#   SpanHead         (float32 weights)

#   Here's exactly where the crash happens in _run_pass():
#   # Step 1: encode_frames returns float16 (SigLIP is float16)
#   frame_embs = visual_encoder.encode_frames(pil_imgs)   # dtype=float16

#   # Step 2: temporal_context — SURVIVES by accident
#   # Inside temporal_context.forward():
#   #   x = x + self.pos_enc(timestamps)
#   #   pos_enc creates float32 tensors
#   #   float16 + float32 → PyTorch silently upcasts → x is now float32
#   #   Conv1d (float32 weights) receives float32 input → OK
#       # Step 3: encode_query returns float16 (SigLIP is float16)
#   query_emb = text_encoder.encode_query(query)           # dtype=float16
#   # query_emb never hits pos_enc, stays float16

#   # Step 4: CrossModalTransformer — CRASHES HERE
#   # Inside CrossAttentionBlock:
#   self.attn(
#       query=query_emb,   # float16  ← different!
#       key=frame_embs,    # float32  (was upcasted by pos_enc)
#       value=frame_embs,  # float32
#   )
#   # nn.MultiheadAttention requires all three to be the same dtype
#   # RuntimeError: expected scalar type Half but found Float

#   The fix (.float()) forces everything to float32 right after encoding,
#   before any custom module touches it:
#   frame_embs = visual_encoder.encode_frames(...).float()   # float16 →
#   float32
#   query_emb  = text_encoder.encode_query(...).float()      # float16 →
#   float32
#   # now both are float32, same as all custom modules → no crash

#   Why not just load SigLIP in float32?
#   You'd need 4.4 GB just for the frozen backbone. With float16 you use 2.2
#   GB for frozen SigLIP + ~0.6 GB for custom modules = 2.8 GB total, fits
#   comfortably on a GPU with memory left for training batches.

#   What about the training loop?
#   Same issue. When Phase 5 training loop calls encode_frames() and passes
#   the result to forward(), it must also call .float(). We'll write it
#   correctly in Phase 5, but it's good to know now.






#  You're right to check. The screenshot confirms: 0.9B params (not 400M,
#   and not my estimate of 1.1B — I was slightly off).

#   The name "So400m" is misleading:
#   - "So" = Shape-optimized
#   - "400m" = the vision encoder alone is ~400M
#   - But the full model = vision tower (~400M) + text tower (~500M) = ~0.9B
#   total

#   Our test confirmed this — count_params() showed 1136M per encoder, but
#   that's because both VisualEncoder and TextEncoder load the full model
#   (both towers together) and then only use one tower each. So each holds
#   0.9B params even though it only uses half.

#   VisualEncoder loads full SigLIP 2 → 1136M counted (uses vision side only)
#   TextEncoder   loads full SigLIP 2 → 1136M counted (uses text side only)
#   Total in memory: 2 × ~1.1B = ~2.2B params frozen  ← wasteful, fix in
#   Phase 5
#   Actual unique params: ~0.9B

#   The dtype issue still applies regardless of size:
#   - 0.9B × 2 bytes (float16) = 1.8 GB GPU memory for SigLIP
#   - 0.9B × 4 bytes (float32) = 3.6 GB — that's why we load in float16

#   The .float() fix stands — SigLIP outputs float16 regardless of whether
#   it's 0.9B or 1.1B.



#   ┌─────────────────────┬──────────────────────────────────────────────────────┐
#   │        File         │                        Change                        │
#   ├─────────────────────┼──────────────────────────────────────────────────────┤
#   │ visual_encoder.py   │ Removed torch_dtype=torch.float16 — loads float32,   │
#   │                     │ updated docstrings                                   │
#   ├─────────────────────┼──────────────────────────────────────────────────────┤
#   │ text_encoder.py     │ Removed torch_dtype=torch.float16 — loads float32    │
#   ├─────────────────────┼──────────────────────────────────────────────────────┤
#   │ run_baseline.py     │ Removed dtype=torch.float16                          │
#   ├─────────────────────┼──────────────────────────────────────────────────────┤
#   │ groundzero_model.py │ Removed .float() casts — no longer needed            │
#   ├─────────────────────┼──────────────────────────────────────────────────────┤
#   │ baseline.py         │ Removed .float() on normalize calls                  │
#   └─────────────────────┴──────────────────────────────────────────────────────┘

#   The dtype mismatch bug is eliminated entirely — everything is float32 from end
#   to end. On Colab (T4, 15 GB) the double-backbone load will use ~7.2 GB which
#   still fits; after Phase 5 shares one backbone it drops to ~3.6 GB. Ready for
#   Phase 4?