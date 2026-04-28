"""
scripts/run_baseline.py — End-to-End Zero-Shot Baseline Demo
============================================================
Connects the frame extractor + SigLIP 2 into a working temporal
grounding pipeline — no training, pure zero-shot matching.

Run from groundzero-backend/:
    python scripts/run_baseline.py

Change VIDEO and QUERIES at the top to try your own inputs.
"""

import sys
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModel

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.baseline import siglip_zeroshot_baseline

# ------------------------------------------------------------------ #
#  Config — change these to try different inputs                       #
# ------------------------------------------------------------------ #
MODEL_ID  = "google/siglip2-so400m-patch14-384"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# Test video (the hero background from the frontend)
VIDEO = Path("d:/groundzero/groundzero-frontend/public/Camera_Lens_Video_Generation-enhanced.mp4")

# Queries to test — try anything descriptive
QUERIES = [
    "a camera lens focusing",
    "light reflecting off a surface",
    "a fast moving object",
    "darkness or a black screen",
    "bright light or explosion",
]


# ------------------------------------------------------------------ #
#  Load model (cached after first run)                                 #
# ------------------------------------------------------------------ #
def load_model():
    print(f"\n{'='*60}")
    print(f"  Loading SigLIP 2 So400m (from cache)...")
    print(f"  Device: {DEVICE.upper()}")
    print(f"{'='*60}\n")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16)
    model     = model.to(DEVICE).eval()
    return processor, model


# ------------------------------------------------------------------ #
#  Run baseline for each query                                         #
# ------------------------------------------------------------------ #
def run_all_queries(processor, model):
    print(f"  Video: {VIDEO.name}\n")

    results = []
    for i, query in enumerate(QUERIES):
        print(f"  Query {i+1}/{len(QUERIES)}: \"{query}\"")
        t0     = time.perf_counter()
        result = siglip_zeroshot_baseline(
            video_path=VIDEO,
            query=query,
            model=model,
            processor=processor,
            fps=1.0,
            window_sec=2.0,       # ±2s for this short 8s test video
            device=DEVICE,
        )
        elapsed = time.perf_counter() - t0

        print(f"  → Best frame:  t={result['best_ts']:.2f}s  (score={result['best_score']:+.4f})")
        print(f"  → Predicted:   [{result['pred_start']:.1f}s, {result['pred_end']:.1f}s]")
        print(f"  → Latency:     {elapsed:.2f}s for {result['n_frames']} frames\n")

        results.append((query, result))

    return results


# ------------------------------------------------------------------ #
#  Print full similarity table                                         #
# ------------------------------------------------------------------ #
def print_full_table(results: list):
    print(f"\n{'='*60}")
    print("  FULL SIMILARITY TABLE (all frames × all queries)")
    print(f"{'='*60}")

    # Get timestamps from first result
    all_ts = results[0][1]["all_ts"]

    # Header
    print(f"\n  {'t (s)':<8}", end="")
    for query, _ in results:
        short = query[:16] + "…" if len(query) > 17 else query
        print(f"  {short:<18}", end="")
    print()
    print(f"  {'─'*8}", end="")
    for _ in results:
        print(f"  {'─'*18}", end="")
    print()

    # Each frame row
    for frame_i, ts in enumerate(all_ts):
        print(f"  {ts:<8.2f}", end="")
        for query, result in results:
            score  = result["all_scores"][frame_i]
            is_best = frame_i == result["all_scores"].index(max(result["all_scores"]))
            marker = " ✅" if is_best else "   "
            print(f"  {score:+.4f}{marker:<11}", end="")
        print()

    print()


# ------------------------------------------------------------------ #
#  Key takeaway                                                        #
# ------------------------------------------------------------------ #
def print_conclusion(results: list):
    print(f"{'='*60}")
    print("  WHAT THIS DEMONSTRATES")
    print(f"{'='*60}")
    print()
    print("  ✅ The full zero-shot pipeline just ran end-to-end:")
    print("     video → frames → SigLIP 2 visual tower → cosine sim → timestamp")
    print()
    print("  ⚠️  ALL frames are getting VERY similar scores (~0.01–0.05 range).")
    print("     This is the core problem:")
    print()
    print("     1. Consecutive frames look nearly identical → SigLIP 2 can't")
    print("        tell which one is the START vs END of an event.")
    print()
    print("     2. The ±2s fixed window is a guess. If the query matches")
    print("        a 30-second event, ±2s will be completely wrong.")
    print()
    print("     3. SigLIP 2 was trained on still photos, not video moments.")
    print("        Abstract queries ('focusing', 'reflecting') don't map")
    print("        cleanly to individual frames.")
    print()
    print("  This is EXACTLY why we build the 3 custom modules:")
    print("     Temporal Context Module  → each frame knows ±60s of context")
    print("     Cross-Modal Transformer  → query searches the full sequence")
    print("     Span Extraction Head     → predicts real start AND end, not ±5s")
    print()
    print(f"{'='*60}\n")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    print("\n" + "="*60)
    print("  GroundZero — Zero-Shot Baseline (Phase 3.1)")
    print("  frame_extractor + SigLIP 2 → temporal prediction")
    print("="*60)

    if not VIDEO.exists():
        print(f"\n  ❌ Video not found: {VIDEO}")
        sys.exit(1)

    processor, model = load_model()

    print(f"\n{'─'*60}")
    print("  RUNNING BASELINE")
    print(f"{'─'*60}")

    results = run_all_queries(processor, model)
    print_full_table(results)
    print_conclusion(results)


if __name__ == "__main__":
    main()

# ============================================================
#   GroundZero — Zero-Shot Baseline (Phase 3.1)
#   frame_extractor + SigLIP 2 → temporal prediction
# ============================================================

# ============================================================
#   Loading SigLIP 2 So400m (from cache)...
#   Device: CUDA
# ============================================================

# Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.    

# ────────────────────────────────────────────────────────────
#   RUNNING BASELINE
# ────────────────────────────────────────────────────────────
#   Video: Camera_Lens_Video_Generation-enhanced.mp4

#   Query 1/5: "a camera lens focusing"
#   → Best frame:  t=7.00s  (score=+0.0231)
#   → Predicted:   [5.0s, 8.0s]
#   → Latency:     8.72s for 8 frames

#   Query 2/5: "light reflecting off a surface"
#   → Best frame:  t=6.00s  (score=+0.0356)
#   → Predicted:   [4.0s, 8.0s]
#   → Latency:     5.83s for 8 frames

#   Query 3/5: "a fast moving object"
#   → Best frame:  t=7.00s  (score=+0.0333)
#   → Predicted:   [5.0s, 8.0s]
#   → Latency:     5.80s for 8 frames

#   Query 4/5: "darkness or a black screen"
#   → Best frame:  t=7.00s  (score=+0.0476)
#   → Predicted:   [5.0s, 8.0s]
#   → Latency:     5.83s for 8 frames

#   Query 5/5: "bright light or explosion"
#   → Best frame:  t=7.00s  (score=+0.0193)
#   → Predicted:   [5.0s, 8.0s]
#   → Latency:     5.60s for 8 frames


# ============================================================
#   FULL SIMILARITY TABLE (all frames × all queries)
# ============================================================

#   t (s)     a camera lens fo…   light reflecting…   a fast moving ob…   darkness or a bl…   bright light or …

#   ────────  ──────────────────  ──────────────────  ──────────────────  ──────────────────  ──────────────────
#   0.00      +0.0067             +0.0216             +0.0004             +0.0212             +0.0075          

#   1.00      +0.0097             +0.0204             +0.0001             +0.0188             +0.0076          

#   2.00      +0.0056             +0.0210             +0.0095             +0.0255             +0.0050          

#   3.00      +0.0153             +0.0220             +0.0129             +0.0307             +0.0027          

#   4.00      +0.0186             +0.0280             +0.0196             +0.0371             +0.0091          

#   5.00      +0.0226             +0.0281             +0.0282             +0.0448             +0.0138          

#   6.00      +0.0197             +0.0356 ✅           +0.0289             +0.0448             +0.0157         

#   7.00      +0.0231 ✅           +0.0331             +0.0333 ✅           +0.0476 ✅           +0.0193 ✅    


# ============================================================
#   WHAT THIS DEMONSTRATES
# ============================================================

#   ✅ The full zero-shot pipeline just ran end-to-end:
#      video → frames → SigLIP 2 visual tower → cosine sim → timestamp

#   ⚠️  ALL frames are getting VERY similar scores (~0.01–0.05 range).
#      This is the core problem:

#      1. Consecutive frames look nearly identical → SigLIP 2 can't
#         tell which one is the START vs END of an event.

#      2. The ±2s fixed window is a guess. If the query matches
#         a 30-second event, ±2s will be completely wrong.

#      3. SigLIP 2 was trained on still photos, not video moments.
#         Abstract queries ('focusing', 'reflecting') don't map
#         cleanly to individual frames.

#   This is EXACTLY why we build the 3 custom modules:
#      Temporal Context Module  → each frame knows ±60s of context
#      Cross-Modal Transformer  → query searches the full sequence
#      Span Extraction Head     → predicts real start AND end, not ±5s

# ============================================================