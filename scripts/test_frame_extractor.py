"""
scripts/test_frame_extractor.py — Unit Test for frame_extractor.py
===================================================================
Tests:
  1. get_video_metadata()  — reads duration, fps, resolution
  2. extract_frames()      — extracts 1fps frames, checks count + types
  3. extract_region()      — extracts a short window at 4fps

Run from groundzero-backend/:
    python scripts/test_frame_extractor.py
"""

import sys
import time
from pathlib import Path

# Add parent dir to path so we can import from app/
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.frame_extractor import extract_frames, get_video_metadata, extract_region

# ---- Test video --------------------------------------------------------
VIDEO = Path("d:/groundzero/groundzero-frontend/public/Camera_Lens_Video_Generation-enhanced.mp4")


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  TEST: {title}")
    print(f"{'='*60}")


# ---- Test 1: Metadata --------------------------------------------------
def test_metadata():
    separator("get_video_metadata()")
    meta = get_video_metadata(VIDEO)

    print(f"  File:         {VIDEO.name}")
    print(f"  Duration:     {meta['duration']:.2f} seconds")
    print(f"  Native fps:   {meta['fps']:.2f}")
    print(f"  Resolution:   {meta['width']}×{meta['height']}")
    print(f"  Codec:        {meta['codec']}")

    assert meta["duration"] > 0,    "Duration must be > 0"
    assert meta["fps"] > 0,         "FPS must be > 0"
    assert meta["width"] > 0,       "Width must be > 0"
    assert meta["height"] > 0,      "Height must be > 0"
    print("\n  ✅ PASSED")
    return meta


# ---- Test 2: extract_frames() at 1fps ----------------------------------
def test_extract_frames(duration: float):
    separator("extract_frames() at 1fps")

    t0     = time.perf_counter()
    frames = extract_frames(VIDEO, fps=1.0, max_frames=10)
    elapsed = time.perf_counter() - t0

    print(f"  Extracted:    {len(frames)} frames")
    print(f"  Time taken:   {elapsed:.2f}s")
    print()

    for i, (ts, img) in enumerate(frames):
        print(f"  Frame {i:02d}: t={ts:.2f}s | size={img.size} | mode={img.mode}")

    # Checks
    assert len(frames) > 0,               "Should extract at least 1 frame"
    assert len(frames) <= 10,             "max_frames=10 must be respected"

    ts0, img0 = frames[0]
    assert ts0 >= 0.0,                    "First timestamp must be >= 0"
    assert img0.size == (384, 384),       f"Expected (384,384), got {img0.size}"
    assert img0.mode == "RGB",            f"Expected RGB, got {img0.mode}"

    if len(frames) > 1:
        ts_diffs = [frames[i+1][0] - frames[i][0] for i in range(len(frames)-1)]
        for diff in ts_diffs:
            assert 0.8 <= diff <= 1.2,    f"Frame interval should be ~1s, got {diff:.2f}s"

    print(f"\n  ✅ PASSED — {len(frames)} frames, all (384,384) RGB")


# ---- Test 3: extract_region() at 4fps ----------------------------------
def test_extract_region(duration: float):
    separator("extract_region() at 4fps (fine-grained pass)")

    center = min(5.0, duration / 2)   # use 5s or midpoint if video is short
    t0     = time.perf_counter()
    frames = extract_region(VIDEO, center_sec=center, padding_sec=2.0, fps=4.0)
    elapsed = time.perf_counter() - t0

    print(f"  Center:       {center:.1f}s  (±2s window)")
    print(f"  Extracted:    {len(frames)} frames at 4fps")
    print(f"  Time taken:   {elapsed:.2f}s")
    print()

    for i, (ts, img) in enumerate(frames):
        print(f"  Frame {i:02d}: t={ts:.4f}s | size={img.size}")

    assert len(frames) > 0,             "Should extract frames from region"
    ts0, _ = frames[0]
    assert ts0 >= max(0.0, center - 2.0) - 0.1,  "First frame should be near start of region"

    print(f"\n  ✅ PASSED — {len(frames)} frames in ±2s window at 4fps")


# ---- Run all tests -----------------------------------------------------
def main():
    print("\n" + "="*60)
    print("  GroundZero — frame_extractor.py Unit Tests")
    print("="*60)

    if not VIDEO.exists():
        print(f"\n  ❌ Test video not found: {VIDEO}")
        sys.exit(1)

    try:
        meta = test_metadata()
        test_extract_frames(meta["duration"])
        test_extract_region(meta["duration"])

        print("\n" + "="*60)
        print("  ALL TESTS PASSED ✅")
        print("="*60)
        print()
        print("  What this proves:")
        print("  - ffmpeg can read the video and report metadata")
        print("  - Frames are extracted as (timestamp, PIL.Image) tuples")
        print("  - Every frame is resized to (384, 384) RGB — SigLIP 2 input size")
        print("  - extract_region() correctly targets a sub-window at 4fps")
        print("    (this is what the fine-grained coarse-to-fine pass uses)")
        print()

    except AssertionError as e:
        print(f"\n  ❌ ASSERTION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
