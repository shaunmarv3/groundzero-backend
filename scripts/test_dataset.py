"""
test_dataset.py — Unit tests for app/training/dataset.py
Phase 5.1

No real videos needed — creates temp directories with dummy JPEGs and a
dummy JSONL, then tests all dataset behaviors.

Run from groundzero-backend/:
    python scripts/test_dataset.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from PIL import Image

from app.training.dataset import GroundingDataset, collate_fn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_dataset(tmp_dir: Path, n_videos: int = 4, n_frames: int = 30,
                  augment: bool = False) -> GroundingDataset:
    frames_root = tmp_dir / "frames"
    jsonl_path  = tmp_dir / "annotations.jsonl"
    duration    = float(n_frames)  # 1fps → duration = n_frames seconds

    with open(jsonl_path, "w") as f:
        for v in range(n_videos):
            vid = f"vid_{v:04d}"
            # Create n_frames dummy 384×384 JPEGs
            frame_dir = frames_root / vid
            frame_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_frames):
                ts = float(i) + 0.5                  # 0.5, 1.5, 2.5, ...
                color = ((v * 40) % 256, (i * 8) % 256, 128)
                Image.new("RGB", (384, 384), color=color).save(
                    frame_dir / f"{ts:08.3f}.jpg"
                )
            # GT span: middle third of the video
            gt_start = duration / 3
            gt_end   = 2 * duration / 3
            f.write(json.dumps({
                "vid":          vid,
                "query":        f"test query for video {v}",
                "duration":     duration,
                "gt_start_sec": gt_start,
                "gt_end_sec":   gt_end,
                "n_frames":     n_frames,
            }) + "\n")

    return GroundingDataset(jsonl_path, frames_root, augment=augment)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_length():
    print("=== GroundingDataset: __len__ ===")
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_dataset(Path(tmp), n_videos=4)
        assert len(ds) == 4, f"Expected 4, got {len(ds)}"
    print("  len=4 ✓")


def test_getitem_keys():
    print("=== GroundingDataset: __getitem__ returns correct keys ===")
    with tempfile.TemporaryDirectory() as tmp:
        ds     = _make_dataset(Path(tmp))
        sample = ds[0]
        expected = {"frames", "timestamps", "query", "gt_start_idx",
                    "gt_end_idx", "is_negative", "duration", "vid"}
        assert set(sample.keys()) == expected, f"Missing keys: {expected - set(sample.keys())}"
    print("  all keys present ✓")


def test_frames_are_pil():
    print("=== GroundingDataset: frames are PIL Images ===")
    with tempfile.TemporaryDirectory() as tmp:
        ds     = _make_dataset(Path(tmp), n_frames=10)
        sample = ds[0]
        assert len(sample["frames"]) == 10, "Wrong frame count"
        assert isinstance(sample["frames"][0], Image.Image), "Frame is not PIL.Image"
        assert sample["frames"][0].size == (384, 384), "Wrong frame size"
    print("  PIL Images, 384×384 ✓")


def test_timestamps_match_frames():
    print("=== GroundingDataset: timestamps align with frames ===")
    with tempfile.TemporaryDirectory() as tmp:
        ds     = _make_dataset(Path(tmp), n_frames=20)
        sample = ds[0]
        assert len(sample["timestamps"]) == len(sample["frames"]), \
            "timestamps and frames length mismatch"
        assert sample["timestamps"] == sorted(sample["timestamps"]), \
            "timestamps not sorted"
    print("  timestamps aligned and sorted ✓")


def test_gt_indices_in_range():
    print("=== GroundingDataset: gt indices in [0, N-1], end >= start ===")
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_dataset(Path(tmp), n_videos=4, n_frames=30)
        for i in range(len(ds)):
            s = ds[i]
            N = len(s["frames"])
            assert 0 <= s["gt_start_idx"] < N,  f"gt_start_idx={s['gt_start_idx']} out of range"
            assert 0 <= s["gt_end_idx"]   < N,  f"gt_end_idx={s['gt_end_idx']} out of range"
            assert s["gt_end_idx"] >= s["gt_start_idx"], "end < start"
    print("  GT indices valid for all samples ✓")


def test_no_augment_deterministic():
    print("=== GroundingDataset: augment=False is deterministic ===")
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_dataset(Path(tmp), augment=False)
        s1 = ds[0]
        s2 = ds[0]
        assert s1["gt_start_idx"] == s2["gt_start_idx"], "Non-deterministic without augment"
        assert s1["query"] == s2["query"], "Query changed without augment"
    print("  deterministic ✓")


def test_is_negative_false_by_default():
    print("=== GroundingDataset: is_negative=False when augment=False ===")
    with tempfile.TemporaryDirectory() as tmp:
        ds = _make_dataset(Path(tmp), augment=False)
        for i in range(len(ds)):
            assert ds[i]["is_negative"] is False, f"Sample {i} wrongly flagged as negative"
    print("  is_negative=False for all samples ✓")


def test_collate_fn_shapes():
    print("=== collate_fn: scalar fields become tensors ===")
    with tempfile.TemporaryDirectory() as tmp:
        ds     = _make_dataset(Path(tmp), n_videos=4, augment=False)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
        batch  = next(iter(loader))

        assert isinstance(batch["frames"], list),   "frames should be list"
        assert len(batch["frames"]) == 2,           "batch size should be 2"
        assert isinstance(batch["query"], list),    "query should be list of str"
        assert batch["gt_start_idx"].shape == (2,), "gt_start_idx shape wrong"
        assert batch["gt_end_idx"].shape   == (2,), "gt_end_idx shape wrong"
        assert batch["is_negative"].dtype  == torch.bool
        assert batch["duration"].dtype     == torch.float32
    print("  shapes correct ✓")


def test_collate_variable_length():
    print("=== collate_fn: handles videos with different N frames ===")
    with tempfile.TemporaryDirectory() as tmp:
        # Create two videos with different frame counts
        frames_root = Path(tmp) / "frames"
        jsonl_path  = Path(tmp) / "annotations.jsonl"
        configs = [("vid_short", 10), ("vid_long", 30)]
        with open(jsonl_path, "w") as f:
            for vid, n in configs:
                frame_dir = frames_root / vid
                frame_dir.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    ts = float(i) + 0.5
                    Image.new("RGB", (384, 384), (100, 100, 100)).save(
                        frame_dir / f"{ts:08.3f}.jpg"
                    )
                f.write(json.dumps({
                    "vid": vid, "query": "test", "duration": float(n),
                    "gt_start_sec": 2.0, "gt_end_sec": float(n) - 2.0,
                    "n_frames": n,
                }) + "\n")

        ds    = GroundingDataset(jsonl_path, frames_root, augment=False)
        batch = collate_fn([ds[0], ds[1]])

        assert len(batch["frames"][0]) == 10, "Short video frame count wrong"
        assert len(batch["frames"][1]) == 30, "Long video frame count wrong"
    print("  variable-length batch handled ✓")


if __name__ == "__main__":
    test_length()
    test_getitem_keys()
    test_frames_are_pil()
    test_timestamps_match_frames()
    test_gt_indices_in_range()
    test_no_augment_deterministic()
    test_is_negative_false_by_default()
    test_collate_fn_shapes()
    test_collate_variable_length()
    print("\nAll dataset tests passed ✓")

# PS D:\groundzero\groundzero-backend> python -u "d:\groundzero\groundzero-backend\scripts\test_dataset.py"
# === GroundingDataset: __len__ ===
#   len=4 ✓
# === GroundingDataset: __getitem__ returns correct keys ===
#   all keys present ✓
# === GroundingDataset: frames are PIL Images ===
#   PIL Images, 384×384 ✓
# === GroundingDataset: timestamps align with frames ===
#   timestamps aligned and sorted ✓
# === GroundingDataset: gt indices in [0, N-1], end >= start ===
#   GT indices valid for all samples ✓
# === GroundingDataset: augment=False is deterministic ===
#   deterministic ✓
# === GroundingDataset: is_negative=False when augment=False ===
#   is_negative=False for all samples ✓
# === collate_fn: scalar fields become tensors ===
#   shapes correct ✓
# === collate_fn: handles videos with different N frames ===
#   variable-length batch handled ✓

# All dataset tests passed ✓
# PS D:\groundzero\groundzero-backend> 