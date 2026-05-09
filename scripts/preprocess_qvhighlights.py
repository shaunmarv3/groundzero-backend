"""
preprocess_qvhighlights.py — One-time QVHighlights frame extraction.
Phase 2.1

Run this in a Colab CPU session (NOT locally — needs internet + disk space).

Setup in Colab:
    !git clone https://github.com/YOUR_USERNAME/groundzero.git
    %cd groundzero/groundzero-backend
    !pip install -r requirements.txt datasets huggingface_hub

Change START_IDX / END_IDX for each session:
    Session 1: START_IDX = 0,    END_IDX = 2945
    Session 2: START_IDX = 2945, END_IDX = 5890
    Session 3: START_IDX = 5890, END_IDX = None

Source datasets:
    Annotations : jwnt4/qvhighlights-50frames  (vid, query, relevant_windows)
    Videos      : ayushsdev/qvhighlights-videos (MP4s matched by vid filename)

Output:
    /content/qvhighlights_frames/{vid}/{ts:08.3f}.jpg   (384x384 JPEGs, 1fps)
    /content/annotations_train.jsonl                    (one JSON per video)
    Uploaded to HF_REPO_ID on HuggingFace every UPLOAD_EVERY videos.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config — set these as env vars in a Colab cell before running ─────────────
#
#   import os
#   os.environ["HF_TOKEN"]   = "hf_xxxxxxxxxxxx"
#   os.environ["HF_REPO_ID"] = "your_username/qvhighlights-1fps"
#   os.environ["START_IDX"]  = "0"       # Session 1: 0,    Session 2: 2945, Session 3: 5890
#   os.environ["END_IDX"]    = "2945"    # Session 1: 2945, Session 2: 5890, Session 3: None
#
_end = os.environ.get("END_IDX", "2945")
START_IDX    = int(os.environ.get("START_IDX",    "0"))
END_IDX      = int(_end) if _end.lower() != "none" else None
HF_REPO_ID   = os.environ.get("HF_REPO_ID",   "YOUR_USERNAME/qvhighlights-1fps")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
FRAMES_ROOT  = os.environ.get("FRAMES_ROOT",  "/content/qvhighlights_frames")
JSONL_PATH   = os.environ.get("JSONL_PATH",   "/content/annotations_train.jsonl")
FPS          = float(os.environ.get("FPS",    "1.0"))
FRAME_SIZE   = (384, 384)
UPLOAD_EVERY = int(os.environ.get("UPLOAD_EVERY", "50"))

assert HF_TOKEN, "Set os.environ['HF_TOKEN'] before running"
assert "YOUR_USERNAME" not in HF_REPO_ID, "Set os.environ['HF_REPO_ID'] before running"
# ─────────────────────────────────────────────────────────────────────────────

import re
import json
import requests
from pathlib import Path

from huggingface_hub import HfApi, login
from PIL import Image

from app.pipeline.frame_extractor import extract_frames

Path(FRAMES_ROOT).mkdir(parents=True, exist_ok=True)
login(token=HF_TOKEN)
api = HfApi()


def _extract_query(conversations: list) -> str:
    content = conversations[0]["content"]
    m = re.search(r'\*\*Activity\*\*: (.+?)\n', content)
    return m.group(1).strip() if m else ""


def _get_gt_span(relevant_windows: list) -> tuple:
    starts = [w[0] for w in relevant_windows]
    ends   = [w[1] for w in relevant_windows]
    return float(min(starts)), float(max(ends))


def _download_mp4(vid: str, save_path: Path) -> bool:
    url = (
        f"https://huggingface.co/datasets/ayushsdev/qvhighlights-videos"
        f"/resolve/main/{vid[0].lower()}/{vid}.mp4"
    )
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        return False
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
    return True


def _upload_checkpoint(new_vids: list):
    """Upload all new video frame folders in ONE commit + updated JSONL in one more."""
    print(f"  Uploading {len(new_vids)} video folders in one commit...", flush=True)

    # Collect all new JPEG paths as (local_path, repo_path) pairs
    from huggingface_hub import CommitOperationAdd
    operations = []
    for vid in new_vids:
        vid_frame_dir = Path(FRAMES_ROOT) / vid
        if not vid_frame_dir.exists():
            continue
        for jpg in sorted(vid_frame_dir.glob("*.jpg")):
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"frames/{vid}/{jpg.name}",
                    path_or_fileobj=str(jpg),
                )
            )
    # Add JSONL to same commit
    operations.append(
        CommitOperationAdd(
            path_in_repo="annotations_train.jsonl",
            path_or_fileobj=JSONL_PATH,
        )
    )

    api.create_commit(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add {len(new_vids)} videos",
    )
    print("  Checkpoint uploaded ✓", flush=True)


# ── Load annotations (text only — no image pixels in this dataset) ────────────
# The JSON is stored as {qid: sample_dict}, not a list — load_dataset can't
# parse this format. Download the raw file and parse it manually instead.
print("Downloading annotation JSON from jwnt4/qvhighlights-50frames...")
from huggingface_hub import hf_hub_download
json_path = hf_hub_download(
    repo_id="jwnt4/qvhighlights-50frames",
    filename="p1/train_v1.json",
    repo_type="dataset",
)
with open(json_path, encoding="utf-8") as f:
    raw = json.load(f)  # {qid_str: {vid, conversations, relevant_windows, ...}}
samples = list(raw.values())
print(f"Total samples: {len(samples)}")

chunk = samples[START_IDX : END_IDX]
print(f"Processing {len(chunk)} videos (indices {START_IDX}–{END_IDX or len(samples)})")

# ── Main processing loop ──────────────────────────────────────────────────────
done       = 0
skipped    = 0
batch_vids = []   # tracks vids processed since last upload

with open(JSONL_PATH, "a", encoding="utf-8") as jsonl_f:
    for i, sample in enumerate(chunk):
        vid     = sample["vid"]
        tmp_mp4 = Path(f"/tmp/{vid}.mp4")

        try:
            query = _extract_query(sample["conversations"])
            if not query:
                skipped += 1
                continue

            gt_start_sec, gt_end_sec = _get_gt_span(sample["relevant_windows"])
            duration = float(sample["duration"])

            # 1. Download MP4 (~8MB, deleted immediately after extraction)
            if not _download_mp4(vid, tmp_mp4):
                print(f"  [SKIP] download failed: {vid}")
                skipped += 1
                continue

            # 2. Extract 1fps frames
            frames_data = extract_frames(tmp_mp4, fps=FPS)
            tmp_mp4.unlink(missing_ok=True)

            if not frames_data:
                skipped += 1
                continue

            # 3. Save as 384×384 JPEGs, filename = timestamp
            frame_dir = Path(FRAMES_ROOT) / vid
            frame_dir.mkdir(exist_ok=True)
            timestamps = []
            for ts, img in frames_data:
                img.resize(FRAME_SIZE, Image.LANCZOS).save(
                    frame_dir / f"{ts:08.3f}.jpg", quality=85
                )
                timestamps.append(ts)

            # 4. Write annotation
            jsonl_f.write(json.dumps({
                "vid":          vid,
                "query":        query,
                "duration":     duration,
                "gt_start_sec": gt_start_sec,
                "gt_end_sec":   gt_end_sec,
                "n_frames":     len(timestamps),
            }) + "\n")
            jsonl_f.flush()

            done += 1
            batch_vids.append(vid)

            if done % 50 == 0:
                print(f"  [{done}/{len(chunk)}] done={done}  skipped={skipped}",
                      flush=True)

            if done % UPLOAD_EVERY == 0:
                _upload_checkpoint(batch_vids)
                batch_vids = []   # reset — only upload NEW ones next time

        except Exception as e:
            print(f"  [ERROR] {vid}: {e}")
            tmp_mp4.unlink(missing_ok=True)
            skipped += 1

_upload_checkpoint(batch_vids)   # final upload of remaining videos
print(f"\nFinished. done={done}  skipped={skipped}")
