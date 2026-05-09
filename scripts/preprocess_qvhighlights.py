"""
preprocess_qvhighlights.py — One-time QVHighlights frame extraction.
Phase 2.1

Run this in a Colab CPU session (NOT locally — needs internet + disk space).

Setup in Colab:
    !git clone https://github.com/YOUR_USERNAME/groundzero.git
    %cd groundzero/groundzero-backend
    !pip install "huggingface_hub>=0.40.0" -q   # must be >=0.40 for upload_large_folder
    !pip install -r requirements.txt datasets

Change START_IDX / END_IDX for each session:
    Session 1: START_IDX =    0, END_IDX =  400   (done — 229 frames, 400 JSONL merged)
    Session 2: START_IDX =  400, END_IDX = 1200
    Session 3: START_IDX = 1200, END_IDX = None

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
FRAMES_ROOT  = os.environ.get("FRAMES_ROOT",  "/content/hf_upload/frames")
JSONL_PATH   = os.environ.get("JSONL_PATH",   "/content/annotations_train.jsonl")
FPS          = float(os.environ.get("FPS",    "1.0"))
FRAME_SIZE   = (384, 384)
UPLOAD_EVERY = int(os.environ.get("UPLOAD_EVERY", "100"))

assert HF_TOKEN, "Set os.environ['HF_TOKEN'] before running"
assert "YOUR_USERNAME" not in HF_REPO_ID, "Set os.environ['HF_REPO_ID'] before running"
# ─────────────────────────────────────────────────────────────────────────────

import re
import json
import requests
from pathlib import Path

from huggingface_hub import HfApi, login, hf_hub_download
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
    """Upload new frames via upload_large_folder (resumable) + JSONL via upload_file."""
    print(f"  Uploading checkpoint ({len(new_vids)} new videos)...", flush=True)

    # upload_large_folder hashes all files and only uploads what's new —
    # safe to call repeatedly, never re-uploads already-committed files.
    # hf_upload/ contains frames/ so files land at frames/{vid}/ in the repo.
    api.upload_large_folder(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        folder_path=str(Path(FRAMES_ROOT).parent),  # /content/hf_upload
        num_workers=4,
        print_report=False,
    )

    api.upload_file(
        path_or_fileobj=JSONL_PATH,
        path_in_repo="annotations_train.jsonl",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print("  Checkpoint uploaded ✓", flush=True)


# ── Restore existing JSONL from HuggingFace (prevents overwriting on new session)
import shutil
try:
    existing = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="annotations_train.jsonl",
        repo_type="dataset",
    )
    shutil.copy(existing, JSONL_PATH)
    n = open(JSONL_PATH).read().count("\n")
    print(f"Restored {n} existing annotations from HuggingFace")
except Exception:
    print("No existing JSONL on HuggingFace — starting fresh")

# ── Load annotations (text only — no image pixels in this dataset) ────────────
# The JSON is stored as {qid: sample_dict}, not a list — load_dataset can't
# parse this format. Download the raw file and parse it manually instead.
print("Downloading annotation JSON from jwnt4/qvhighlights-50frames...")
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
