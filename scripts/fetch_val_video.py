"""
fetch_val_video.py — pull ONE real QVHighlights val clip from YouTube for Chunk 4.

The dataset only stores frames + embeddings, so we reconstruct the actual video:
  vid = "<youtube_id>_<start>_<end>"  ->  download that YouTube section, slice [start,end].

Saves test_videos/qvh_<vid>.mp4 and prints the real query + ground-truth window
(gt_start_sec / gt_end_sec) so we can judge /api/predict against a known answer.

Fast + observable: caps at 360p, uses range-based section download (no whole-video
keyframe re-encode), prints live download %, and falls through to the next candidate
if a video is unavailable.

Usage:
    python scripts/fetch_val_video.py            # auto-pick first working val clip
    python scripts/fetch_val_video.py <vid>      # force a specific vid
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
import imageio_ffmpeg
import yt_dlp

REPO = "shaunmarvell/qvhighlights-1fps"
BACKEND = Path(__file__).resolve().parent.parent
OUT_DIR = BACKEND / "test_videos"
OUT_DIR.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MAX_TRIES = 8  # try up to N val clips before giving up


def parse_vid(vid: str):
    """'-4Mlqc7PbZY_210.0_360.0' -> ('-4Mlqc7PbZY', 210.0, 360.0)."""
    yid, start, end = vid.rsplit("_", 2)
    return yid, float(start), float(end)


def _hook(d):
    if d.get("status") == "downloading":
        pct = d.get("_percent_str", "?").strip()
        spd = d.get("_speed_str", "?").strip()
        eta = d.get("_eta_str", "?").strip()
        print(f"    dl {pct}  {spd}  eta {eta}", flush=True)
    elif d.get("status") == "finished":
        print("    download finished, post-processing...", flush=True)


def try_fetch(r) -> bool:
    vid = r["vid"]
    yid, start, end = parse_vid(vid)
    url = f"https://www.youtube.com/watch?v={yid}"
    print(f"\n=== trying {vid} ===", flush=True)
    print(f"  query: {r.get('query')}", flush=True)
    print(f"  GT: [{r.get('gt_start_sec')}, {r.get('gt_end_sec')}]s  duration: {r.get('duration')}s", flush=True)
    print(f"  youtube: {url}  slice [{start}, {end}]", flush=True)

    tmpdir = Path(tempfile.mkdtemp())
    ydl_opts = {
        # cap at 360p to keep the download tiny
        "format": "best[height<=360][ext=mp4]/worst[ext=mp4]/worst",
        "outtmpl": str(tmpdir / "sec.%(ext)s"),
        # range-based section download (no full-video keyframe re-encode -> fast)
        "download_ranges": yt_dlp.utils.download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": False,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_hook],
        "retries": 2,
        "socket_timeout": 20,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:  # noqa
        print(f"  yt-dlp FAILED: {e}", flush=True)
        return False

    dl = next(iter(tmpdir.glob("sec.*")), None)
    if dl is None:
        print("  no file produced", flush=True)
        return False

    # re-encode to a clean, seekable mp4 (the section is already ~[start,end])
    out_mp4 = OUT_DIR / f"qvh_{vid}.mp4"
    cmd = [FFMPEG, "-y", "-i", str(dl), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-an", str(out_mp4)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg FAILED: {e.stderr.decode(errors='ignore')[:400]}", flush=True)
        return False

    mb = out_mp4.stat().st_size / 1e6
    print(f"\nWROTE: {out_mp4}  ({mb:.2f} MB)", flush=True)
    print("\n--- USE THIS FOR /api/predict ---", flush=True)
    print(f"  video : {out_mp4}", flush=True)
    print(f"  query : {r.get('query')}", flush=True)
    print(f"  GT    : [{r.get('gt_start_sec')}, {r.get('gt_end_sec')}] s (relative to clip start)", flush=True)
    return True


def main():
    ann = hf_hub_download(REPO, "annotations_val.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(ann, encoding="utf-8") if l.strip()]
    print(f"val rows: {len(rows)} | keys: {list(rows[0].keys())}", flush=True)

    want = sys.argv[1] if len(sys.argv) > 1 else None
    tried = 0
    for r in rows:
        if want and r["vid"] != want:
            continue
        if try_fetch(r):
            return
        tried += 1
        if want or tried >= MAX_TRIES:
            break
    raise SystemExit(f"could not fetch a val video after {tried} attempt(s)")


if __name__ == "__main__":
    main()
