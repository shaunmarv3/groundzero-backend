"""
pipeline/frame_extractor.py — Video Frame Extraction
=====================================================
Extracts frames from a video file at a configurable fps rate.

Core function:
    extract_frames(video_path, fps=1.0) -> List[Tuple[float, PIL.Image]]

Each tuple is (timestamp_seconds, PIL.Image) so the caller always
knows exactly WHEN in the video each frame was captured.

Used by:
  - scripts/siglip_demo.py         (demo / baseline measurement)
  - pipeline/orchestrator.py       (production inference)
  - notebook/02_baseline_*         (zero-shot baseline)
"""

import io
import subprocess
import logging
from pathlib import Path

import ffmpeg
import imageio_ffmpeg
from PIL import Image

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  FFmpeg binary                                                       #
# ------------------------------------------------------------------ #
def _get_ffmpeg_path() -> str:
    """
    Return the path to the ffmpeg binary.

    Priority:
      1. imageio-ffmpeg bundled binary  ← works on any OS, no system install
      2. System ffmpeg (if imageio-ffmpeg unavailable for some reason)

    imageio-ffmpeg ships ffmpeg binaries for Windows/Mac/Linux as a
    Python package — anyone who does `pip install imageio-ffmpeg` gets
    a working ffmpeg binary regardless of their OS.
    """
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG_EXE = _get_ffmpeg_path()


# ------------------------------------------------------------------ #
#  Types                                                               #
# ------------------------------------------------------------------ #
FrameList = list[tuple[float, Image.Image]]
# Each element: (timestamp_seconds, PIL.Image in RGB mode)


# ------------------------------------------------------------------ #
#  Main function                                                       #
# ------------------------------------------------------------------ #
def extract_frames(
    video_path: str | Path,
    fps: float = 1.0,
    max_frames: int | None = None,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> FrameList:
    """
    Extract frames from a video file using ffmpeg.

    Args:
        video_path:  Path to the video file (.mp4, .avi, .mov, etc.)
        fps:         Frames per second to sample.
                     0.5 = 1 frame every 2 seconds (long videos)
                     1.0 = 1 frame per second (default, coarse pass)
                     4.0 = 4 frames per second (fine-grained pass)
        max_frames:  If set, stop after this many frames.
        start_sec:   Start timestamp in seconds (for the fine-grained
                     re-sampling pass — don't re-process the whole video).
        end_sec:     End timestamp in seconds (inclusive).

    Returns:
        List of (timestamp_sec, PIL.Image) tuples, ordered by time.
        Images are RGB, 384×384 (resized to match SigLIP 2 So400m input).

    Raises:
        FileNotFoundError: if video_path does not exist.
        RuntimeError:      if ffmpeg fails or video is unreadable.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # ---- Get video metadata ----------------------------------------
    meta       = get_video_metadata(video_path)
    duration   = meta["duration"]
    logger.info(
        f"Video: {video_path.name} | "
        f"duration={duration:.1f}s | "
        f"native_fps={meta['fps']:.2f} | "
        f"sampling_fps={fps}"
    )

    # ---- Clamp requested range to video duration -------------------
    start_sec = max(0.0, start_sec)
    if end_sec is None:
        end_sec = duration
    end_sec = min(end_sec, duration)

    if start_sec >= end_sec:
        logger.warning(f"start_sec ({start_sec}) >= end_sec ({end_sec}), returning empty list")
        return []

    # ---- Build timestamp list --------------------------------------
    # We compute exact timestamps ourselves rather than letting ffmpeg
    # decide — this guarantees we know the timestamp of every frame.
    interval   = 1.0 / fps
    timestamps = []
    t = start_sec
    while t < end_sec:
        timestamps.append(round(t, 4))
        t += interval
    if max_frames is not None:
        timestamps = timestamps[:max_frames]

    logger.info(f"Extracting {len(timestamps)} frames at {fps}fps from {start_sec:.1f}s–{end_sec:.1f}s")

    # ---- Extract each frame ----------------------------------------
    frames: FrameList = []
    for ts in timestamps:
        img = _extract_single_frame(video_path, ts)
        if img is not None:
            frames.append((ts, img))

    logger.info(f"Extracted {len(frames)}/{len(timestamps)} frames successfully")
    return frames


# ------------------------------------------------------------------ #
#  Video metadata                                                      #
# ------------------------------------------------------------------ #
def get_video_metadata(video_path: str | Path) -> dict:
    """
    Return basic metadata for a video file.

    Returns dict with keys:
        duration (float):  total length in seconds
        fps (float):       native frame rate
        width (int):       frame width in pixels
        height (int):      frame height in pixels
        codec (str):       video codec name
    """
    video_path = Path(video_path)
    try:
        probe = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as e:
        raise RuntimeError(
            f"ffmpeg could not read '{video_path.name}'. "
            f"Is it a valid video file?\n{e.stderr.decode()}"
        )

    # Find the first video stream
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"),
        None,
    )
    if video_stream is None:
        raise RuntimeError(f"No video stream found in '{video_path.name}'")

    # Parse fps (stored as "30/1" or "25" string in ffmpeg)
    fps_raw = video_stream.get("avg_frame_rate", "0/1")
    try:
        if "/" in fps_raw:
            num, den = fps_raw.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        else:
            fps = float(fps_raw)
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    # Duration: prefer stream-level, fall back to format-level
    duration = float(
        video_stream.get("duration")
        or probe.get("format", {}).get("duration", 0)
    )

    return {
        "duration": duration,
        "fps":      fps,
        "width":    int(video_stream.get("width",  0)),
        "height":   int(video_stream.get("height", 0)),
        "codec":    video_stream.get("codec_name", "unknown"),
    }


# ------------------------------------------------------------------ #
#  Single-frame extraction                                             #
# ------------------------------------------------------------------ #
def _extract_single_frame(video_path: Path, timestamp: float) -> Image.Image | None:
    """
    Seek to `timestamp` seconds in the video and return a single frame
    as a PIL.Image (RGB, 384×384).

    Returns None if the frame could not be extracted (e.g. seek past end).

    How it works:
      ffmpeg -ss <timestamp> -i <video> -frames:v 1 -f image2pipe -vcodec png pipe:1
      → raw PNG bytes streamed to stdout → PIL.Image.open()

    -ss before -i uses fast keyframe seek (not frame-accurate, but fast).
    For 1fps sampling this is accurate enough; small errors (<0.5s) don't
    matter at this granularity.
    """
    try:
        out, _ = (
            ffmpeg
            .input(str(video_path), ss=timestamp)
            .filter("scale", 384, 384)
            .output("pipe:", vframes=1, format="image2pipe", vcodec="png")
            .run(
                cmd=FFMPEG_EXE,                          # use bundled binary
                capture_stdout=True,
                capture_stderr=True,
                quiet=True,
            )
        )

        if not out:
            return None

        img = Image.open(io.BytesIO(out)).convert("RGB")
        return img

    except ffmpeg.Error:
        # Frame beyond video end, or corrupt seek — skip silently
        return None
    except Exception as e:
        logger.debug(f"Skipping frame at t={timestamp:.2f}s: {e}")
        return None


# ------------------------------------------------------------------ #
#  Convenience: extract region for fine-grained pass                   #
# ------------------------------------------------------------------ #
def extract_region(
    video_path: str | Path,
    center_sec: float,
    padding_sec: float = 5.0,
    fps: float = 4.0,
    video_duration: float | None = None,
) -> FrameList:
    """
    Extract a region around `center_sec` at high fps for the fine-grained pass.

    This is used in Pass 2 of the coarse-to-fine inference:
      1. Pass 1 (1fps): finds approximate region, e.g. center=315s
      2. Pass 2 (4fps): re-samples 310s–320s at 4fps → 40 frames
                        → much more precise start/end boundary

    Args:
        video_path:       Path to the video file
        center_sec:       Center of the region to re-sample
        padding_sec:      Seconds to extend before and after center (default ±5s)
        fps:              Sample rate for the fine pass (default 4fps)
        video_duration:   If known, avoids an extra ffprobe call to clamp end_sec

    Returns:
        FrameList — same format as extract_frames()
    """
    start = max(0.0, center_sec - padding_sec)
    end   = center_sec + padding_sec
    if video_duration is not None:
        end = min(end, video_duration)

    logger.info(
        f"Fine-grained re-sample: center={center_sec:.1f}s "
        f"region=[{start:.1f}s, {end:.1f}s] at {fps}fps"
    )
    return extract_frames(video_path, fps=fps, start_sec=start, end_sec=end)
