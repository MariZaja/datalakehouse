import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_OPENFACE_META_COLS = ["frame", "timestamp", "confidence", "success"]
_OPENFACE_HEAD_POSE_COLS = [
    "pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz",
]
_OPENFACE_GAZE_COLS = [
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
]
_OPENFACE_AU_INTENSITY_COLS = [
    f"AU{n:02d}_r" for n in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
]
_OPENFACE_AU_PRESENCE_COLS = [
    f"AU{n:02d}_c" for n in [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 28, 45]
]


def init_openface() -> bool:
    """Return True if the OpenFace FeatureExtraction CLI is available."""
    path = os.getenv("OPENFACE_PATH") or shutil.which("FeatureExtraction")
    if path:
        logger.info("OpenFace FeatureExtraction CLI found: %s", path)
        return True
    logger.warning("OpenFace CLI (FeatureExtraction) not found — video extraction will be skipped")
    return False


def _aggregate_openface_windows(df: pd.DataFrame, window_size_s: float) -> pd.DataFrame:
    """Aggregate per-frame OpenFace features into fixed-length time windows (mean).

    Windows with no successful frames are kept as NaN rows so the output always
    covers the full timestamp range without gaps.
    """
    if df.empty or "timestamp" not in df.columns:
        return df
    df = df.copy()
    df["window_id"] = (df["timestamp"] / window_size_s).astype(int) + 1
    all_window_ids = range(int(df["window_id"].min()), int(df["window_id"].max()) + 1)
    skip = {"frame", "timestamp", "confidence", "success", "window_id"}
    feat_cols = [c for c in df.columns if c not in skip]
    if "success" in df.columns:
        df = df[df["success"] == 1]
    agg = df.groupby("window_id")[feat_cols].mean().reindex(all_window_ids).reset_index()
    agg.insert(1, "window_start_s", ((agg["window_id"] - 1) * window_size_s).round(3))
    agg.insert(2, "window_end_s", (agg["window_id"] * window_size_s).round(3))
    return agg.reset_index(drop=True)


def extract_video_openface(
    video_bytes: bytes,
    file_id: str,
    max_duration_s: Optional[float] = None,
    window_size_s: float = 0.3,
) -> Optional[pd.DataFrame]:
    """Run OpenFace FeatureExtraction CLI; return head pose, eye gaze, and AU aggregated
    into fixed-length windows (window_size_s).

    If max_duration_s is given, only frames within the debate window are included.
    """
    tmp_video = None
    tmp_out_dir = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_video = tmp.name

        tmp_out_dir = tempfile.mkdtemp()
        openface_path = os.getenv("OPENFACE_PATH")
        openface_cwd = str(Path(openface_path).parent) if openface_path else None

        # Trim to debate window before OpenFace to avoid processing the full file
        if max_duration_s is not None:
            tmp_trimmed = tmp_video + "_trimmed.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_video, "-t", str(max_duration_s),
                 "-c", "copy", tmp_trimmed],
                capture_output=True, check=True,
            )
            os.unlink(tmp_video)
            tmp_video = tmp_trimmed

        result = subprocess.run(
            [
                openface_path,
                "-f", tmp_video,
                "-out_dir", tmp_out_dir,
                "-aus", "-pose", "-gaze",
                "-quiet",
            ],
            capture_output=True,
            timeout=7200,
            cwd=openface_cwd,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace").strip()
            stdout = result.stdout.decode(errors="replace").strip()
            logger.error(
                "OpenFace failed for %s (rc=%d):\n  stderr: %s\n  stdout: %s",
                file_id, result.returncode, stderr[:500] or "<empty>", stdout[:500] or "<empty>",
            )
            return None

        csv_files = list(Path(tmp_out_dir).glob("*.csv"))
        if not csv_files:
            logger.error("OpenFace produced no output CSV for %s", file_id)
            return None

        df = pd.read_csv(csv_files[0])
        df.columns = [c.strip() for c in df.columns]

        keep = (
            _OPENFACE_META_COLS
            + _OPENFACE_HEAD_POSE_COLS
            + _OPENFACE_GAZE_COLS
            + _OPENFACE_AU_INTENSITY_COLS
            + _OPENFACE_AU_PRESENCE_COLS
        )
        df = df[[c for c in keep if c in df.columns]].copy()
        if max_duration_s is not None and "timestamp" in df.columns:
            df = df[df["timestamp"] <= max_duration_s].copy()
        if df.empty:
            return None
        df = _aggregate_openface_windows(df, window_size_s)
        if df.empty:
            return None
        df.insert(0, "file_id", file_id)
        return df
    except FileNotFoundError:
        logger.error("OpenFace CLI (FeatureExtraction) not found — skipping video extraction")
        return None
    except Exception as e:
        logger.error("OpenFace extraction failed for %s: %s", file_id, e)
        return None
    finally:
        if tmp_video and os.path.exists(tmp_video):
            os.unlink(tmp_video)
        if tmp_out_dir:
            shutil.rmtree(tmp_out_dir, ignore_errors=True)
