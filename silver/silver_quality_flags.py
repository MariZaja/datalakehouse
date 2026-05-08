import argparse
import io
import logging
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

import config as project_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("silver_quality_flags")


# ── MinIO helpers ──────────────────────────────────────────────────────────────

def download_object(minio_client, bucket: str, key: str) -> Optional[bytes]:
    try:
        response = minio_client.get_object(bucket, key)
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except Exception as e:
        logger.error("Failed to download %s/%s: %s", bucket, key, e)
        return None


def _group_objects_by_entity(minio_client, bucket: str, prefix: str) -> Dict[str, List[Any]]:
    entity_objects: Dict[str, List[Any]] = {}
    full_prefix = prefix.rstrip("/") + "/"
    for obj in minio_client.list_objects(bucket, prefix=full_prefix, recursive=True):
        for seg in obj.object_name.split("/"):
            if seg.startswith("entity="):
                eid = seg[len("entity="):]
                entity_objects.setdefault(eid, []).append(obj)
                break
    return entity_objects


def upload_csv(minio_client, bucket: str, key: str, df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    minio_client.put_object(
        bucket, key,
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="text/csv",
    )


# ── Missingness report ─────────────────────────────────────────────────────────

def load_missingness_report(
    minio_client, bucket: str, key: str
) -> Tuple[Dict[Tuple[str, str, str], float], Set[Tuple[str, str, str]]]:
    """Return (sample_rate_lookup, total_missing_set).

    sample_rate_lookup: {(participant_id, unit_id, signal_type): sample_rate_hz}
    total_missing_set:  keys where status == 'total_missing' — skip processing entirely
    """
    data = download_object(minio_client, bucket, key)
    if data is None:
        logger.warning("Missingness report not found at %s/%s — sample rates will fall back to config", bucket, key)
        return {}, set()
    df = pd.read_csv(io.BytesIO(data))
    sr_lookup: Dict[Tuple[str, str, str], float] = {}
    miss_skip: Set[Tuple[str, str, str]] = set()
    for _, row in df.iterrows():
        key_tuple = (str(row["participant_id"]), str(row["unit_id"]), str(row["signal_type"]))
        if str(row.get("status", "")) == "total_missing":
            miss_skip.add(key_tuple)
        sr = row.get("sample_rate_hz")
        if pd.notna(sr) and float(sr) > 0:
            sr_lookup[key_tuple] = float(sr)
    return sr_lookup, miss_skip


# ── K-EmoCon subjects ──────────────────────────────────────────────────────────

def load_kemocon_subjects(minio_client, bucket: str, path: str) -> Dict[int, Dict[str, int]]:
    data = download_object(minio_client, bucket, path)
    if data is None:
        logger.error("Cannot load subjects.csv from %s/%s", bucket, path)
        return {}
    df = pd.read_csv(io.BytesIO(data))
    result = {}
    for _, row in df.iterrows():
        pid = int(row["pid"])
        result[pid] = {"startTime": int(row["startTime"]), "endTime": int(row["endTime"])}
    return result


def _pid_from_entity_id(entity_id: str) -> int:
    return int(entity_id[1:])


# ── Signal readers ─────────────────────────────────────────────────────────────

def read_csv_signal(
    data: bytes,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    ref_start_s: Optional[float] = None,
    value_cols: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    cols = value_cols or ["value"]
    try:
        df = pd.read_csv(io.BytesIO(data))
        if timestamp_col not in df.columns:
            logger.warning("CSV missing timestamp column '%s'", timestamp_col)
            return None

        ts_raw = pd.to_numeric(df[timestamp_col], errors="coerce")
        if timestamp_unit_ms:
            ts_raw = ts_raw / 1000.0

        offset = ref_start_s if ref_start_s is not None else float(ts_raw.dropna().iloc[0]) if ts_raw.notna().any() else 0.0
        rel_ts = (ts_raw - offset).values

        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            logger.warning("CSV missing columns %s — cannot compute quality. Available: %s", missing_cols, list(df.columns))
            return None
        vals = df[cols].apply(pd.to_numeric, errors="coerce").values

        return {
            "rel_ts": rel_ts.astype(float),
            "values": vals.astype(float),
            "valid_mask": ~np.isnan(rel_ts),
        }
    except Exception as e:
        logger.warning("CSV signal read failed: %s", e)
        return None


def read_wav_signal(data: bytes) -> Optional[Dict[str, Any]]:
    try:
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(data)) as f:
            samples = f.read(dtype="float32")
            sr = float(f.samplerate)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return {"samples": samples.astype(np.float64), "sample_rate_hz": sr}
    except Exception as e:
        logger.warning("WAV signal read failed: %s", e)
        return None



def read_video_signal(data: bytes) -> Optional[Dict[str, Any]]:
    """Read video frame-by-frame; compute per-frame metrics for all quality analyses."""
    tmp_path = None
    try:
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

        lap_vars: List[float] = []
        clipped_ratios: List[float] = []
        noise_sigmas: List[float] = []
        frame_diffs: List[float] = []   # frame_diffs[i] = diff(frame[i+1], frame[i])

        prev_gray: Optional[np.ndarray] = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            total_px = gray.size

            lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            clipped_ratios.append(float(np.sum((gray < 5) | (gray > 250)) / total_px))

            blurred = cv2.medianBlur(gray, 3)
            noise_sigmas.append(float(np.std(gray.astype(np.float32) - blurred.astype(np.float32))))

            if prev_gray is not None:
                frame_diffs.append(
                    float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0)
                )
            prev_gray = gray

        cap.release()
        if not lap_vars:
            return None

        return {
            "lap_vars": np.array(lap_vars),
            "clipped_ratios": np.array(clipped_ratios),
            "noise_sigmas": np.array(noise_sigmas),
            "frame_diffs": np.array(frame_diffs),
            "sample_rate_hz": fps,
            "n_samples": len(lap_vars),
        }
    except Exception as e:
        logger.warning("Video signal read failed: %s", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_mat_eeg(data: bytes) -> Optional[np.ndarray]:
    """Load EEG .mat; always return array in MATLAB axis convention (tp, ch, instances)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        if data[:8] == b"\x89HDF\r\n\x1a\n":
            import h5py
            with h5py.File(tmp_path, "r") as hf:
                for key in hf.keys():
                    if not key.startswith("#"):
                        return np.array(hf[key]).T
            return None

        import scipy.io
        try:
            mat = scipy.io.loadmat(tmp_path)
            data_vars = [(n, v) for n, v in mat.items() if not n.startswith("_")]
            return data_vars[0][1] if data_vars else None
        except Exception as scipy_err:
            # MATLAB v7.3 embeds HDF5 after a text header — scipy can't read it, h5py can
            try:
                import h5py
                with h5py.File(tmp_path, "r") as hf:
                    for key in hf.keys():
                        if not key.startswith("#"):
                            return np.array(hf[key]).T
                return None
            except Exception:
                raise scipy_err  # re-raise original: file is likely corrupt/truncated
    except Exception as e:
        logger.warning("MAT EEG load failed: %s", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Output filename ────────────────────────────────────────────────────────────

_AUDIO_VIDEO_SIGNALS = {"audio", "video"}

_SIGNAL_TO_FILESTEM: Dict[str, str] = {
    "E4_BVP":    "biosignal_bvp",
    "E4_EDA":    "biosignal_eda",
    "E4_ACC":    "biosignal_acc",
    "E4_HR":     "biosignal_hr",
    "E4_IBI":    "biosignal_ibi",
    "E4_TEMP":   "biosignal_temp",
    "BrainWave": "biosignal_brainwave",
    "Attention": "biosignal_attention",
    "Meditation":"biosignal_meditation",
    "Polar_HR":  "biosignal_polar_hr",
    "eeg":       "biosignal_eeg",
    "audio":     "audio",
    "video":     "video",
}


def _output_filename(entity_id: str, signal_type: str) -> str:
    stem = _SIGNAL_TO_FILESTEM.get(signal_type, f"biosignal_{signal_type.lower()}")
    return f"{entity_id}_{stem}_quality_flags.csv"


# ── ACC helpers ───────────────────────────────────────────────────────────────

def _acc_quality_problem(
    completeness: float,
    out_ratio: float,
    clip_ratio: float,
    is_flat: bool,
    M_mean: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Completeness
    if completeness >= 0.95:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.80:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 2. Physical range — any out-of-range sample corrupts the window
    if out_ratio == 0.0:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 3. Clipping — max ratio across axes
    if clip_ratio <= 0.02:
        results.append(("GOOD", "NONE"))
    elif clip_ratio <= 0.10:
        results.append(("NOISY", "CLIPPING"))
    else:
        results.append(("BAD", "CLIPPING"))

    # 4. Flat line / sensor disconnection
    if is_flat:
        results.append(("BAD", "FLAT"))

    # 5. Non-physical magnitude
    if 10.0 <= M_mean <= 180.0:
        results.append(("GOOD", "NONE"))
    elif M_mean < 10.0:
        results.append(("BAD", "FLAT"))
    else:
        results.append(("NOISY", "ARTIFACT"))

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


# ── BVP helpers ───────────────────────────────────────────────────────────────

def _count_bvp_peaks(x: np.ndarray, sr: float) -> int:
    if len(x) < 10:
        return 0
    try:
        from scipy import signal as sp_signal
        nyq = sr / 2.0
        b, a = sp_signal.butter(2, [0.5 / nyq, min(3.0 / nyq, 0.99)], btype="bandpass")
        filtered = sp_signal.filtfilt(b, a, x)
        peaks, _ = sp_signal.find_peaks(filtered, distance=max(1, int(sr * 0.2)))
        return len(peaks)
    except Exception:
        return 0


def _bvp_quality_problem(
    completeness: float,
    amplitude: float,
    std_val: float,
    clip_ratio: float,
    peaks_count: int,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Completeness
    if completeness >= 0.95:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.80:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 2+3. Amplitude (std confirms FLAT — protects against outlier-driven false FLAT)
    if amplitude < 5:
        results.append(("BAD", "FLAT" if std_val < 2 else "ARTIFACT"))
    elif amplitude <= 400:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 4. Clipping
    if clip_ratio <= 0.02:
        results.append(("GOOD", "NONE"))
    elif clip_ratio <= 0.10:
        results.append(("NOISY", "CLIPPING"))
    else:
        results.append(("BAD", "CLIPPING"))

    # 5. Peaks — lower priority: can raise to NOISY but never to BAD alone
    if 1 <= peaks_count <= 3:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("NOISY", "ARTIFACT"))

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


# ── Per-signal processing functions — K-EmoCon ────────────────────────────────
#
# Each function receives raw bytes of the signal file (or None if missing),
# entity context, and config; returns a per-window quality DataFrame or None.
# BVP columns: window_id, window_start_s, window_end_s,
#              completeness, amplitude, std, clip_ratio, peaks_count,
#              quality_flag, problem_flag
#
# ──────────────────────────────────────────────────────────────────────────────

def process_e4_bvp(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    sr = sample_rate_hz or 64.0
    expected_per_window = int(round(sr * window_size_s))
    total_windows = math.ceil(debate_duration_s / window_size_s)

    def _missing_row(wid: int, completeness: float = 0.0) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "completeness": completeness,
            "amplitude": float("nan"),
            "std": float("nan"),
            "clip_ratio": float("nan"),
            "peaks_count": -1,
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    sig = read_csv_signal(file_data, timestamp_col, timestamp_unit_ms, ref_start_s=debate_start_s)
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"].flatten()
    valid_mask = sig["valid_mask"]

    # p99.9 computed once from the entire file before windowing
    finite_vals = values[valid_mask & np.isfinite(values)]
    p999 = float(np.percentile(finite_vals, 99.9)) if len(finite_vals) > 0 else np.inf

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_actual = int(in_window.sum())
        x = values[in_window]
        x = x[np.isfinite(x)]

        completeness = round(min(1.0, n_actual / expected_per_window), 4) if expected_per_window > 0 else 0.0

        if len(x) == 0:
            rows.append(_missing_row(wid, completeness))
            continue

        amplitude = round(float(np.max(x) - np.min(x)), 4)
        std_val = round(float(np.std(x)), 4)
        clip_ratio = round(float(np.sum(x >= p999) / n_actual), 4)
        peaks_count = _count_bvp_peaks(x, sr)

        quality_flag, problem_flag = _bvp_quality_problem(
            completeness, amplitude, std_val, clip_ratio, peaks_count
        )

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "amplitude": amplitude,
            "std": std_val,
            "clip_ratio": clip_ratio,
            "peaks_count": peaks_count,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


# ── EDA helpers ───────────────────────────────────────────────────────────────

def _eda_quality_problem(
    completeness: float,
    out_ratio: float,
    flat_line_std: float,
    delta: float,
    drift: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Completeness (4 Hz → 4 samples/window)
    if completeness >= 1.0:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.75:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 2. Physical range [0, 20] µS
    if out_ratio == 0.0:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 3. Flat line — rolling 30s std
    if not np.isnan(flat_line_std):
        if flat_line_std >= 0.005:
            results.append(("GOOD", "NONE"))
        elif flat_line_std >= 0.001:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

    # 4. Transient jump within window
    if not np.isnan(delta):
        if delta <= 0.5:
            results.append(("GOOD", "NONE"))
        elif delta <= 2.0:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    # 5. Drift vs 60s baseline
    if not np.isnan(drift):
        if drift <= 2.0:
            results.append(("GOOD", "NONE"))
        elif drift <= 5.0:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    if not results:
        return "BAD", "ARTIFACT"

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


def process_e4_eda(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    sr = sample_rate_hz or 4.0
    expected_per_window = int(round(sr * window_size_s))  # 4 at 4 Hz
    total_windows = math.ceil(debate_duration_s / window_size_s)

    def _missing_row(wid: int, completeness: float = 0.0) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "completeness": completeness,
            "out_ratio": float("nan"),
            "flat_line_30": float("nan"),
            "delta": float("nan"),
            "drift": float("nan"),
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    # Load with ref_start_s so pre-debate samples have negative rel_ts (warmup)
    sig = read_csv_signal(file_data, timestamp_col, timestamp_unit_ms, ref_start_s=debate_start_s)
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"].flatten()
    valid_mask = sig["valid_mask"] & np.isfinite(values)

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s

        # Current window
        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_actual = int(in_window.sum())
        x = values[in_window]

        completeness = round(min(1.0, n_actual / expected_per_window), 4) if expected_per_window > 0 else 0.0

        if len(x) == 0:
            rows.append(_missing_row(wid, completeness))
            continue

        # 2. Physical range [0, 20] µS
        out_ratio = round(float(np.sum((x < 0) | (x > 20)) / len(x)), 4)

        # 3. Flat line — rolling 30s std; pre-debate data used as warmup
        in_roll = valid_mask & (rel_ts >= w_end - 30.0) & (rel_ts < w_end)
        x_roll = values[in_roll]
        flat_line_30 = round(float(np.std(x_roll)), 6) if len(x_roll) >= 2 else float("nan")

        # 4. Transient jump within window
        delta = round(float(np.max(x) - np.min(x)), 6)

        # 5. Drift vs 60s baseline before window; pre-debate data used as warmup
        in_baseline = valid_mask & (rel_ts >= w_start - 60.0) & (rel_ts < w_start)
        x_baseline = values[in_baseline]
        drift = round(float(abs(np.mean(x) - np.mean(x_baseline))), 6) if len(x_baseline) >= 1 else float("nan")

        quality_flag, problem_flag = _eda_quality_problem(
            completeness, out_ratio, flat_line_30, delta, drift
        )

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "out_ratio": out_ratio,
            "flat_line_30": flat_line_30,
            "delta": delta,
            "drift": drift,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


def process_e4_acc(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    sr = sample_rate_hz or 32.0
    expected_per_window = int(round(sr * window_size_s))
    total_windows = math.ceil(debate_duration_s / window_size_s)

    def _missing_row(wid: int, completeness: float = 0.0) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "completeness": completeness,
            "out_ratio": float("nan"),
            "clip_ratio": float("nan"),
            "FLAT": True,
            "M": float("nan"),
            "M_mean": float("nan"),
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    sig = read_csv_signal(
        file_data, timestamp_col, timestamp_unit_ms,
        ref_start_s=debate_start_s, value_cols=["x", "y", "z"],
    )
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"]       # shape (n, 3): columns x, y, z
    valid_mask = sig["valid_mask"]

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_actual = int(in_window.sum())
        xyz = values[in_window]      # shape (n_actual, 3)

        # Drop rows where any axis is NaN
        finite_mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite_mask]
        n_finite = len(xyz)

        completeness = round(min(1.0, n_actual / expected_per_window), 4) if expected_per_window > 0 else 0.0

        if n_finite == 0:
            rows.append(_missing_row(wid, completeness))
            continue

        # 2. Physical range
        out_of_range = ((xyz < -128) | (xyz > 128)).any(axis=1)
        out_ratio = round(float(out_of_range.sum() / n_finite), 4)

        # 3. Clipping — max ratio across axes
        per_axis_clip = (np.abs(xyz) >= 126).sum(axis=0) / n_finite
        clip_ratio = round(float(per_axis_clip.max()), 4)

        # 4. Flat line / sensor disconnection
        sigma = xyz.std(axis=0)          # shape (3,): std per axis
        magnitude = np.sqrt((xyz ** 2).sum(axis=1))
        M = float(magnitude.mean())
        is_flat = bool(
            (sigma[0] < 0.5) and (sigma[1] < 0.5) and (sigma[2] < 0.5)
            and abs(M - 64) > 10
        )

        # 5. Non-physical magnitude (same formula as analysis 4)
        M_mean = round(M, 4)
        M = round(M, 4)

        quality_flag, problem_flag = _acc_quality_problem(
            completeness, out_ratio, clip_ratio, is_flat, M_mean
        )

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "out_ratio": out_ratio,
            "clip_ratio": clip_ratio,
            "FLAT": is_flat,
            "M": M,
            "M_mean": M_mean,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


def _hr_quality_problem(
    n_window: int,
    x: float,
    roc: float,
    flat_line_30: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Presence — 0 samples is immediately BAD/ARTIFACT, no further checks
    if n_window != 1:
        return "BAD", "ARTIFACT"
    results.append(("GOOD", "NONE"))

    # 2. Physical range
    if 40 <= x <= 180:
        results.append(("GOOD", "NONE"))
    elif (30 <= x < 40) or (180 < x <= 220):
        results.append(("NOISY", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 3. Rate of change — caller passes NaN when previous window was BAD
    if not np.isnan(roc):
        if roc <= 10:
            results.append(("GOOD", "NONE"))
        elif roc <= 20:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    # 4. Extended flat (rolling 30s std)
    if not np.isnan(flat_line_30):
        if flat_line_30 >= 0.5:
            results.append(("GOOD", "NONE"))
        elif flat_line_30 >= 0.1:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


def process_e4_hr(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
    bvp_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    total_windows = math.ceil(debate_duration_s / window_size_s)

    bvp_flags: Dict[int, str] = (
        dict(zip(bvp_df["window_id"], bvp_df["quality_flag"]))
        if bvp_df is not None and "quality_flag" in bvp_df.columns
        else {}
    )

    def _missing_row(wid: int) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "samples": 0,
            "in_range": False,
            "RoC": float("nan"),
            "flat_line_30": float("nan"),
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
            "bvp_quality_flag": bvp_flags.get(wid),
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    sig = read_csv_signal(file_data, timestamp_col, timestamp_unit_ms, ref_start_s=debate_start_s)
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"].flatten()
    valid_mask = sig["valid_mask"] & np.isfinite(values)

    rows: List[Dict[str, Any]] = []
    prev_quality_flag: Optional[str] = None
    prev_x: Optional[float] = None

    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s

        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_window = int(in_window.sum())

        if n_window == 0:
            rows.append(_missing_row(wid))
            prev_quality_flag = "BAD"
            prev_x = None
            continue

        x = float(values[in_window][0])

        in_range = bool(30 <= x <= 220)

        # RoC: skip when previous window was BAD (don't penalize against an unreliable predecessor)
        if prev_x is not None and prev_quality_flag != "BAD":
            roc = round(abs(x - prev_x), 4)
        else:
            roc = float("nan")

        # flat_line_30: rolling 30s std; pre-debate data used as warmup
        in_roll = valid_mask & (rel_ts >= w_end - 30.0) & (rel_ts < w_end)
        x_roll = values[in_roll]
        flat_line_30 = round(float(np.std(x_roll)), 6) if len(x_roll) >= 2 else float("nan")

        quality_flag, problem_flag = _hr_quality_problem(n_window, x, roc, flat_line_30)

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "samples": n_window,
            "in_range": in_range,
            "RoC": roc,
            "flat_line_30": flat_line_30,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
            "bvp_quality_flag": bvp_flags.get(wid),
        })

        prev_quality_flag = quality_flag
        prev_x = x

    return pd.DataFrame(rows) if rows else None


def _ibi_quality_problem(
    n_window: int,
    gap: float,
    out_ratio: float,
    jump: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Window presence
    if n_window >= 1:
        results.append(("GOOD", "NONE"))

    # 2. Gap detection — N_window >= 1 resets to GOOD regardless of prior gap;
    #    only empty windows are penalised by gap duration
    if n_window >= 1:
        results.append(("GOOD", "NONE"))
    elif not np.isnan(gap):
        if gap <= 2.0:
            results.append(("GOOD", "NONE"))
        elif gap <= 5.0:
            results.append(("NOISY", "ARTIFACT"))
        else:
            results.append(("BAD", "ARTIFACT"))

    # 3. Physical range [0.27, 2.0] s — skipped if N_window = 0
    if n_window >= 1 and not np.isnan(out_ratio):
        if out_ratio == 0.0:
            results.append(("GOOD", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    # 4. Short-term consistency — only when N_window >= 2
    if n_window >= 2 and not np.isnan(jump):
        if jump <= 0.20:
            results.append(("GOOD", "NONE"))
        elif jump <= 0.40:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    if not results:
        return "BAD", "ARTIFACT"

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


def process_e4_ibi(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    total_windows = math.ceil(debate_duration_s / window_size_s)

    def _missing_row(wid: int) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "samples": 0,
            "gap": float("nan"),
            "gap_before_s": 0.0,
            "out_ratio": float("nan"),
            "jump": float("nan"),
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    sig = read_csv_signal(file_data, timestamp_col, timestamp_unit_ms, ref_start_s=debate_start_s)
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"].flatten()
    valid_mask = sig["valid_mask"] & np.isfinite(values)

    valid_ts = rel_ts[valid_mask]
    valid_vals = values[valid_mask]

    # Use any pre-debate IBI as warmup for the first gap
    pre_ts = valid_ts[valid_ts < 0.0]
    t_last_ibi: float = float(pre_ts[-1]) if len(pre_ts) > 0 else float("nan")
    was_in_gap: bool = False

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s

        in_window = (valid_ts >= w_start) & (valid_ts < w_end)
        n_window = int(in_window.sum())
        x = valid_vals[in_window]
        ts_in = valid_ts[in_window]

        gap = round(float(w_start - t_last_ibi), 4) if not np.isnan(t_last_ibi) else float("nan")

        # gap_before_s: records G only at the moment the signal resumes after a gap
        gap_before_s = round(gap, 4) if (n_window >= 1 and was_in_gap and not np.isnan(gap)) else 0.0

        out_ratio = (
            round(float(np.sum((x < 270) | (x > 2000)) / n_window), 4)
            if n_window >= 1 else float("nan")
        )

        if n_window >= 2:
            x1, x2 = x[0], x[1]
            mean_x = (x1 + x2) / 2.0
            jump = round(float(abs(x1 - x2) / mean_x), 4) if mean_x > 0 else float("nan")
        else:
            jump = float("nan")

        quality_flag, problem_flag = _ibi_quality_problem(n_window, gap, out_ratio, jump)

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "samples": n_window,
            "gap": gap,
            "gap_before_s": gap_before_s,
            "out_ratio": out_ratio,
            "jump": jump,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

        if n_window > 0:
            t_last_ibi = float(ts_in[-1])
            was_in_gap = False
        elif not np.isnan(t_last_ibi):
            was_in_gap = True

    return pd.DataFrame(rows) if rows else None


def _temp_quality_problem(
    completeness: float,
    out_ratio: float,
    mean_temp: float,
    flat_line_60: float,
    roc: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Completeness (4 Hz → 4 samples/window)
    if completeness >= 1.0:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.75:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 2. Physical range [20, 42] °C
    if out_ratio == 0.0:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 3. Suspicious range (sensor off skin): 20–25°C
    if not np.isnan(mean_temp):
        if 25.0 <= mean_temp <= 40.0:
            results.append(("GOOD", "NONE"))
        elif 20.0 <= mean_temp < 25.0:
            results.append(("NOISY", "NONE"))
        # outside [20, 42] is already handled by analysis 2

    # 4. Flat line — rolling 60s std
    if not np.isnan(flat_line_60):
        if flat_line_60 >= 0.01:
            results.append(("GOOD", "NONE"))
        elif flat_line_60 >= 0.005:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

    # 5. Rate of change within 1s window
    if not np.isnan(roc):
        if roc <= 0.1:
            results.append(("GOOD", "NONE"))
        elif roc <= 0.5:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    if not results:
        return "BAD", "ARTIFACT"

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


def process_e4_temp(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    sr = sample_rate_hz or 4.0
    expected_per_window = int(round(sr * window_size_s))  # 4 at 4 Hz
    total_windows = math.ceil(debate_duration_s / window_size_s)

    def _missing_row(wid: int, completeness: float = 0.0) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "completeness": completeness,
            "out_ratio": float("nan"),
            "mean_temp": float("nan"),
            "flat_line_60": float("nan"),
            "RoC": float("nan"),
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    # Load with ref_start_s so pre-debate samples have negative rel_ts (warmup for 60s flat-line)
    sig = read_csv_signal(file_data, timestamp_col, timestamp_unit_ms, ref_start_s=debate_start_s)
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"].flatten()
    valid_mask = sig["valid_mask"] & np.isfinite(values)

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s

        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_actual = int(in_window.sum())
        x = values[in_window]

        completeness = round(min(1.0, n_actual / expected_per_window), 4) if expected_per_window > 0 else 0.0

        if len(x) == 0:
            rows.append(_missing_row(wid, completeness))
            continue

        # 2. Physical range [20, 42] °C
        out_ratio = round(float(np.sum((x < 20) | (x > 42)) / len(x)), 4)

        # 3. Suspicious range — mean of current window
        mean_temp = round(float(np.mean(x)), 4)

        # 4. Flat line — rolling 60s std; pre-debate data used as warmup
        in_roll = valid_mask & (rel_ts >= w_end - 60.0) & (rel_ts < w_end)
        x_roll = values[in_roll]
        flat_line_60 = round(float(np.std(x_roll)), 6) if len(x_roll) >= 2 else float("nan")

        # 5. Rate of change = |max - min| within current 1s window
        roc = round(float(np.max(x) - np.min(x)), 6)

        quality_flag, problem_flag = _temp_quality_problem(
            completeness, out_ratio, mean_temp, flat_line_60, roc
        )

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "out_ratio": out_ratio,
            "mean_temp": mean_temp,
            "flat_line_60": flat_line_60,
            "RoC": roc,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


_BRAINWAVE_CHANNELS = [
    "delta", "lowAlpha", "highAlpha", "lowBeta",
    "highBeta", "lowGamma", "middleGamma", "theta",
]

_BW_QUALITY_ORDER = {"GOOD": 0, "NOISY": 1, "BAD": 2}
_BW_PROBLEM_ORDER = {"NONE": 0, "ARTIFACT": 1, "CLIPPING": 2, "FLAT": 3}


def _bw_signal_quality_problem(
    zeros: bool, flat: bool, clipping: bool, spike: bool
) -> Tuple[str, str]:
    if not any([zeros, flat, clipping, spike]):
        return "GOOD", "NONE"
    quality = "GOOD"
    problems: set = set()
    if zeros:
        quality = "BAD"
        problems.add("ARTIFACT")
    if flat:
        quality = "BAD"
        problems.add("FLAT")
    if clipping:
        quality = "BAD"
        problems.add("CLIPPING")
    if spike:
        if quality == "GOOD":
            quality = "NOISY"
        problems.add("ARTIFACT")
    problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problems)
    return quality, problem_flag


def process_brainwave(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    channels = _BRAINWAVE_CHANNELS
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    total_windows = math.ceil(debate_duration_s / window_size_s)

    def _bad_row(wid: int) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
        }
        for ch in channels:
            row[f"{ch}_zeros"] = False
            row[f"{ch}_flat"] = False
            row[f"{ch}_clipping"] = False
            row[f"{ch}_spike"] = False
        row["quality_flag"] = "BAD"
        row["problem_flag"] = "ARTIFACT"
        row["flag_reason"] = ""
        return row

    if file_data is None:
        return pd.DataFrame([_bad_row(w) for w in range(total_windows)])

    sig = read_csv_signal(
        file_data, timestamp_col, timestamp_unit_ms,
        ref_start_s=debate_start_s, value_cols=channels,
    )
    if sig is None:
        return pd.DataFrame([_bad_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"]          # shape (N, 8)
    valid_mask = sig["valid_mask"]

    # ── Per-file thresholds (computed once across all windows) ─────────────────
    clip_thresholds: List[float] = []
    spike_thresholds: List[float] = []
    flat_masks: List[np.ndarray] = []
    spike_masks: List[np.ndarray] = []

    for ch_idx in range(len(channels)):
        col = values[:, ch_idx]
        finite_valid = col[valid_mask & np.isfinite(col)]

        # Analysis 3 — clipping: 99.5th percentile per file
        clip_thr = float(np.percentile(finite_valid, 99.5)) if len(finite_valid) > 0 else np.inf
        clip_thresholds.append(clip_thr)

        # Analysis 4 — spike: Tukey's fencing on absolute first differences
        diffs = np.abs(np.diff(col))
        valid_pair = valid_mask[:-1] & valid_mask[1:] & np.isfinite(diffs)
        finite_diffs = diffs[valid_pair]
        if len(finite_diffs) >= 4:
            q1, q3 = np.percentile(finite_diffs, [25, 75])
            spike_thr = float(q3 + 1.5 * (q3 - q1))
        else:
            spike_thr = np.inf
        spike_thresholds.append(spike_thr)

        # Analysis 2 — flat: three identical consecutive values
        flat_m = np.zeros(len(col), dtype=bool)
        if len(col) >= 3:
            flat_m[2:] = (col[2:] == col[1:-1]) & (col[1:-1] == col[:-2])
        flat_masks.append(flat_m)

        # Spike mask (per-sample, starting from index 1)
        spike_m = np.zeros(len(col), dtype=bool)
        if len(col) >= 2:
            spike_m[1:] = diffs > spike_thr
        spike_masks.append(spike_m)

    # ── Window loop ────────────────────────────────────────────────────────────
    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)

        row: Dict[str, Any] = {
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
        }

        sig_quality: Dict[str, str] = {}
        sig_problem: Dict[str, str] = {}

        for ch_idx, ch in enumerate(channels):
            x = values[:, ch_idx][in_window]

            if len(x) == 0:
                z = f = c = s = False
            else:
                z = bool(np.any(x == 0))                                  # Analysis 1
                f = bool(np.any(flat_masks[ch_idx][in_window]))            # Analysis 2
                c = bool(np.any(x >= clip_thresholds[ch_idx]))             # Analysis 3
                s = bool(np.any(spike_masks[ch_idx][in_window]))           # Analysis 4

            row[f"{ch}_zeros"] = z
            row[f"{ch}_flat"] = f
            row[f"{ch}_clipping"] = c
            row[f"{ch}_spike"] = s

            sig_quality[ch], sig_problem[ch] = _bw_signal_quality_problem(z, f, c, s)

        # Worst flags across all channels
        worst_q = max(sig_quality.values(), key=lambda q: _BW_QUALITY_ORDER[q])
        worst_p = max(sig_problem.values(), key=lambda p: _BW_PROBLEM_ORDER[p])

        # flag_reason: channels that carry the row-level quality flag (excluding GOOD)
        flagged = [ch for ch in channels if sig_quality[ch] == worst_q and worst_q != "GOOD"]
        row["quality_flag"] = worst_q
        row["problem_flag"] = worst_p
        row["flag_reason"] = ", ".join(flagged)
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


def process_attention(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    total_windows = math.ceil(debate_duration_s / window_size_s)

    def _missing_row(wid: int) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "samples": 0,
            "zeros": False,
            "low_value": False,
            "delta": float("nan"),
            "run_100": 0,
            "run": 0,
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    sig = read_csv_signal(file_data, timestamp_col, timestamp_unit_ms, ref_start_s=debate_start_s)
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"].flatten()
    valid_mask = sig["valid_mask"] & np.isfinite(values)

    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}

    rows: List[Dict[str, Any]] = []
    prev_x: Optional[float] = None  # persists across empty windows
    cur_run_100: int = 0
    cur_run_val: Optional[float] = None
    cur_run_len: int = 0

    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s

        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_actual = int(in_window.sum())
        xs = values[in_window]

        if n_actual == 0:
            rows.append({
                "window_id": wid,
                "window_start_s": round(w_start, 3),
                "window_end_s": round(w_end, 3),
                "samples": 0,
                "zeros": False,
                "low_value": False,
                "delta": float("nan"),
                "run_100": 0,
                "run": 0,
                "quality_flag": "BAD",
                "problem_flag": "ARTIFACT",
            })
            # prev_x persists — last known value stays available for delta after a gap
            cur_run_100 = 0
            cur_run_val = None
            cur_run_len = 0
            continue

        # Analysis 2 — zero is the SDK's "unable to compute" sentinel; any sample triggers
        zeros = bool(np.any(xs == 0.0))

        # Analysis 3 — values 1–9 are non-informative eSense noise floor; any sample triggers
        low_value = bool(np.any((xs > 0.0) & (xs < 10.0)))

        # Analysis 4 — spike: max consecutive delta from prev_x through all samples in window
        seq = np.concatenate([[prev_x], xs]) if prev_x is not None else xs
        delta = round(float(np.abs(np.diff(seq)).max()), 4) if len(seq) >= 2 else float("nan")

        # Analysis 5 — clipping: update run_100 per sample, report value at end of window
        for x in xs:
            cur_run_100 = cur_run_100 + 1 if x == 100.0 else 0

        # Analysis 6 — flat line: update run per sample, report max across window
        run_in_window: List[int] = []
        for x in xs:
            if x == cur_run_val:
                cur_run_len += 1
            else:
                cur_run_val = x
                cur_run_len = 1
            run_in_window.append(cur_run_len)
        run_max = max(run_in_window)

        prev_x = float(xs[-1])

        results: List[Tuple[str, str]] = [("GOOD", "NONE")]  # Analysis 1: sample present
        results.append(("BAD", "ARTIFACT") if zeros else ("GOOD", "NONE"))
        results.append(("NOISY", "ARTIFACT") if low_value else ("GOOD", "NONE"))
        if not np.isnan(delta):
            results.append(("BAD", "ARTIFACT") if delta > 40 else ("GOOD", "NONE"))
        if cur_run_100 > 15:
            results.append(("BAD", "CLIPPING"))
        elif cur_run_100 > 10:
            results.append(("NOISY", "CLIPPING"))
        else:
            results.append(("GOOD", "NONE"))
        if run_max >= 60:
            results.append(("BAD", "FLAT"))
        elif run_max >= 30:
            results.append(("NOISY", "FLAT"))
        else:
            results.append(("GOOD", "NONE"))

        quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
        problem_set = {r[1] for r in results}
        problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "samples": n_actual,
            "zeros": zeros,
            "low_value": low_value,
            "delta": delta,
            "run_100": cur_run_100,
            "run": run_max,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


def process_meditation(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    total_windows = math.ceil(debate_duration_s / window_size_s)

    def _missing_row(wid: int) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "samples": 0,
            "zeros": False,
            "low_value": False,
            "delta": float("nan"),
            "run_100": 0,
            "run": 0,
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    sig = read_csv_signal(file_data, timestamp_col, timestamp_unit_ms, ref_start_s=debate_start_s)
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"].flatten()
    valid_mask = sig["valid_mask"] & np.isfinite(values)

    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}

    rows: List[Dict[str, Any]] = []
    prev_x: Optional[float] = None
    cur_run_100: int = 0
    cur_run_val: Optional[float] = None
    cur_run_len: int = 0

    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s

        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_actual = int(in_window.sum())
        xs = values[in_window]

        if n_actual == 0:
            rows.append({
                "window_id": wid,
                "window_start_s": round(w_start, 3),
                "window_end_s": round(w_end, 3),
                "samples": 0,
                "zeros": False,
                "low_value": False,
                "delta": float("nan"),
                "run_100": 0,
                "run": 0,
                "quality_flag": "BAD",
                "problem_flag": "ARTIFACT",
            })
            cur_run_100 = 0
            cur_run_val = None
            cur_run_len = 0
            continue

        # Analysis 2 — 0 is the NeuroSky SDK sentinel for "unable to compute"
        zeros = bool(np.any(xs == 0.0))

        # Analysis 3 — values 1–4 are non-informative eSense noise floor
        low_value = bool(np.any((xs > 0.0) & (xs < 5.0)))

        # Analysis 4 — spike: max consecutive delta from prev_x through all samples in window
        seq = np.concatenate([[prev_x], xs]) if prev_x is not None else xs
        delta = round(float(np.abs(np.diff(seq)).max()), 4) if len(seq) >= 2 else float("nan")

        # Analysis 5 — clipping: update run_100 per sample, report value at end of window
        for x in xs:
            cur_run_100 = cur_run_100 + 1 if x == 100.0 else 0

        # Analysis 6 — flat line: update run per sample, report max across window
        run_in_window: List[int] = []
        for x in xs:
            if x == cur_run_val:
                cur_run_len += 1
            else:
                cur_run_val = x
                cur_run_len = 1
            run_in_window.append(cur_run_len)
        run_max = max(run_in_window)

        prev_x = float(xs[-1])

        results: List[Tuple[str, str]] = [("GOOD", "NONE")]  # Analysis 1: sample present
        results.append(("BAD", "ARTIFACT") if zeros else ("GOOD", "NONE"))
        results.append(("NOISY", "ARTIFACT") if low_value else ("GOOD", "NONE"))
        if not np.isnan(delta):
            results.append(("BAD", "ARTIFACT") if delta > 40 else ("GOOD", "NONE"))
        if cur_run_100 > 120:
            results.append(("BAD", "CLIPPING"))
        elif cur_run_100 > 60:
            results.append(("NOISY", "CLIPPING"))
        else:
            results.append(("GOOD", "NONE"))
        if run_max >= 300:
            results.append(("BAD", "FLAT"))
        elif run_max >= 120:
            results.append(("NOISY", "FLAT"))
        else:
            results.append(("GOOD", "NONE"))

        quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
        problem_set = {r[1] for r in results}
        problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "samples": n_actual,
            "zeros": zeros,
            "low_value": low_value,
            "delta": delta,
            "run_100": cur_run_100,
            "run": run_max,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


def _polar_hr_quality_problem(
    n_window: int,
    x: float,
    roc: float,
    flat_line_30: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Presence — n_window == 0 is handled by the caller; any n_window >= 1 passes
    results.append(("GOOD", "NONE"))

    # 2. Physical range
    if 40 <= x <= 180:
        results.append(("GOOD", "NONE"))
    elif (30 <= x < 40) or (180 < x <= 220):
        results.append(("NOISY", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 3. Rate of change — caller passes NaN when previous window was BAD
    if not np.isnan(roc):
        if roc <= 20:
            results.append(("GOOD", "NONE"))
        elif roc <= 35:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    # 4. Extended flat (rolling 30s std)
    if not np.isnan(flat_line_30):
        if flat_line_30 >= 2.0:
            results.append(("GOOD", "NONE"))
        elif flat_line_30 >= 0.5:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


def process_polar_hr(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
    e4_hr_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 1))
    total_windows = math.ceil(debate_duration_s / window_size_s)

    e4_flags: Dict[int, str] = (
        dict(zip(e4_hr_df["window_id"], e4_hr_df["quality_flag"]))
        if e4_hr_df is not None and "quality_flag" in e4_hr_df.columns
        else {}
    )

    def _missing_row(wid: int) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "samples": 0,
            "in_range": False,
            "RoC": float("nan"),
            "flat_line_30": float("nan"),
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
            "e4_HR_quality_flag": e4_flags.get(wid),
        }

    if file_data is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    sig = read_csv_signal(file_data, timestamp_col, timestamp_unit_ms, ref_start_s=debate_start_s)
    if sig is None:
        return pd.DataFrame([_missing_row(w) for w in range(total_windows)])

    rel_ts = sig["rel_ts"]
    values = sig["values"].flatten()
    valid_mask = sig["valid_mask"] & np.isfinite(values)

    rows: List[Dict[str, Any]] = []
    prev_quality_flag: Optional[str] = None
    prev_x: Optional[float] = None

    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s

        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_window = int(in_window.sum())

        if n_window == 0:
            rows.append(_missing_row(wid))
            prev_quality_flag = "BAD"
            prev_x = None
            continue

        x = float(values[in_window][0])

        in_range = bool(30 <= x <= 220)

        # RoC: skip when previous window was BAD (don't penalize against an unreliable predecessor)
        if prev_x is not None and prev_quality_flag != "BAD":
            roc = round(abs(x - prev_x), 4)
        else:
            roc = float("nan")

        # flat_line_30: rolling 30s std; pre-debate data used as warmup
        in_roll = valid_mask & (rel_ts >= w_end - 30.0) & (rel_ts < w_end)
        x_roll = values[in_roll]
        flat_line_30 = round(float(np.std(x_roll)), 6) if len(x_roll) >= 2 else float("nan")

        quality_flag, problem_flag = _polar_hr_quality_problem(n_window, x, roc, flat_line_30)

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "samples": n_window,
            "in_range": in_range,
            "RoC": roc,
            "flat_line_30": flat_line_30,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
            "e4_HR_quality_flag": e4_flags.get(wid),
        })

        prev_quality_flag = quality_flag
        prev_x = x

    return pd.DataFrame(rows) if rows else None


# ── Audio QC helpers ──────────────────────────────────────────────────────────

def _max_true_run(mask: np.ndarray) -> int:
    """Longest consecutive run of True in a boolean array."""
    if len(mask) == 0 or not mask.any():
        return 0
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return int((ends - starts).max())


def _spectral_flatness_audio(x: np.ndarray) -> float:
    power = np.abs(np.fft.rfft(x)) ** 2
    power = np.maximum(power, 1e-10)
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


def _audio_window_metrics(x: np.ndarray) -> Dict[str, Any]:
    from scipy.stats import kurtosis as scipy_kurtosis
    N = len(x)
    rms = float(np.sqrt(np.mean(x ** 2)))
    rms_db = round(20.0 * np.log10(rms) if rms > 1e-12 else -120.0, 4)
    zero_ratio = round(float(np.sum(np.abs(x) < 1e-6) / N), 6)
    clip_mask = np.abs(x) >= 0.999
    clip_ratio = round(float(clip_mask.sum() / N), 6)
    clip_run_max = _max_true_run(clip_mask)
    kurt = round(float(scipy_kurtosis(x, fisher=True)), 4)
    max_delta = round(float(np.max(np.abs(np.diff(x)))) if N > 1 else 0.0, 6)
    sf = round(_spectral_flatness_audio(x), 6)
    return {
        "rms_db": rms_db,
        "zero_ratio": zero_ratio,
        "clip_ratio": clip_ratio,
        "clip_run_max": clip_run_max,
        "kurtosis": kurt,
        "max_delta": max_delta,
        "spectral_flatness": sf,
    }


def _audio_problem_quality(m: Dict[str, Any]) -> Tuple[str, str]:
    if m["rms_db"] < -60 or m["zero_ratio"] > 0.5:
        problem = "FLAT"
    elif m["clip_ratio"] > 0.001 or m["clip_run_max"] >= 3:
        problem = "CLIPPING"
    elif m["kurtosis"] > 10 or m["max_delta"] > 0.5:
        problem = "ARTIFACT"
    else:
        problem = "NONE"

    if problem in ("FLAT", "CLIPPING"):
        quality = "BAD"
    elif problem == "ARTIFACT" or (problem == "NONE" and m["spectral_flatness"] > 0.5):
        quality = "NOISY"
    else:
        quality = "GOOD"

    return quality, problem


def _process_single_wav(
    data: bytes,
    dataset: str,
    participant_id: str,
    file_id: str,
    window_size_s: float,
) -> Optional[pd.DataFrame]:
    sig = read_wav_signal(data)
    if sig is None:
        return None
    samples = sig["samples"]
    sr = sig["sample_rate_hz"]
    window_n = int(round(sr * window_size_s))
    if window_n == 0:
        return None
    total_windows = math.ceil(len(samples) / window_n)
    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        start = wid * window_n
        end = min((wid + 1) * window_n, len(samples))
        x = samples[start:end]
        if len(x) == 0:
            continue
        m = _audio_window_metrics(x)
        quality, problem = _audio_problem_quality(m)
        rows.append({
            "dataset": dataset,
            "participant_id": participant_id,
            "file_id": file_id,
            "window_start_s": round(start / sr, 3),
            "window_end_s": round(end / sr, 3),
            **m,
            "problem_flag": problem,
            "quality_flag": quality,
        })
    return pd.DataFrame(rows) if rows else None


# ── Video quality helpers ──────────────────────────────────────────────────────

def _video_quality_problem(
    window_blur: float,
    window_clipping: float,
    frozen_ratio: float,
    spike: bool,
) -> Tuple[str, str]:
    # problem_flag hierarchy: FLAT > CLIPPING > ARTIFACT > NONE
    if frozen_ratio > 0.5:
        problem_flag = "FLAT"
    elif window_clipping >= 0.02:
        problem_flag = "CLIPPING"
    elif (
        window_blur < 50
        or frozen_ratio > 0.1
        or spike
    ):
        problem_flag = "ARTIFACT"
    else:
        problem_flag = "NONE"

    # quality_flag hierarchy: BAD > NOISY > GOOD
    if (
        window_blur < 50
        or window_clipping >= 0.02
        or frozen_ratio > 0.5
    ):
        quality_flag = "BAD"
    elif (
        window_blur < 150
        or window_clipping >= 0.005
    ):
        quality_flag = "NOISY"
    else:
        quality_flag = "GOOD"

    return quality_flag, problem_flag


def _compute_video_windows(
    sd: Dict[str, Any],
    total_windows: int,
    window_size_s: float,
) -> List[Dict[str, Any]]:
    fps = sd["sample_rate_hz"]
    frames_per_window = int(round(fps * window_size_s)) if fps > 0 else 1

    lap_vars = sd["lap_vars"]
    clipped_ratios = sd["clipped_ratios"]
    noise_sigmas = sd["noise_sigmas"]
    frame_diffs = sd["frame_diffs"]   # len = n_frames - 1
    n_frames = len(lap_vars)

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        f_start = wid * frames_per_window
        f_end = min((wid + 1) * frames_per_window, n_frames)
        actual = f_end - f_start

        w_start = round(wid * window_size_s, 3)
        w_end = round((wid + 1) * window_size_s, 3)

        if actual == 0:
            rows.append({
                "window_id": wid, "window_start_s": w_start, "window_end_s": w_end,
                "window_blur": float("nan"), "window_clipping": float("nan"),
                "window_noise": float("nan"),
                "frozen_ratio": float("nan"), "spike": False,
                "quality_flag": "BAD", "problem_flag": "FLAT",
            })
            continue

        window_blur = float(np.mean(lap_vars[f_start:f_end]))
        window_clipping = float(np.mean(clipped_ratios[f_start:f_end]))
        window_noise = float(np.mean(noise_sigmas[f_start:f_end]))

        # frame_diffs[i] = diff(frame[i+1], frame[i])
        # Transitions within window: between frames f_start..f_end-1 → diffs at [f_start:f_end-1]
        win_diffs = frame_diffs[f_start:f_end - 1] if f_end - f_start > 1 else np.array([])

        if len(win_diffs) == 0:
            frozen_ratio, spike = 0.0, False
        else:
            frozen_ratio = float(np.sum(win_diffs < 0.001) / len(win_diffs))
            mean_d = float(np.mean(win_diffs))
            if mean_d > 0:
                med_d = float(np.median(win_diffs))
                spike = bool(np.any(win_diffs > 3 * med_d))
            else:
                spike = False

        quality_flag, problem_flag = _video_quality_problem(
            window_blur, window_clipping, frozen_ratio, spike,
        )

        rows.append({
            "window_id": wid,
            "window_start_s": w_start,
            "window_end_s": w_end,
            "window_blur": round(window_blur, 4),
            "window_clipping": round(window_clipping, 6),
            "window_noise": round(window_noise, 4),
            "frozen_ratio": round(frozen_ratio, 4),
            "spike": spike,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return rows


# ── Per-signal processing functions — K-EmoCon audio / video ──────────────────

def process_kemocon_audio(
    file_data: Optional[bytes],
    entity_id: str,
    debate_duration_s: float,
    qf_cfg: Dict[str, Any],
    file_id: str = "audio",
) -> Optional[pd.DataFrame]:
    if file_data is None:
        return None
    window_size_s = float(qf_cfg.get("window_size_s", 1.0))
    return _process_single_wav(file_data, "k-emocon", entity_id, file_id, window_size_s)


def process_kemocon_video(
    file_data: Optional[bytes],
    entity_id: str,
    debate_duration_s: float,
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    if file_data is None:
        return None
    window_size_s = float(qf_cfg.get("window_size_s", 1.0))
    total_windows = math.ceil(debate_duration_s / window_size_s)
    sd = read_video_signal(file_data)
    if sd is None:
        logger.warning("[K-EmoCon] [%s] video: could not decode file", entity_id)
        return None
    rows = _compute_video_windows(sd, total_windows, window_size_s)
    return pd.DataFrame(rows) if rows else None


# ── Per-signal processing functions — EAV ─────────────────────────────────────

def _max_run_length(x: np.ndarray) -> int:
    """Longest consecutive run of identical values in x."""
    if len(x) == 0:
        return 0
    changes = np.concatenate([[True], x[1:] != x[:-1], [True]])
    run_lengths = np.diff(np.where(changes)[0])
    return int(run_lengths.max())


def process_eav_eeg(
    eeg_arr: Optional[np.ndarray],
    entity_id: str,
    eeg_sig_cfg: Dict[str, Any],
    eeg_tp_axis: int,
    eeg_inst_axis: int,
    qf_cfg: Dict[str, Any],
    miss_lookup: Dict[Tuple[str, str, str], float],
    miss_skip: Set[Tuple[str, str, str]],
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Yield (session_id, DataFrame) for each EEG session.

    Each row is one (window, channel) pair with columns:
    window_id, channel_id, samples, std, max_run, peak_to_peak, quality_flag, problem_flag.

    Analyses (applied per channel per window):
      1. Incomplete window — n_samples < 400 → BAD/ARTIFACT; 400–499 → NOISY/ARTIFACT
      2. Flat line         — std < 0.5 µV    → BAD/FLAT
      3. Clipping          — max_run ≥ 5     → BAD/CLIPPING
      4. Artifact          — peak_to_peak > 500 µV → NOISY/ARTIFACT

    Priority: problem FLAT > CLIPPING > ARTIFACT > NONE; quality BAD > NOISY > GOOD.
    """
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}

    if eeg_arr is None or eeg_arr.ndim < 2:
        return

    declared_hz = float(eeg_sig_cfg.get("declared_hz", 500.0))
    n_instances = int(eeg_sig_cfg.get("expected_instances", 200))
    window_size_s = float(qf_cfg.get("window_size_s", 1.0))
    expected_per_window = int(round(declared_hz * window_size_s))       # 500 at 500 Hz
    incomplete_bad_thr = int(round(0.8 * expected_per_window))          # 400

    n_instances_actual = eeg_arr.shape[eeg_inst_axis] if eeg_arr.ndim == 3 else 1

    for inst_idx in range(min(n_instances, n_instances_actual)):
        session_id = f"{inst_idx:03d}"

        if (entity_id, session_id, "eeg") in miss_skip:
            logger.info("[EAV] [%s] Session %s → excluded (total_missing)", entity_id, session_id)
            continue

        # Extract (timepoints, channels) slice for this session
        if eeg_arr.ndim == 3:
            eeg_slice = np.take(eeg_arr, inst_idx, axis=eeg_inst_axis)
            tp_axis_in_slice = eeg_tp_axis if eeg_inst_axis > eeg_tp_axis else eeg_tp_axis - 1
            if tp_axis_in_slice != 0:
                eeg_slice = np.moveaxis(eeg_slice, tp_axis_in_slice, 0)
        else:
            eeg_slice = eeg_arr

        if eeg_slice.ndim == 1:
            eeg_slice = eeg_slice.reshape(-1, 1)

        n_timepoints, n_channels = eeg_slice.shape
        total_windows = math.ceil(n_timepoints / expected_per_window)

        rows: List[Dict[str, Any]] = []
        for wid in range(total_windows):
            start_idx = wid * expected_per_window
            end_idx = min((wid + 1) * expected_per_window, n_timepoints)
            window_data = eeg_slice[start_idx:end_idx, :]

            for ch_idx in range(n_channels):
                x = window_data[:, ch_idx]
                x = x[np.isfinite(x)]
                n_samples = len(x)

                # Analysis 1 — incomplete window
                if n_samples < incomplete_bad_thr:
                    a1_quality, a1_problem = "BAD", "ARTIFACT"
                elif n_samples < expected_per_window:
                    a1_quality, a1_problem = "NOISY", "ARTIFACT"
                else:
                    a1_quality, a1_problem = "GOOD", "NONE"

                if n_samples == 0:
                    rows.append({
                        "window_id": wid,
                        "channel_id": ch_idx,
                        "samples": 0,
                        "std": float("nan"),
                        "max_run": 0,
                        "peak_to_peak": float("nan"),
                        "quality_flag": "BAD",
                        "problem_flag": "ARTIFACT",
                    })
                    continue

                std_val = round(float(np.std(x)), 6)
                max_run = _max_run_length(x)
                peak_to_peak = round(float(np.max(x) - np.min(x)), 4)

                # Analysis 2 — flat line
                a2_quality = "BAD" if std_val < 0.5 else "GOOD"
                a2_problem = "FLAT" if std_val < 0.5 else "NONE"

                # Analysis 3 — clipping
                a3_quality = "BAD" if max_run >= 5 else "GOOD"
                a3_problem = "CLIPPING" if max_run >= 5 else "NONE"

                # Analysis 4 — artifact
                a4_quality = "NOISY" if peak_to_peak > 500 else "GOOD"
                a4_problem = "ARTIFACT" if peak_to_peak > 500 else "NONE"

                all_results = [
                    (a1_quality, a1_problem),
                    (a2_quality, a2_problem),
                    (a3_quality, a3_problem),
                    (a4_quality, a4_problem),
                ]
                quality_flag = max((r[0] for r in all_results), key=lambda q: _q[q])
                problem_set = {r[1] for r in all_results}
                problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)

                rows.append({
                    "window_id": wid,
                    "channel_id": ch_idx,
                    "samples": n_samples,
                    "std": std_val,
                    "max_run": max_run,
                    "peak_to_peak": peak_to_peak,
                    "quality_flag": quality_flag,
                    "problem_flag": problem_flag,
                })

        if rows:
            logger.info(
                "[EAV] [%s] Session %s → %d windows × %d channels",
                entity_id, session_id, total_windows, n_channels,
            )
            yield session_id, pd.DataFrame(rows)


def process_eav_audio(
    wav_objs: Dict[str, Any],
    entity_id: str,
    minio_client,
    bucket: str,
    sig_cfg: Dict[str, Any],
    qf_cfg: Dict[str, Any],
    miss_lookup: Dict[Tuple[str, str, str], float],
    miss_skip: Set[Tuple[str, str, str]],
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Yield (source_object_name, df) for each trial WAV file."""
    window_size_s = float(qf_cfg.get("window_size_s", 1.0))
    for trial_id, obj in sorted(wav_objs.items(), key=lambda kv: kv[0]):
        if (entity_id, trial_id, "audio") in miss_skip:
            logger.info("[EAV] [%s] Trial %s audio → excluded (total_missing)", entity_id, trial_id)
            continue
        data = download_object(minio_client, bucket, obj.object_name)
        if data is None:
            logger.warning("[EAV] [%s] Trial %s audio → download failed", entity_id, trial_id)
            continue
        file_id = obj.object_name.split("/")[-1]
        df = _process_single_wav(data, "eav", entity_id, file_id, window_size_s)
        if df is not None and not df.empty:
            yield obj.object_name, df
        else:
            logger.warning("[EAV] [%s] Trial %s audio → no windows produced", entity_id, trial_id)


def process_eav_video(
    mp4_objs: Dict[str, Any],
    entity_id: str,
    minio_client,
    bucket: str,
    sig_cfg: Dict[str, Any],
    qf_cfg: Dict[str, Any],
    miss_lookup: Dict[Tuple[str, str, str], float],
    miss_skip: Set[Tuple[str, str, str]],
) -> Iterator[Tuple[str, pd.DataFrame]]:
    """Yield (source_object_name, df) for each trial MP4 file."""
    window_size_s = float(qf_cfg.get("window_size_s", 1.0))
    trial_duration_s = float(sig_cfg.get("trial_duration_s", 20.0))
    total_windows = math.ceil(trial_duration_s / window_size_s)

    for trial_id, obj in sorted(mp4_objs.items(), key=lambda kv: kv[0]):
        if (entity_id, trial_id, "video") in miss_skip:
            logger.info("[EAV] [%s] Trial %s video → excluded (total_missing)", entity_id, trial_id)
            continue
        data = download_object(minio_client, bucket, obj.object_name)
        if data is None:
            logger.warning("[EAV] [%s] Trial %s video → download failed", entity_id, trial_id)
            continue
        sd = read_video_signal(data)
        if sd is None:
            logger.warning("[EAV] [%s] Trial %s video → could not decode", entity_id, trial_id)
            continue
        rows = _compute_video_windows(sd, total_windows, window_size_s)
        if rows:
            yield obj.object_name, pd.DataFrame(rows)
        else:
            logger.warning("[EAV] [%s] Trial %s video → no windows produced", entity_id, trial_id)


# ── Signal dispatch — K-EmoCon ─────────────────────────────────────────────────

_KEMOCON_SIGNAL_PROCESSORS = {
    "E4_BVP":    process_e4_bvp,
    "E4_EDA":    process_e4_eda,
    "E4_ACC":    process_e4_acc,
    "E4_HR":     process_e4_hr,
    "E4_IBI":    process_e4_ibi,
    "E4_TEMP":   process_e4_temp,
    "BrainWave": process_brainwave,
    "Attention": process_attention,
    "Meditation": process_meditation,
    "Polar_HR":  process_polar_hr,
    "audio":     process_kemocon_audio,
    "video":     process_kemocon_video,
}


# ── K-EmoCon entity processing ─────────────────────────────────────────────────

def process_kemocon_entity(
    minio_client,
    bucket: str,
    entity_id: str,
    objects: List[Any],
    subjects_map: Dict[int, Dict[str, int]],
    kemocon_md_cfg: Dict[str, Any],
    qf_cfg: Dict[str, Any],
    miss_lookup: Dict[Tuple[str, str, str], float],
    miss_skip: Set[Tuple[str, str, str]],
    output_prefix: str,
    skip_video: bool = False,
    video_only: bool = False,
) -> None:
    """Process all signals for one K-EmoCon entity; upload one CSV per signal."""
    pid = _pid_from_entity_id(entity_id)
    subject = subjects_map.get(pid)
    if not subject:
        logger.warning("[K-EmoCon] [%s] No subject entry — skipping", entity_id)
        return

    debate_start_s = subject["startTime"] / 1000.0
    debate_end_s = subject["endTime"] / 1000.0
    debate_duration_s = debate_end_s - debate_start_s

    timestamp_col = kemocon_md_cfg.get("timestamp_col", "timestamp")
    ts_unit_ms = kemocon_md_cfg.get("timestamp_unit_ms", False)
    expected_signals: List[Dict] = kemocon_md_cfg.get("expected_signals", [])

    obj_by_fname: Dict[str, Any] = {}
    obj_by_ext: Dict[str, List[Any]] = {}
    for obj in objects:
        fname = obj.object_name.split("/")[-1]
        obj_by_fname[fname] = obj
        obj_by_ext.setdefault(Path(fname).suffix.lower(), []).append(obj)

    computed_dfs: Dict[str, pd.DataFrame] = {}

    for sig in expected_signals:
        signal_type = sig["signal_type"]
        modality = sig.get("modality", signal_type.lower())
        if skip_video and signal_type == "video":
            logger.info("[K-EmoCon] [%s] video — skipped (--skip-video)", entity_id)
            continue
        if video_only and signal_type != "video":
            continue
        processor = _KEMOCON_SIGNAL_PROCESSORS.get(signal_type)
        if processor is None:
            logger.warning("[K-EmoCon] [%s] No processor for signal '%s' — skipping", entity_id, signal_type)
            continue

        # Locate the file for this signal
        file_obj = None
        if sig.get("filename"):
            file_obj = obj_by_fname.get(sig["filename"])
        elif sig.get("ext"):
            candidates = obj_by_ext.get(sig["ext"], [])
            file_obj = candidates[0] if candidates else None

        file_data: Optional[bytes] = None
        if file_obj is not None:
            file_data = download_object(minio_client, bucket, file_obj.object_name)

        # Route to the appropriate processor
        declared_hz: Optional[float] = sig.get("declared_hz")
        sample_rate_hz = miss_lookup.get((entity_id, entity_id, signal_type)) or declared_hz

        try:
            if signal_type == "audio":
                file_id = file_obj.object_name.split("/")[-1] if file_obj else "audio"
                df = processor(file_data, entity_id, debate_duration_s, qf_cfg, file_id=file_id)
            elif signal_type == "video":
                df = processor(file_data, entity_id, debate_duration_s, qf_cfg)
            elif signal_type == "E4_IBI":
                df = processor(
                    file_data, entity_id, debate_start_s, debate_duration_s,
                    timestamp_col, ts_unit_ms, qf_cfg,
                )
            elif signal_type == "E4_HR":
                df = processor(
                    file_data, entity_id, debate_start_s, debate_duration_s,
                    timestamp_col, ts_unit_ms, sample_rate_hz, qf_cfg,
                    bvp_df=computed_dfs.get("E4_BVP"),
                )
            elif signal_type == "Polar_HR":
                df = processor(
                    file_data, entity_id, debate_start_s, debate_duration_s,
                    timestamp_col, ts_unit_ms, sample_rate_hz, qf_cfg,
                    e4_hr_df=computed_dfs.get("E4_HR"),
                )
            else:
                df = processor(
                    file_data, entity_id, debate_start_s, debate_duration_s,
                    timestamp_col, ts_unit_ms, sample_rate_hz, qf_cfg,
                )
        except Exception as e:
            logger.error("[K-EmoCon] [%s] %s: processor failed: %s", entity_id, signal_type, e)
            continue

        if df is None or df.empty:
            logger.warning("[K-EmoCon] [%s] %s: no output produced", entity_id, signal_type)
            continue

        computed_dfs[signal_type] = df

        if signal_type == "audio" and file_obj is not None:
            stem = Path(file_obj.object_name.split("/")[-1]).stem
            output_key = (
                f"{output_prefix}/k-emocon/files/entity={entity_id}"
                f"/modality={modality}/{stem}_quality_flags.csv"
            )
        else:
            output_key = (
                f"{output_prefix}/k-emocon/files/entity={entity_id}"
                f"/modality={modality}/{_output_filename(entity_id, signal_type)}"
            )
        upload_csv(minio_client, bucket, output_key, df)
        logger.info("[K-EmoCon] [%s] %-12s → uploaded %s", entity_id, signal_type, output_key)


# ── EAV entity processing ──────────────────────────────────────────────────────

def process_eav_entity(
    minio_client,
    bucket: str,
    entity_id: str,
    objects: List[Any],
    eav_md_cfg: Dict[str, Any],
    qf_cfg: Dict[str, Any],
    miss_lookup: Dict[Tuple[str, str, str], float],
    miss_skip: Set[Tuple[str, str, str]],
    output_prefix: str,
    skip_video: bool = False,
    video_only: bool = False,
) -> None:
    """Process all signals for one EAV entity; upload one CSV per signal."""
    eeg_label_suffix = eav_md_cfg.get("eeg_label_suffix", "_label")
    eeg_tp_axis = int(eav_md_cfg.get("eeg_timepoints_axis", 0))
    eeg_inst_axis = int(eav_md_cfg.get("eeg_instances_axis", 2))
    trial_id_pattern = eav_md_cfg.get("trial_id_pattern", r"^(\d+)_")
    expected_signals: List[Dict] = eav_md_cfg.get("expected_signals", [])

    def _extract_trial_id(filename: str) -> Optional[str]:
        m = re.match(trial_id_pattern, filename)
        return m.group(1) if m else None

    mat_objs = [
        o for o in objects
        if o.object_name.endswith(".mat")
        and not Path(o.object_name).stem.endswith(eeg_label_suffix)
    ]
    wav_objs = {
        _extract_trial_id(o.object_name.split("/")[-1]): o
        for o in objects if o.object_name.endswith(".wav")
    }
    mp4_objs = {
        _extract_trial_id(o.object_name.split("/")[-1]): o
        for o in objects if o.object_name.endswith(".mp4")
    }
    wav_objs = {k: v for k, v in wav_objs.items() if k is not None}
    mp4_objs = {k: v for k, v in mp4_objs.items() if k is not None}

    for sig in expected_signals:
        signal_type = sig["signal_type"]
        modality = sig.get("modality", signal_type.lower())
        ext = sig.get("ext", "")

        if skip_video and signal_type == "video":
            logger.info("[EAV] [%s] video — skipped (--skip-video)", entity_id)
            continue
        if video_only and signal_type != "video":
            continue

        try:
            if signal_type == "eeg" and ext == ".mat":
                eeg_arr: Optional[np.ndarray] = None
                if mat_objs:
                    mat_data = download_object(minio_client, bucket, mat_objs[0].object_name)
                    if mat_data is not None:
                        eeg_arr = load_mat_eeg(mat_data)
                        del mat_data
                        if eeg_arr is not None:
                            logger.info("[EAV] [%s] EEG mat shape: %s", entity_id, eeg_arr.shape)

                n_uploaded = 0
                for session_id, df in process_eav_eeg(
                    eeg_arr, entity_id, sig, eeg_tp_axis, eeg_inst_axis,
                    qf_cfg, miss_lookup, miss_skip,
                ):
                    output_key = (
                        f"{output_prefix}/eav/files/entity={entity_id}"
                        f"/modality={modality}/{entity_id}_{session_id}_biosignal_eeg_quality_flags.csv"
                    )
                    upload_csv(minio_client, bucket, output_key, df)
                    n_uploaded += 1
                if n_uploaded == 0:
                    logger.warning("[EAV] [%s] eeg: no session output produced", entity_id)
                else:
                    logger.info("[EAV] [%s] eeg → uploaded %d session files", entity_id, n_uploaded)
                continue

            elif signal_type == "audio" and ext == ".wav":
                n_uploaded = 0
                for source_path, df in process_eav_audio(
                    wav_objs, entity_id, minio_client, bucket,
                    sig, qf_cfg, miss_lookup, miss_skip,
                ):
                    stem = Path(source_path.split("/")[-1]).stem
                    output_key = (
                        f"{output_prefix}/eav/files/entity={entity_id}"
                        f"/modality={modality}/{stem}_quality_flags.csv"
                    )
                    upload_csv(minio_client, bucket, output_key, df)
                    n_uploaded += 1
                if n_uploaded == 0:
                    logger.warning("[EAV] [%s] audio: no trial output produced", entity_id)
                else:
                    logger.info("[EAV] [%s] audio → uploaded %d trial files", entity_id, n_uploaded)
                continue

            elif signal_type == "video" and ext == ".mp4":
                n_uploaded = 0
                for source_path, df in process_eav_video(
                    mp4_objs, entity_id, minio_client, bucket,
                    sig, qf_cfg, miss_lookup, miss_skip,
                ):
                    stem = Path(source_path.split("/")[-1]).stem
                    output_key = (
                        f"{output_prefix}/eav/files/entity={entity_id}"
                        f"/modality={modality}/{stem}_quality_flags.csv"
                    )
                    upload_csv(minio_client, bucket, output_key, df)
                    n_uploaded += 1
                if n_uploaded == 0:
                    logger.warning("[EAV] [%s] video: no trial output produced", entity_id)
                else:
                    logger.info("[EAV] [%s] video → uploaded %d trial files", entity_id, n_uploaded)
                continue

            else:
                logger.warning("[EAV] [%s] Unknown signal '%s' (ext=%s) — skipping", entity_id, signal_type, ext)
                continue

        except Exception as e:
            logger.error("[EAV] [%s] %s: processor failed: %s", entity_id, signal_type, e)
            continue


# ── Orchestration ──────────────────────────────────────────────────────────────

def run_quality_flags(
    minio_client,
    silver_bucket: str,
    cfg: Dict[str, Any],
    dataset_filter: str,
    test_mode: bool,
    skip_video: bool = False,
    video_only: bool = False,
) -> None:
    qf_cfg = cfg.get("quality_flags", {})
    output_prefix = qf_cfg.get("output_prefix", "04_quality_flags").rstrip("/")
    md_cfg = cfg.get("missingness_detection", {})

    miss_report_prefix = md_cfg.get("output_prefix", "03_missingness").rstrip("/")
    miss_report_file = md_cfg.get("output_report_filename", "missingness_report.csv")
    miss_lookup, miss_skip = load_missingness_report(
        minio_client, silver_bucket, f"{miss_report_prefix}/{miss_report_file}"
    )
    logger.info(
        "Missingness report: %d sample-rate entries, %d total_missing entries",
        len(miss_lookup), len(miss_skip),
    )

    run_kemocon = dataset_filter in ("all", "k-emocon")
    run_eav = dataset_filter in ("all", "eav")

    # ── K-EmoCon ──────────────────────────────────────────────────────────────
    if run_kemocon:
        kemocon_md = md_cfg.get("datasets", {}).get("kemocon", {})
        kemocon_qf = qf_cfg.get("datasets", {}).get("kemocon", {})
        if kemocon_md:
            logger.info("=== Processing K-EmoCon ===")
            subjects_bucket = kemocon_qf.get("subjects_bucket") or kemocon_md.get("subjects_bucket", "silver")
            subjects_path = kemocon_qf.get("subjects_path") or kemocon_md.get("subjects_path", "")
            silver_files_prefix = kemocon_qf.get("silver_files_prefix") or kemocon_md.get("silver_files_prefix", "")

            subjects_map = load_kemocon_subjects(minio_client, subjects_bucket, subjects_path)
            logger.info("K-EmoCon subjects loaded: %d", len(subjects_map))

            entity_objects = _group_objects_by_entity(minio_client, silver_bucket, silver_files_prefix)
            logger.info("K-EmoCon entities: %d", len(entity_objects))

            entity_ids = sorted(entity_objects.keys())
            if test_mode and entity_ids:
                entity_ids = entity_ids[:1]
                logger.info("TEST MODE: processing only %s", entity_ids[0])

            for entity_id in entity_ids:
                logger.info("[K-EmoCon] Processing %s", entity_id)
                try:
                    process_kemocon_entity(
                        minio_client, silver_bucket, entity_id,
                        entity_objects[entity_id], subjects_map,
                        kemocon_md, qf_cfg, miss_lookup, miss_skip,
                        output_prefix, skip_video=skip_video, video_only=video_only,
                    )
                except Exception as e:
                    logger.error("[K-EmoCon] [%s] Fatal error: %s", entity_id, e)

    # ── EAV ───────────────────────────────────────────────────────────────────
    if run_eav:
        eav_md = md_cfg.get("datasets", {}).get("eav", {})
        eav_qf = qf_cfg.get("datasets", {}).get("eav", {})
        if eav_md:
            logger.info("=== Processing EAV ===")
            silver_files_prefix = eav_qf.get("silver_files_prefix") or eav_md.get("silver_files_prefix", "")

            entity_objects = _group_objects_by_entity(minio_client, silver_bucket, silver_files_prefix)
            logger.info("EAV entities: %d", len(entity_objects))

            entity_ids = sorted(entity_objects.keys())
            if test_mode and entity_ids:
                entity_ids = entity_ids[:1]
                logger.info("TEST MODE: processing only %s", entity_ids[0])

            for entity_id in entity_ids:
                logger.info("[EAV] Processing %s", entity_id)
                try:
                    process_eav_entity(
                        minio_client, silver_bucket, entity_id,
                        entity_objects[entity_id], eav_md, qf_cfg,
                        miss_lookup, miss_skip, output_prefix, skip_video=skip_video, video_only=video_only,
                    )
                except Exception as e:
                    logger.error("[EAV] [%s] Fatal error: %s", entity_id, e)


# ── CLI ────────────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver — Step 04: Noise Detection & Quality Flags.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument(
        "--dataset", default="all", choices=["k-emocon", "eav", "all"],
        help="Dataset to process (default: all).",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Process only the first entity of k-emocon (for development/testing).",
    )
    parser.add_argument(
        "--skip-video", action="store_true",
        help="Skip video signal processing (faster runs without CV-heavy checks).",
    )
    parser.add_argument(
        "--video-only", action="store_true",
        help="Process only video signals, skip all other modalities.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client, _ = project_config.config()
    silver_bucket = cfg["bucket_silver"]

    logger.info("Starting Silver — Step 04: Noise Detection & Quality Flags")

    run_quality_flags(
        minio_client, silver_bucket, cfg,
        dataset_filter=args.dataset,
        test_mode=args.test,
        skip_video=args.skip_video,
        video_only=args.video_only,
    )

    logger.info("Quality Flags complete.")


if __name__ == "__main__":
    main()
