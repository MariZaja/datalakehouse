import io
import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _read_e4_csv(data: bytes) -> Tuple[float, float, np.ndarray]:
    """Parse Empatica E4 CSV in silver format.

    Silver format: header row with 'timestamp' (Unix ms), 'pid', 'value' (scalar
    signals) or 'x','y','z' (ACC), and optional metadata columns.
    Returns (start_ts_s, sfreq_hz, arr):
      start_ts_s: Unix time of first sample in seconds
      sfreq_hz:   sample rate inferred from median inter-sample interval
      arr:        1-D for scalar signals; (n, 3) for ACC
    """
    df = pd.read_csv(io.BytesIO(data))
    if "timestamp" not in df.columns:
        raise ValueError("E4 CSV missing 'timestamp' column")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("E4 CSV has no valid rows")
    ts_ms = df["timestamp"].values
    start_ts = float(ts_ms[0]) / 1000.0
    if len(ts_ms) > 1:
        diffs_s = np.diff(ts_ms) / 1000.0
        pos_diffs = diffs_s[diffs_s > 0]
        sfreq = 1.0 / float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0
    else:
        sfreq = 1.0
    if all(c in df.columns for c in ("x", "y", "z")):
        for c in ("x", "y", "z"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return start_ts, sfreq, df[["x", "y", "z"]].values.astype(np.float64)
    if "value" not in df.columns:
        raise ValueError("E4 CSV has no 'value' column")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return start_ts, sfreq, df["value"].values.astype(np.float64)


def _sliding_windows(
    rel_times: np.ndarray,
    values: np.ndarray,
    total_duration_s: float,
    window_size_s: float,
    step_s: float,
    entity_id: str,
    feature_fn: Callable,
) -> List[Dict[str, Any]]:
    """Apply feature_fn over lookback sliding windows; returns list of row dicts.

    Each window ends at the current step position and looks back window_size_s.
    window_id=1 covers [-window_size_s+step_s, step_s], i.e. the first window ends at step_s (not 0).
    """
    rows: List[Dict[str, Any]] = []
    wid = 1
    while True:
        end_s = round(wid * step_s, 6)
        if end_s > total_duration_s:
            break
        start_s = end_s - window_size_s
        mask = (rel_times >= start_s) & (rel_times < end_s)
        feats = feature_fn(values[mask])
        rows.append({
            "entity_id": entity_id,
            "window_id": wid,
            "window_start_s": round(start_s, 3),
            "window_end_s": round(end_s, 3),
            **feats,
        })
        wid += 1
    return rows


# ── BVP ───────────────────────────────────────────────────────────────────────

def _bvp_window_features(window: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Compute BVP features for one fixed-length window.

    Detects systolic peaks and returns mean peak amplitude, mean rise time,
    SDNN, RMSSD, pNN50. HRV metrics require >= 3 peaks and return NaN when
    insufficient (typical for 300 ms windows at normal heart rates).
    """
    from scipy.signal import find_peaks

    nan = float("nan")
    result: Dict[str, float] = {
        "bvp_mean_peak_amplitude": nan,
        "bvp_mean_rise_time_s": nan,
        "bvp_sdnn_ms": nan,
        "bvp_rmssd_ms": nan,
        "bvp_pnn50": nan,
    }
    if len(window) < 3:
        return result

    min_dist = max(1, int(sfreq * 0.3))  # 200 BPM ceiling
    peaks, _ = find_peaks(window, distance=min_dist)
    if len(peaks) == 0:
        return result

    result["bvp_mean_peak_amplitude"] = float(np.mean(window[peaks]))

    rise_times: List[float] = []
    for peak_idx in peaks:
        search_start = max(0, peak_idx - int(sfreq * 0.5))
        segment = window[search_start:peak_idx]
        if len(segment) > 0:
            trough_local = int(np.argmin(segment))
            rise_times.append((peak_idx - (search_start + trough_local)) / sfreq)
    if rise_times:
        result["bvp_mean_rise_time_s"] = float(np.mean(rise_times))

    if len(peaks) >= 2:
        nn_ms = np.diff(peaks) / sfreq * 1000.0
        if len(nn_ms) >= 2:
            result["bvp_sdnn_ms"] = float(np.std(nn_ms, ddof=1))
            diffs = np.diff(nn_ms)
            result["bvp_rmssd_ms"] = float(np.sqrt(np.mean(diffs ** 2)))
            result["bvp_pnn50"] = float(np.sum(np.abs(diffs) > 50.0) / len(diffs))
    return result


def extract_bvp_features(
    data: bytes,
    entity_id: str,
    window_size_s: float = 0.3,
    debate_start_ms: Optional[float] = None,
    debate_end_ms: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Extract BVP features in fixed-length windows from an Empatica E4 BVP CSV.

    Features per window: bvp_mean_peak_amplitude, bvp_mean_rise_time_s,
    bvp_sdnn_ms, bvp_rmssd_ms, bvp_pnn50.
    Signal is trimmed to the debate window when timestamps are provided.
    """
    try:
        from scipy.signal import find_peaks  # noqa: F401
    except ImportError:
        logger.error("scipy not installed — BVP extraction skipped")
        return None

    try:
        start_ts, sfreq, arr = _read_e4_csv(data)
    except Exception as e:
        logger.error("[BVP] [%s] Failed to parse E4 BVP CSV: %s", entity_id, e)
        return None

    if arr.ndim != 1 or len(arr) == 0:
        logger.error("[BVP] [%s] Unexpected BVP array shape: %s", entity_id, arr.shape)
        return None

    if debate_start_ms is not None and debate_end_ms is not None:
        s_idx = int(round((debate_start_ms / 1000.0 - start_ts) * sfreq))
        e_idx = int(round((debate_end_ms / 1000.0 - start_ts) * sfreq))
        arr = arr[max(0, s_idx):min(len(arr), e_idx)]

    if len(arr) == 0:
        logger.warning("[BVP] [%s] No BVP samples in debate window", entity_id)
        return None

    window_n = max(1, int(round(sfreq * window_size_s)))
    n_windows = math.ceil(len(arr) / window_n)
    rows: List[Dict[str, Any]] = []
    for wid in range(1, n_windows + 1):
        s = (wid - 1) * window_n
        e = min(wid * window_n, len(arr))
        feats = _bvp_window_features(arr[s:e], sfreq)
        rows.append({
            "entity_id": entity_id,
            "window_id": wid,
            "window_start_s": round(s / sfreq, 3),
            "window_end_s": round((wid * window_n) / sfreq, 3),
            **feats,
        })
    return pd.DataFrame(rows) if rows else None


# ── ACC ───────────────────────────────────────────────────────────────────────

def extract_acc_features(
    data: bytes,
    entity_id: str,
    window_size_s: float = 0.3,
    debate_start_ms: Optional[float] = None,
    debate_end_ms: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Extract ACC features in fixed-length windows from an Empatica E4 ACC CSV.

    Features per window: acc_{x,y,z}_mean and acc_{x,y,z}_std.
    Signal is trimmed to the debate window when timestamps are provided.
    """
    try:
        start_ts, sfreq, arr = _read_e4_csv(data)
    except Exception as e:
        logger.error("[ACC] [%s] Failed to parse E4 ACC CSV: %s", entity_id, e)
        return None

    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        logger.error("[ACC] [%s] Unexpected ACC array shape: %s", entity_id, arr.shape)
        return None

    if debate_start_ms is not None and debate_end_ms is not None:
        s_idx = int(round((debate_start_ms / 1000.0 - start_ts) * sfreq))
        e_idx = int(round((debate_end_ms / 1000.0 - start_ts) * sfreq))
        arr = arr[max(0, s_idx):min(len(arr), e_idx)]

    if len(arr) == 0:
        logger.warning("[ACC] [%s] No ACC samples in debate window", entity_id)
        return None

    window_n = max(1, int(round(sfreq * window_size_s)))
    n_windows = math.ceil(len(arr) / window_n)
    nan = float("nan")
    rows: List[Dict[str, Any]] = []
    for wid in range(1, n_windows + 1):
        s = (wid - 1) * window_n
        e = min(wid * window_n, len(arr))
        chunk = arr[s:e]
        row: Dict[str, Any] = {
            "entity_id": entity_id,
            "window_id": wid,
            "window_start_s": round(s / sfreq, 3),
            "window_end_s": round((wid * window_n) / sfreq, 3),
        }
        for i, axis in enumerate(("x", "y", "z")):
            col = chunk[:, i]
            row[f"acc_{axis}_mean"] = float(np.mean(col))
            row[f"acc_{axis}_std"] = float(np.std(col, ddof=1)) if len(col) > 1 else nan
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


# ── EDA ───────────────────────────────────────────────────────────────────────

def _eda_window_features(window: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Compute EDA features for one sliding window.

    SCR (skin conductance response) peaks are detected with a minimum inter-peak
    distance of 1 s. Amplitude of each SCR = peak value − preceding local trough.
    """
    from scipy.signal import find_peaks

    nan = float("nan")
    result: Dict[str, float] = {
        "eda_mean": nan,
        "eda_std": nan,
        "eda_slope": nan,
        "eda_scr_count": nan,
        "eda_scr_amplitude_mean": nan,
    }
    n = len(window)
    if n == 0:
        return result
    result["eda_mean"] = float(np.mean(window))
    result["eda_std"] = float(np.std(window, ddof=1)) if n > 1 else nan
    if n >= 2:
        t = np.arange(n, dtype=np.float64) / sfreq
        slope, _ = np.polyfit(t, window, 1)
        result["eda_slope"] = float(slope)
    min_dist = max(1, int(sfreq * 1.0))
    peaks, _ = find_peaks(window, distance=min_dist)
    result["eda_scr_count"] = float(len(peaks))
    if len(peaks) > 0:
        look_back = int(sfreq * 2.0)
        amps: List[float] = []
        for p in peaks:
            seg = window[max(0, p - look_back):p]
            trough = float(np.min(seg)) if len(seg) > 0 else float(window[p])
            amps.append(float(window[p]) - trough)
        result["eda_scr_amplitude_mean"] = float(np.mean(amps))
    return result


def extract_eda_features(
    data: bytes,
    entity_id: str,
    window_size_s: float = 30.0,
    step_s: float = 0.3,
    debate_start_ms: Optional[float] = None,
    debate_end_ms: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Extract EDA features in sliding windows from an Empatica E4 EDA CSV.

    Features per window: eda_mean (SCL), eda_std, eda_slope, eda_scr_count,
    eda_scr_amplitude_mean. Signal is trimmed to the debate window when provided.
    """
    try:
        from scipy.signal import find_peaks  # noqa: F401
    except ImportError:
        logger.error("scipy not installed — EDA extraction skipped")
        return None

    try:
        start_ts, sfreq, arr = _read_e4_csv(data)
    except Exception as e:
        logger.error("[EDA] [%s] Failed to parse E4 EDA CSV: %s", entity_id, e)
        return None

    if arr.ndim != 1 or len(arr) == 0:
        logger.error("[EDA] [%s] Unexpected EDA array shape: %s", entity_id, arr.shape)
        return None
    if sfreq <= 0:
        logger.error("[EDA] [%s] Invalid sfreq=%.3f", entity_id, sfreq)
        return None

    abs_times = start_ts + np.arange(len(arr)) / sfreq

    if debate_start_ms is not None and debate_end_ms is not None:
        # Extend the trim backwards by window_size_s so lookback windows at t=0 have data.
        trim_start_s = debate_start_ms / 1000.0 - window_size_s
        mask = (abs_times >= trim_start_s) & (abs_times <= debate_end_ms / 1000.0)
        rel_times = abs_times[mask] - debate_start_ms / 1000.0
        eda_vals = arr[mask]
        total_s = (debate_end_ms - debate_start_ms) / 1000.0
    else:
        rel_times = abs_times - start_ts
        eda_vals = arr
        total_s = rel_times[-1] + 1.0 / sfreq if len(arr) > 0 else 0.0

    if len(eda_vals) == 0:
        logger.warning("[EDA] [%s] No samples in debate window", entity_id)
        return None

    rows = _sliding_windows(
        rel_times, eda_vals, total_s, window_size_s, step_s, entity_id,
        lambda w: _eda_window_features(w, sfreq),
    )
    return pd.DataFrame(rows) if rows else None


# ── TEMP ──────────────────────────────────────────────────────────────────────

def _temp_window_features(window: np.ndarray, sfreq: float) -> Dict[str, float]:
    """Compute TEMP features for one sliding window."""
    nan = float("nan")
    result: Dict[str, float] = {
        "temp_mean": nan,
        "temp_std": nan,
        "temp_slope": nan,
    }
    n = len(window)
    if n == 0:
        return result
    result["temp_mean"] = float(np.mean(window))
    result["temp_std"] = float(np.std(window, ddof=1)) if n > 1 else nan
    if n >= 2:
        t = np.arange(n, dtype=np.float64) / sfreq
        slope, _ = np.polyfit(t, window, 1)
        result["temp_slope"] = float(slope)
    return result


def extract_temp_features(
    data: bytes,
    entity_id: str,
    window_size_s: float = 30.0,
    step_s: float = 0.3,
    debate_start_ms: Optional[float] = None,
    debate_end_ms: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Extract TEMP features in sliding windows from an Empatica E4 TEMP CSV.

    Features per window: temp_mean, temp_std, temp_slope.
    Signal is trimmed to the debate window when provided.
    """
    try:
        start_ts, sfreq, arr = _read_e4_csv(data)
    except Exception as e:
        logger.error("[TEMP] [%s] Failed to parse E4 TEMP CSV: %s", entity_id, e)
        return None

    if arr.ndim != 1 or len(arr) == 0:
        logger.error("[TEMP] [%s] Unexpected TEMP array shape: %s", entity_id, arr.shape)
        return None
    if sfreq <= 0:
        logger.error("[TEMP] [%s] Invalid sfreq=%.3f", entity_id, sfreq)
        return None

    abs_times = start_ts + np.arange(len(arr)) / sfreq

    if debate_start_ms is not None and debate_end_ms is not None:
        trim_start_s = debate_start_ms / 1000.0 - window_size_s
        mask = (abs_times >= trim_start_s) & (abs_times <= debate_end_ms / 1000.0)
        rel_times = abs_times[mask] - debate_start_ms / 1000.0
        temp_vals = arr[mask]
        total_s = (debate_end_ms - debate_start_ms) / 1000.0
    else:
        rel_times = abs_times - start_ts
        temp_vals = arr
        total_s = rel_times[-1] + 1.0 / sfreq if len(arr) > 0 else 0.0

    if len(temp_vals) == 0:
        logger.warning("[TEMP] [%s] No samples in debate window", entity_id)
        return None

    rows = _sliding_windows(
        rel_times, temp_vals, total_s, window_size_s, step_s, entity_id,
        lambda w: _temp_window_features(w, sfreq),
    )
    return pd.DataFrame(rows) if rows else None


# ── HR ────────────────────────────────────────────────────────────────────────

def _hr_window_features(hr_bpm: np.ndarray) -> Dict[str, float]:
    """Compute HR features for one window from HR values in BPM."""
    nan = float("nan")
    result: Dict[str, float] = {
        "hr_mean_bpm": nan,
        "hr_mean_nn_ms": nan,
        "hr_std_bpm": nan,
        "hr_min_nn_ms": nan,
    }
    hr_bpm = hr_bpm[hr_bpm > 0]
    if len(hr_bpm) == 0:
        return result
    nn_ms = 60000.0 / hr_bpm
    result["hr_mean_bpm"] = float(np.mean(hr_bpm))
    result["hr_mean_nn_ms"] = float(np.mean(nn_ms))
    result["hr_min_nn_ms"] = float(np.min(nn_ms))
    if len(hr_bpm) > 1:
        result["hr_std_bpm"] = float(np.std(hr_bpm, ddof=1))
    return result


def extract_e4_hr_features(
    data: bytes,
    entity_id: str,
    window_size_s: float = 10.0,
    step_s: float = 0.3,
    debate_start_ms: Optional[float] = None,
    debate_end_ms: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Extract HR features in sliding windows from an Empatica E4 HR CSV.

    Features per window: hr_mean_bpm, hr_mean_nn_ms, hr_std_bpm, hr_min_nn_ms.
    """
    try:
        start_ts, sfreq, arr = _read_e4_csv(data)
    except Exception as e:
        logger.error("[E4 HR] [%s] Failed to parse CSV: %s", entity_id, e)
        return None

    if arr.ndim != 1 or len(arr) == 0:
        logger.error("[E4 HR] [%s] Unexpected array shape: %s", entity_id, arr.shape)
        return None
    if sfreq <= 0:
        logger.error("[E4 HR] [%s] Invalid sfreq=%.3f", entity_id, sfreq)
        return None

    abs_times = start_ts + np.arange(len(arr)) / sfreq

    if debate_start_ms is not None and debate_end_ms is not None:
        trim_start_s = debate_start_ms / 1000.0 - window_size_s
        mask = (abs_times >= trim_start_s) & (abs_times <= debate_end_ms / 1000.0)
        rel_times = abs_times[mask] - debate_start_ms / 1000.0
        hr_vals = arr[mask]
        total_s = (debate_end_ms - debate_start_ms) / 1000.0
    else:
        rel_times = abs_times - start_ts
        hr_vals = arr
        total_s = rel_times[-1] + 1.0 / sfreq if len(arr) > 0 else 0.0

    if len(hr_vals) == 0:
        logger.warning("[E4 HR] [%s] No samples in debate window", entity_id)
        return None

    rows = _sliding_windows(rel_times, hr_vals, total_s, window_size_s, step_s, entity_id, _hr_window_features)
    return pd.DataFrame(rows) if rows else None


def extract_polar_hr_features(
    data: bytes,
    entity_id: str,
    window_size_s: float = 10.0,
    step_s: float = 0.3,
    debate_start_ms: Optional[float] = None,
    debate_end_ms: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Extract HR features in sliding windows from a Polar HR CSV (timestamp ms + value BPM).

    Features per window: hr_mean_bpm, hr_mean_nn_ms, hr_std_bpm, hr_min_nn_ms.
    """
    try:
        df = pd.read_csv(io.BytesIO(data))
        if "timestamp" not in df.columns or "value" not in df.columns:
            logger.error("[Polar HR] [%s] Missing timestamp/value columns", entity_id)
            return None
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["timestamp", "value"])
        if df.empty:
            return None

        if debate_start_ms is not None and debate_end_ms is not None:
            trim_start_ms = debate_start_ms - window_size_s * 1000.0
            df = df[(df["timestamp"] >= trim_start_ms) & (df["timestamp"] <= debate_end_ms)].copy()
            rel_times = (df["timestamp"].values - debate_start_ms) / 1000.0
            total_s = (debate_end_ms - debate_start_ms) / 1000.0
        else:
            rel_times = (df["timestamp"].values - df["timestamp"].iloc[0]) / 1000.0
            total_s = rel_times[-1] + 1.0 if len(rel_times) > 0 else 0.0

        if len(df) == 0:
            logger.warning("[Polar HR] [%s] No samples in debate window", entity_id)
            return None

        rows = _sliding_windows(
            rel_times, df["value"].values, total_s, window_size_s, step_s, entity_id, _hr_window_features,
        )
        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        logger.error("[Polar HR] [%s] Extraction failed: %s", entity_id, e)
        return None


# ── IBI ───────────────────────────────────────────────────────────────────────

def _ibi_window_features(ibi_ms: np.ndarray) -> Dict[str, float]:
    """Compute IBI/HRV features for one window from NN intervals in milliseconds."""
    nan = float("nan")
    result: Dict[str, float] = {
        "ibi_sdnn_ms": nan,
        "ibi_rmssd_ms": nan,
        "ibi_median_nn_ms": nan,
        "ibi_mad_nn_ms": nan,
        "ibi_prc20_nn_ms": nan,
        "ibi_pnn50": nan,
    }
    n = len(ibi_ms)
    if n == 0:
        return result
    median = float(np.median(ibi_ms))
    result["ibi_median_nn_ms"] = median
    result["ibi_mad_nn_ms"] = float(np.median(np.abs(ibi_ms - median)))
    result["ibi_prc20_nn_ms"] = float(np.percentile(ibi_ms, 20))
    if n < 2:
        return result
    result["ibi_sdnn_ms"] = float(np.std(ibi_ms, ddof=1))
    diffs = np.diff(ibi_ms)
    result["ibi_rmssd_ms"] = float(np.sqrt(np.mean(diffs ** 2)))
    result["ibi_pnn50"] = float(np.sum(np.abs(diffs) > 50.0) / len(diffs))
    return result


def extract_ibi_features(
    data: bytes,
    entity_id: str,
    window_size_s: float = 10.0,
    step_s: float = 0.3,
    debate_start_ms: Optional[float] = None,
    debate_end_ms: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Extract IBI/HRV features in sliding windows from an Empatica E4 IBI CSV.

    Silver format: 'timestamp' (Unix ms of each beat), 'value' (IBI in ms).
    Features per window: SDNN, RMSSD, MedianNN, MadNN, Prc20NN, pNN50.
    """
    try:
        df = pd.read_csv(io.BytesIO(data))
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        logger.error("[IBI] [%s] Failed to parse IBI CSV: %s", entity_id, e)
        return None

    if df.empty:
        logger.error("[IBI] [%s] No valid IBI rows", entity_id)
        return None

    abs_times_s = df["timestamp"].values / 1000.0
    ibi_vals_ms = df["value"].values  # already in ms in silver format

    if debate_start_ms is not None and debate_end_ms is not None:
        trim_start_s = debate_start_ms / 1000.0 - window_size_s
        mask = (abs_times_s >= trim_start_s) & (abs_times_s <= debate_end_ms / 1000.0)
        rel_times = abs_times_s[mask] - debate_start_ms / 1000.0
        ibi_vals_ms = ibi_vals_ms[mask]
        total_s = (debate_end_ms - debate_start_ms) / 1000.0
    else:
        rel_times = abs_times_s - abs_times_s[0]
        total_s = (rel_times[-1] + ibi_vals_ms[-1] / 1000.0) if len(rel_times) > 0 else 0.0

    if len(ibi_vals_ms) == 0:
        logger.warning("[IBI] [%s] No IBI events in debate window", entity_id)
        return None

    rows = _sliding_windows(rel_times, ibi_vals_ms, total_s, window_size_s, step_s, entity_id, _ibi_window_features)
    return pd.DataFrame(rows) if rows else None
