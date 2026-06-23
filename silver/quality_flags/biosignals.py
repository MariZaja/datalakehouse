import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .signal_readers import read_csv_signal

logger = logging.getLogger("silver_quality_flags")

_BRAINWAVE_CHANNELS = [
    "delta", "lowAlpha", "highAlpha", "lowBeta",
    "highBeta", "lowGamma", "middleGamma", "theta",
]

_BW_QUALITY_ORDER = {"GOOD": 0, "NOISY": 1, "BAD": 2}
_BW_PROBLEM_ORDER = {"NONE": 0, "ARTIFACT": 1, "CLIPPING": 2, "FLAT": 3}


# ── ACC ────────────────────────────────────────────────────────────────────────

def _acc_quality_problem(
    completeness: float,
    out_ratio: float,
    l_max: int,
    is_flat: bool,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    if completeness >= 0.95:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.80:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if out_ratio == 0.0:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if l_max <= 1:
        results.append(("GOOD", "NONE"))
    elif l_max == 2:
        results.append(("NOISY", "CLIPPING"))
    else:
        results.append(("BAD", "CLIPPING"))

    if is_flat:
        results.append(("BAD", "FLAT"))

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


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
    total_windows = math.floor(debate_duration_s / window_size_s)

    def _missing_row(wid: int, completeness: float = 0.0) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "completeness": completeness,
            "out_ratio": float("nan"),
            "clip_ratio": float("nan"),
            "l_max": 0,
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
    values = sig["values"]
    valid_mask = sig["valid_mask"]

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        n_actual = int(in_window.sum())
        xyz = values[in_window]

        finite_mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite_mask]
        n_finite = len(xyz)

        completeness = round(min(1.0, n_actual / expected_per_window), 4) if expected_per_window > 0 else 0.0

        if n_finite == 0:
            rows.append(_missing_row(wid, completeness))
            continue

        out_of_range = ((xyz < -128) | (xyz > 128)).any(axis=1)
        out_ratio = round(float(out_of_range.sum() / n_finite), 4)

        clipped_any = (np.abs(xyz) >= 126).any(axis=1)
        clip_ratio = round(float(clipped_any.sum() / n_finite), 4)
        if clipped_any.any():
            diff = np.diff(np.concatenate([[0], clipped_any.astype(int), [0]]))
            l_max = int((np.where(diff == -1)[0] - np.where(diff == 1)[0]).max())
        else:
            l_max = 0

        sigma = xyz.std(axis=0)
        magnitude = np.sqrt((xyz ** 2).sum(axis=1))
        M = float(magnitude.mean())
        is_flat = bool(
            (sigma[0] < 0.5) and (sigma[1] < 0.5) and (sigma[2] < 0.5)
            and abs(M - 64) > 10
        )

        M_mean = round(M, 4)
        M = round(M, 4)

        quality_flag, problem_flag = _acc_quality_problem(
            completeness, out_ratio, l_max, is_flat
        )

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "out_ratio": out_ratio,
            "clip_ratio": clip_ratio,
            "l_max": l_max,
            "FLAT": is_flat,
            "M": M,
            "M_mean": M_mean,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


# ── BVP ────────────────────────────────────────────────────────────────────────

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
    l_max: int,
    kurtosis: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    if completeness >= 0.95:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.80:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if amplitude < 5:
        results.append(("BAD", "FLAT" if std_val < 2 else "ARTIFACT"))
    elif amplitude <= 400:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if l_max <= 1:
        results.append(("GOOD", "NONE"))
    elif l_max == 2:
        results.append(("NOISY", "CLIPPING"))
    else:
        results.append(("BAD", "CLIPPING"))

    if not np.isnan(kurtosis):
        if kurtosis <= 3.5:
            results.append(("GOOD", "NONE"))
        else:
            results.append(("NOISY", "ARTIFACT"))

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


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
    total_windows = math.floor(debate_duration_s / window_size_s)

    def _missing_row(wid: int, completeness: float = 0.0) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "completeness": completeness,
            "amplitude": float("nan"),
            "std": float("nan"),
            "clip_ratio": float("nan"),
            "l_max": 0,
            "kurtosis": float("nan"),
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
        clipped = x >= p999
        clip_ratio = round(float(clipped.sum() / n_actual), 4)
        if clipped.any():
            diff = np.diff(np.concatenate([[0], clipped.astype(int), [0]]))
            l_max = int((np.where(diff == -1)[0] - np.where(diff == 1)[0]).max())
        else:
            l_max = 0

        sigma = float(np.std(x))
        kurtosis = round(float(np.mean((x - np.mean(x)) ** 4) / sigma ** 4), 4) if sigma > 0 else float("nan")

        quality_flag, problem_flag = _bvp_quality_problem(
            completeness, amplitude, std_val, l_max, kurtosis
        )

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "amplitude": amplitude,
            "std": std_val,
            "clip_ratio": clip_ratio,
            "l_max": l_max,
            "kurtosis": kurtosis,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


# ── EDA ────────────────────────────────────────────────────────────────────────

def _eda_quality_problem(
    completeness: float,
    out_ratio: float,
    sigma_window: float,
    max_jump: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Completeness of 30s analytics window (120 samples expected)
    if completeness >= 0.95:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.80:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 2. Physical range [0.05, 60] μS
    if out_ratio < 0.05:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 3. Flat line — std of 30s window
    if not np.isnan(sigma_window):
        if sigma_window >= 0.005:
            results.append(("GOOD", "NONE"))
        elif sigma_window >= 0.002:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

    # 4. Max jump between consecutive samples (transient artifact)
    if not np.isnan(max_jump):
        if max_jump <= 5.0:
            results.append(("GOOD", "NONE"))
        elif max_jump <= 10.0:
            results.append(("NOISY", "ARTIFACT"))
        else:
            results.append(("BAD", "ARTIFACT"))

    if not results:
        return "BAD", "ARTIFACT"

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)
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
    window_size_s = float(qf_cfg.get("window_size_s", 0.3))
    analytics_window_s = 30.0
    expected_per_analytics = 120  # 4 Hz × 30s
    use_pre_start = bool(qf_cfg.get("use_pre_start_as_warmup", True))
    total_windows = math.floor(debate_duration_s / window_size_s)

    def _missing_row(wid: int, completeness: float = 0.0) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "completeness": completeness,
            "out_ratio": float("nan"),
            "sigma_window": float("nan"),
            "max_jump": float("nan"),
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

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        analytics_start = w_end - analytics_window_s
        is_warmup = analytics_start < 0.0

        eff_start = analytics_start if use_pre_start else max(0.0, analytics_start)
        in_analytics = valid_mask & (rel_ts >= eff_start) & (rel_ts < w_end)
        n_actual = int(in_analytics.sum())
        x = values[in_analytics]

        completeness = round(min(1.0, n_actual / expected_per_analytics), 4)

        if n_actual == 0:
            rows.append(_missing_row(wid, completeness))
            continue

        out_ratio = round(float(np.sum((x < 0.05) | (x > 60)) / n_actual), 4)

        sigma_window = round(float(np.std(x)), 6) if n_actual >= 2 else float("nan")

        if n_actual >= 2:
            max_jump = round(float(np.abs(np.diff(x)).max()), 6)
        else:
            max_jump = float("nan")

        if is_warmup and not use_pre_start:
            quality_flag = "NOISY"
            problem_flag = "NONE"
        else:
            quality_flag, problem_flag = _eda_quality_problem(
                completeness, out_ratio, sigma_window, max_jump
            )

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "out_ratio": out_ratio,
            "sigma_window": sigma_window,
            "max_jump": max_jump,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


# ── HR ─────────────────────────────────────────────────────────────────────────

def _hr_quality_problem(
    completeness: float,
    hr_mean: float,
    hr_std: float,
    roc_max: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    if completeness >= 0.9:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.7:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if not np.isnan(hr_mean):
        if 40 <= hr_mean <= 180:
            results.append(("GOOD", "NONE"))
        elif (30 <= hr_mean < 40) or (180 < hr_mean <= 220):
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    if not np.isnan(hr_std):
        if hr_std >= 0.5:
            results.append(("GOOD", "NONE"))
        elif hr_std >= 0.1:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

    if not np.isnan(roc_max):
        if roc_max <= 3:
            results.append(("GOOD", "NONE"))
        elif roc_max <= 6:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    if not results:
        return "BAD", "ARTIFACT"

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


def process_hr(
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
    analytics_window_s = 10.0
    use_pre_start = bool(qf_cfg.get("use_pre_start_as_warmup", True))
    total_windows = math.floor(debate_duration_s / window_size_s)

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
            "completeness": 0.0,
            "hr_mean": float("nan"),
            "hr_std": float("nan"),
            "hr_ratio_up": float("nan"),
            "hr_ratio_down": float("nan"),
            "RoC_max": float("nan"),
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
            "annotation": "",
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

    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        analytics_start = w_end - analytics_window_s
        is_warmup = analytics_start < 0.0

        eff_start = analytics_start if use_pre_start else max(0.0, analytics_start)
        in_analytics = valid_mask & (rel_ts >= eff_start) & (rel_ts < w_end)
        x_win = values[in_analytics]
        n_actual = len(x_win)
        completeness = round(min(1.0, n_actual / analytics_window_s), 4)

        if n_actual == 0:
            rows.append(_missing_row(wid))
            continue

        hr_mean = round(float(np.mean(x_win)), 4)
        hr_std = round(float(np.std(x_win)), 6) if n_actual >= 2 else float("nan")

        if n_actual >= 2:
            diffs = np.diff(x_win)
            hr_ratio_up = round(float(np.percentile(diffs, 95)), 4)
            hr_ratio_down = round(float(np.percentile(diffs, 5)), 4)
            roc_max = round(float(max(abs(hr_ratio_up), abs(hr_ratio_down))), 4)
        else:
            hr_ratio_up = float("nan")
            hr_ratio_down = float("nan")
            roc_max = float("nan")

        if is_warmup and not use_pre_start:
            quality_flag = "NOISY"
            problem_flag = "NONE"
            annotation = "insufficient_warmup"
        else:
            quality_flag, problem_flag = _hr_quality_problem(completeness, hr_mean, hr_std, roc_max)
            annotation = ""

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "hr_mean": hr_mean,
            "hr_std": hr_std,
            "hr_ratio_up": hr_ratio_up,
            "hr_ratio_down": hr_ratio_down,
            "RoC_max": roc_max,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
            "annotation": annotation,
            "bvp_quality_flag": bvp_flags.get(wid),
        })

    return pd.DataFrame(rows) if rows else None


# ── IBI ────────────────────────────────────────────────────────────────────────

def _ibi_quality_problem(
    n_ibi: int,
    out_ratio: float,
    g_max: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # Analysis 1: window completeness
    if n_ibi >= 8:
        results.append(("GOOD", "NONE"))
    elif n_ibi >= 5:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # Analysis 2: physical range [0.27s, 2.0s]
    if not np.isnan(out_ratio):
        if out_ratio == 0.0:
            results.append(("GOOD", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

    # Analysis 3: max gap between consecutive timestamps in window
    if not np.isnan(g_max):
        if g_max <= 2.0:
            results.append(("GOOD", "NONE"))
        elif g_max <= 4.0:
            results.append(("NOISY", "ARTIFACT"))
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
    window_size_s = float(qf_cfg.get("window_size_s", 0.3))
    analytics_window_s = 10.0
    total_windows = math.floor(debate_duration_s / window_size_s)

    def _missing_row(wid: int) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "n_ibi": 0,
            "out_ratio": float("nan"),
            "g_max_s": float("nan"),
            "gap_before_s": 0.0,
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

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        analytics_start = w_end - analytics_window_s

        # IBIs in 10s rolling window [t-10s, t]
        in_window = (valid_ts >= analytics_start) & (valid_ts < w_end)
        n_ibi = int(in_window.sum())
        x = valid_vals[in_window]
        ts_in = valid_ts[in_window]

        # Analysis 2: physical range [0.27s, 2.0s] stored in same units as raw values
        out_ratio = (
            round(float(np.sum((x < 270.0) | (x > 2000.0)) / n_ibi), 4)
            if n_ibi >= 1 else float("nan")
        )

        # Analysis 3: max gap between consecutive timestamps inside window
        if n_ibi >= 2:
            g_max = round(float(np.diff(ts_in).max()), 4)
        else:
            g_max = float("nan")

        # Analysis 4: gap_before_s — informational only, does not affect flags
        before_window = valid_ts[valid_ts < analytics_start]
        if n_ibi >= 1 and len(before_window) > 0:
            gap_before_s = round(float(ts_in[0] - before_window[-1]), 4)
        else:
            gap_before_s = 0.0

        quality_flag, problem_flag = _ibi_quality_problem(n_ibi, out_ratio, g_max)

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "n_ibi": n_ibi,
            "out_ratio": out_ratio,
            "g_max_s": g_max,
            "gap_before_s": gap_before_s,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


# ── TEMP ───────────────────────────────────────────────────────────────────────

def _temp_quality_problem(
    completeness: float,
    out_ratio: float,
    mean_temp: float,
    sigma_window: float,
    max_jump: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # 1. Completeness of 30s analytics window (120 samples expected)
    if completeness >= 0.95:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.80:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 2. Physical range [25, 36] °C
    if out_ratio <= 0.05:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # 3. Suspect range (sensor off skin); values outside [20, 42] already caught by analysis 2
    if not np.isnan(mean_temp):
        if 28.0 <= mean_temp <= 36.0:
            results.append(("GOOD", "NONE"))
        elif 25.0 <= mean_temp < 28.0:
            results.append(("NOISY", "NONE"))

    # 4. Flat line — std of 30s analytics window
    if not np.isnan(sigma_window):
        if sigma_window >= 0.01:
            results.append(("GOOD", "NONE"))
        elif sigma_window >= 0.005:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

    # 5. Max jump between consecutive samples (transient artifact)
    if not np.isnan(max_jump):
        if max_jump <= 0.2:
            results.append(("GOOD", "NONE"))
        elif max_jump <= 0.5:
            results.append(("NOISY", "ARTIFACT"))
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
    window_size_s = float(qf_cfg.get("window_size_s", 0.3))
    analytics_window_s = 30.0
    expected_per_analytics = 120  # 4 Hz × 30s
    use_pre_start = bool(qf_cfg.get("use_pre_start_as_warmup", True))
    total_windows = math.floor(debate_duration_s / window_size_s)

    def _missing_row(wid: int, completeness: float = 0.0) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "completeness": completeness,
            "out_ratio": float("nan"),
            "mean_temp": float("nan"),
            "sigma_window": float("nan"),
            "max_jump": float("nan"),
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

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        analytics_start = w_end - analytics_window_s
        is_warmup = analytics_start < 0.0

        eff_start = analytics_start if use_pre_start else max(0.0, analytics_start)
        in_analytics = valid_mask & (rel_ts >= eff_start) & (rel_ts < w_end)
        n_actual = int(in_analytics.sum())
        x = values[in_analytics]

        completeness = round(min(1.0, n_actual / expected_per_analytics), 4)

        if n_actual == 0:
            rows.append(_missing_row(wid, completeness))
            continue

        out_ratio = round(float(np.sum((x < 25) | (x > 36)) / n_actual), 4)
        mean_temp = round(float(np.mean(x)), 4)

        sigma_window = round(float(np.std(x)), 6) if n_actual >= 2 else float("nan")

        if n_actual >= 2:
            max_jump = round(float(np.abs(np.diff(x)).max()), 6)
        else:
            max_jump = float("nan")

        if is_warmup and not use_pre_start:
            quality_flag = "NOISY"
            problem_flag = "NONE"
        else:
            quality_flag, problem_flag = _temp_quality_problem(
                completeness, out_ratio, mean_temp, sigma_window, max_jump
            )

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "completeness": completeness,
            "out_ratio": out_ratio,
            "mean_temp": mean_temp,
            "sigma_window": sigma_window,
            "max_jump": max_jump,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None


# ── BrainWave ──────────────────────────────────────────────────────────────────

def _bw_quality_from_analyses(zero: bool, flat: bool, spike: bool) -> Tuple[str, str]:
    if not any([zero, flat, spike]):
        return "GOOD", "NONE"
    results = []
    if zero:
        results.append(("BAD", "ARTIFACT"))
    if flat:
        results.append(("BAD", "FLAT"))
    if spike:
        results.append(("NOISY", "ARTIFACT"))
    quality = max((r[0] for r in results), key=lambda q: _BW_QUALITY_ORDER[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "ARTIFACT", "NONE") if p in problem_set)
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
    output_window_s = float(qf_cfg.get("window_size_s", 0.3))
    compute_window_s = max(1.0, output_window_s)  # min 1 s (SDK rate); grows with window_size_s
    total_compute = math.floor(debate_duration_s / compute_window_s)
    total_output = math.floor(debate_duration_s / output_window_s)

    def _bad_output_row(wid: int) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * output_window_s, 3),
            "window_end_s": round((wid + 1) * output_window_s, 3),
            "zero": False,
            "flat": False,
            "spike": False,
            "power": float("nan"),
            "power_ratio": float("nan"),
            "quality_flag": "BAD",
            "problem_flag": "ARTIFACT",
        }

    if file_data is None:
        return pd.DataFrame([_bad_output_row(w) for w in range(total_output)])

    sig = read_csv_signal(
        file_data, timestamp_col, timestamp_unit_ms,
        ref_start_s=debate_start_s, value_cols=channels,
    )
    if sig is None:
        return pd.DataFrame([_bad_output_row(w) for w in range(total_output)])

    rel_ts = sig["rel_ts"]
    values = sig["values"]
    valid_mask = sig["valid_mask"]

    # Last pre-start sample provides P(w-1) for the first session window
    pre_mask = valid_mask & (rel_ts < 0.0)
    pre_values = values[pre_mask]
    p_prev: Optional[float] = float(np.sum(pre_values[-1])) if len(pre_values) > 0 else None

    # Per-compute-window results: zero, flat, spike, power, power_ratio, quality, problem
    compute_rows: List[Dict[str, Any]] = []
    prev_vals: Optional[np.ndarray] = None
    prev_prev_vals: Optional[np.ndarray] = None

    for k in range(total_compute):
        w_start = k * compute_window_s
        w_end = w_start + compute_window_s
        in_window = valid_mask & (rel_ts >= w_start) & (rel_ts < w_end)
        x = values[in_window]

        if len(x) == 0:
            compute_rows.append({
                "zero": False, "flat": False, "spike": False,
                "power": float("nan"), "power_ratio": float("nan"),
                "quality_flag": "BAD", "problem_flag": "ARTIFACT",
            })
            prev_prev_vals = prev_vals
            prev_vals = None
            p_prev = None
            continue

        cur = x[-1]  # last sample; fallback for cross-window flat when n == 1
        n_in_window = len(x)

        # Analysis 1: zero — any band == 0 in any sample in window
        zero = bool(np.any(x == 0))

        # Analysis 2: flat — within-window (n≥2) or cross-window fallback (n==1)
        if n_in_window >= 2:
            flat = bool(np.any(np.std(x, axis=0) == 0.0))
        elif prev_vals is not None and prev_prev_vals is not None:
            flat = bool(np.any((cur == prev_vals) & (prev_vals == prev_prev_vals)))
        else:
            flat = False

        # Analysis 3: power spike — mean power per sample, ratio > 2
        p_cur = (float(np.mean(np.sum(x, axis=1))) if n_in_window >= 2
                 else float(np.sum(cur)))
        if p_prev is not None and p_prev > 0:
            power_ratio = p_cur / p_prev
            spike = power_ratio > 2.0
        else:
            power_ratio = float("nan")
            spike = False

        quality_flag, problem_flag = _bw_quality_from_analyses(zero, flat, spike)

        compute_rows.append({
            "zero": zero, "flat": flat, "spike": spike,
            "power": round(p_cur, 4),
            "power_ratio": round(power_ratio, 4) if not math.isnan(power_ratio) else float("nan"),
            "quality_flag": quality_flag, "problem_flag": problem_flag,
        })

        prev_prev_vals = prev_vals
        prev_vals = cur.copy()
        p_prev = p_cur

    # Map each output window to its enclosing compute window by center
    output_rows: List[Dict[str, Any]] = []
    for wid in range(total_output):
        center = wid * output_window_s + output_window_s / 2.0
        k = min(int(center), total_compute - 1)
        cr = compute_rows[k] if k < len(compute_rows) else None

        if cr is None:
            output_rows.append(_bad_output_row(wid))
        else:
            output_rows.append({
                "window_id": wid,
                "window_start_s": round(wid * output_window_s, 3),
                "window_end_s": round((wid + 1) * output_window_s, 3),
                "zero": cr["zero"],
                "flat": cr["flat"],
                "spike": cr["spike"],
                "power": cr["power"],
                "power_ratio": cr["power_ratio"],
                "quality_flag": cr["quality_flag"],
                "problem_flag": cr["problem_flag"],
            })

    return pd.DataFrame(output_rows) if output_rows else None


# ── Attention / Meditation (shared) ────────────────────────────────────────────

_ATT_MED_ANALYTICS_WINDOW_S = 10.0  # minimum analytics window; grows with window_size_s


def _att_med_quality_problem(
    n: int,
    zeros: bool,
    max_delta: float,
    c_clip: int,
    run_max: int,
    expected_n: int,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    # Analysis 1 — sample count vs expected
    if n >= expected_n:
        results.append(("GOOD", "NONE"))
    elif n >= max(1, round(0.7 * expected_n)):
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    # Analysis 2 — zero detection
    results.append(("BAD", "ARTIFACT") if zeros else ("GOOD", "NONE"))

    # Analysis 3 — max consecutive delta > 40
    if not np.isnan(max_delta):
        results.append(("BAD", "ARTIFACT") if max_delta > 40 else ("GOOD", "NONE"))

    # Analysis 4 — clipping at 100
    if c_clip >= expected_n:
        results.append(("BAD", "CLIPPING"))
    elif c_clip >= max(1, round(0.8 * expected_n)):
        results.append(("NOISY", "CLIPPING"))
    else:
        results.append(("GOOD", "NONE"))

    # Analysis 5 — flat line (max run of identical consecutive values)
    if run_max >= expected_n:
        results.append(("BAD", "FLAT"))
    elif run_max >= max(1, round(0.9 * expected_n)):
        results.append(("NOISY", "FLAT"))
    else:
        results.append(("GOOD", "NONE"))

    quality_flag = max((r[0] for r in results), key=lambda q: _q[q])
    problem_set = {r[1] for r in results}
    problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)
    return quality_flag, problem_flag


def process_att_med(
    file_data: Optional[bytes],
    entity_id: str,
    debate_start_s: float,
    debate_duration_s: float,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    sample_rate_hz: Optional[float],
    qf_cfg: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    window_size_s = float(qf_cfg.get("window_size_s", 0.3))
    analytics_window_s = max(_ATT_MED_ANALYTICS_WINDOW_S, window_size_s)
    expected_n = int(round(analytics_window_s))  # 1 Hz
    total_windows = math.floor(debate_duration_s / window_size_s)

    def _missing_row(wid: int) -> Dict[str, Any]:
        return {
            "window_id": wid,
            "window_start_s": round(wid * window_size_s, 3),
            "window_end_s": round((wid + 1) * window_size_s, 3),
            "n_window": 0,
            "zeros": False,
            "max_delta": float("nan"),
            "c_clip": 0,
            "run_max": 0,
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

    rows: List[Dict[str, Any]] = []
    for wid in range(total_windows):
        w_start = wid * window_size_s
        w_end = w_start + window_size_s
        analytics_start = w_end - analytics_window_s

        in_window = valid_mask & (rel_ts >= analytics_start) & (rel_ts < w_end)
        x = values[in_window]
        n = len(x)

        zeros = bool(np.any(x == 0.0)) if n > 0 else False

        max_delta = round(float(np.abs(np.diff(x)).max()), 4) if n >= 2 else float("nan")

        c_clip = int(np.sum(x == 100.0)) if n > 0 else 0

        if n >= 2:
            changes = np.concatenate([[1], x[1:] != x[:-1]])
            runs = np.diff(np.where(np.concatenate([changes, [1]]))[0])
            run_max = int(runs.max())
        elif n == 1:
            run_max = 1
        else:
            run_max = 0

        quality_flag, problem_flag = _att_med_quality_problem(n, zeros, max_delta, c_clip, run_max, expected_n)

        rows.append({
            "window_id": wid,
            "window_start_s": round(w_start, 3),
            "window_end_s": round(w_end, 3),
            "n_window": n,
            "zeros": zeros,
            "max_delta": max_delta,
            "c_clip": c_clip,
            "run_max": run_max,
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return pd.DataFrame(rows) if rows else None
