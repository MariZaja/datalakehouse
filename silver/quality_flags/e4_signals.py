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
    clip_ratio: float,
    is_flat: bool,
    M_mean: float,
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

    if clip_ratio <= 0.02:
        results.append(("GOOD", "NONE"))
    elif clip_ratio <= 0.10:
        results.append(("NOISY", "CLIPPING"))
    else:
        results.append(("BAD", "CLIPPING"))

    if is_flat:
        results.append(("BAD", "FLAT"))

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

        per_axis_clip = (np.abs(xyz) >= 126).sum(axis=0) / n_finite
        clip_ratio = round(float(per_axis_clip.max()), 4)

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
    clip_ratio: float,
    peaks_count: int,
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

    if clip_ratio <= 0.02:
        results.append(("GOOD", "NONE"))
    elif clip_ratio <= 0.10:
        results.append(("NOISY", "CLIPPING"))
    else:
        results.append(("BAD", "CLIPPING"))

    if 1 <= peaks_count <= 3:
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


# ── EDA ────────────────────────────────────────────────────────────────────────

def _eda_quality_problem(
    completeness: float,
    out_ratio: float,
    flat_line_std: float,
    delta: float,
    drift: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    if completeness >= 1.0:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.75:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if out_ratio == 0.0:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if not np.isnan(flat_line_std):
        if flat_line_std >= 0.005:
            results.append(("GOOD", "NONE"))
        elif flat_line_std >= 0.001:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

    if not np.isnan(delta):
        if delta <= 0.5:
            results.append(("GOOD", "NONE"))
        elif delta <= 2.0:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

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
    expected_per_window = int(round(sr * window_size_s))
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

        out_ratio = round(float(np.sum((x < 0) | (x > 20)) / len(x)), 4)

        in_roll = valid_mask & (rel_ts >= w_end - 30.0) & (rel_ts < w_end)
        x_roll = values[in_roll]
        flat_line_30 = round(float(np.std(x_roll)), 6) if len(x_roll) >= 2 else float("nan")

        delta = round(float(np.max(x) - np.min(x)), 6)

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


# ── HR ─────────────────────────────────────────────────────────────────────────

def _hr_quality_problem(
    n_window: int,
    x: float,
    roc: float,
    flat_line_30: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    if n_window != 1:
        return "BAD", "ARTIFACT"
    results.append(("GOOD", "NONE"))

    if 40 <= x <= 180:
        results.append(("GOOD", "NONE"))
    elif (30 <= x < 40) or (180 < x <= 220):
        results.append(("NOISY", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if not np.isnan(roc):
        if roc <= 10:
            results.append(("GOOD", "NONE"))
        elif roc <= 20:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

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

        if prev_x is not None and prev_quality_flag != "BAD":
            roc = round(abs(x - prev_x), 4)
        else:
            roc = float("nan")

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


# ── IBI ────────────────────────────────────────────────────────────────────────

def _ibi_quality_problem(
    n_window: int,
    gap: float,
    out_ratio: float,
    jump: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    if n_window >= 1:
        results.append(("GOOD", "NONE"))

    if n_window >= 1:
        results.append(("GOOD", "NONE"))
    elif not np.isnan(gap):
        if gap <= 2.0:
            results.append(("GOOD", "NONE"))
        elif gap <= 5.0:
            results.append(("NOISY", "ARTIFACT"))
        else:
            results.append(("BAD", "ARTIFACT"))

    if n_window >= 1 and not np.isnan(out_ratio):
        if out_ratio == 0.0:
            results.append(("GOOD", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

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


# ── TEMP ───────────────────────────────────────────────────────────────────────

def _temp_quality_problem(
    completeness: float,
    out_ratio: float,
    mean_temp: float,
    flat_line_60: float,
    roc: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    if completeness >= 1.0:
        results.append(("GOOD", "NONE"))
    elif completeness >= 0.75:
        results.append(("NOISY", "ARTIFACT"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if out_ratio == 0.0:
        results.append(("GOOD", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if not np.isnan(mean_temp):
        if 25.0 <= mean_temp <= 40.0:
            results.append(("GOOD", "NONE"))
        elif 20.0 <= mean_temp < 25.0:
            results.append(("NOISY", "NONE"))

    if not np.isnan(flat_line_60):
        if flat_line_60 >= 0.01:
            results.append(("GOOD", "NONE"))
        elif flat_line_60 >= 0.005:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "FLAT"))

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
    expected_per_window = int(round(sr * window_size_s))
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

        out_ratio = round(float(np.sum((x < 20) | (x > 42)) / len(x)), 4)
        mean_temp = round(float(np.mean(x)), 4)

        in_roll = valid_mask & (rel_ts >= w_end - 60.0) & (rel_ts < w_end)
        x_roll = values[in_roll]
        flat_line_60 = round(float(np.std(x_roll)), 6) if len(x_roll) >= 2 else float("nan")

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


# ── BrainWave ──────────────────────────────────────────────────────────────────

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
    values = sig["values"]
    valid_mask = sig["valid_mask"]

    clip_thresholds: List[float] = []
    spike_thresholds: List[float] = []
    flat_masks: List[np.ndarray] = []
    spike_masks: List[np.ndarray] = []

    for ch_idx in range(len(channels)):
        col = values[:, ch_idx]
        finite_valid = col[valid_mask & np.isfinite(col)]

        clip_thr = float(np.percentile(finite_valid, 99.5)) if len(finite_valid) > 0 else np.inf
        clip_thresholds.append(clip_thr)

        diffs = np.abs(np.diff(col))
        valid_pair = valid_mask[:-1] & valid_mask[1:] & np.isfinite(diffs)
        finite_diffs = diffs[valid_pair]
        if len(finite_diffs) >= 4:
            q1, q3 = np.percentile(finite_diffs, [25, 75])
            spike_thr = float(q3 + 1.5 * (q3 - q1))
        else:
            spike_thr = np.inf
        spike_thresholds.append(spike_thr)

        flat_m = np.zeros(len(col), dtype=bool)
        if len(col) >= 3:
            flat_m[2:] = (col[2:] == col[1:-1]) & (col[1:-1] == col[:-2])
        flat_masks.append(flat_m)

        spike_m = np.zeros(len(col), dtype=bool)
        if len(col) >= 2:
            spike_m[1:] = diffs > spike_thr
        spike_masks.append(spike_m)

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
                z = bool(np.any(x == 0))
                f = bool(np.any(flat_masks[ch_idx][in_window]))
                c = bool(np.any(x >= clip_thresholds[ch_idx]))
                s = bool(np.any(spike_masks[ch_idx][in_window]))

            row[f"{ch}_zeros"] = z
            row[f"{ch}_flat"] = f
            row[f"{ch}_clipping"] = c
            row[f"{ch}_spike"] = s

            sig_quality[ch], sig_problem[ch] = _bw_signal_quality_problem(z, f, c, s)

        worst_q = max(sig_quality.values(), key=lambda q: _BW_QUALITY_ORDER[q])
        worst_p = max(sig_problem.values(), key=lambda p: _BW_PROBLEM_ORDER[p])

        flagged = [ch for ch in channels if sig_quality[ch] == worst_q and worst_q != "GOOD"]
        row["quality_flag"] = worst_q
        row["problem_flag"] = worst_p
        row["flag_reason"] = ", ".join(flagged)
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


# ── Attention ──────────────────────────────────────────────────────────────────

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

        zeros = bool(np.any(xs == 0.0))
        low_value = bool(np.any((xs > 0.0) & (xs < 10.0)))

        seq = np.concatenate([[prev_x], xs]) if prev_x is not None else xs
        delta = round(float(np.abs(np.diff(seq)).max()), 4) if len(seq) >= 2 else float("nan")

        for x in xs:
            cur_run_100 = cur_run_100 + 1 if x == 100.0 else 0

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

        results: List[Tuple[str, str]] = [("GOOD", "NONE")]
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


# ── Meditation ─────────────────────────────────────────────────────────────────

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

        zeros = bool(np.any(xs == 0.0))
        low_value = bool(np.any((xs > 0.0) & (xs < 5.0)))

        seq = np.concatenate([[prev_x], xs]) if prev_x is not None else xs
        delta = round(float(np.abs(np.diff(seq)).max()), 4) if len(seq) >= 2 else float("nan")

        for x in xs:
            cur_run_100 = cur_run_100 + 1 if x == 100.0 else 0

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

        results: List[Tuple[str, str]] = [("GOOD", "NONE")]
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


# ── Polar HR ───────────────────────────────────────────────────────────────────

def _polar_hr_quality_problem(
    n_window: int,
    x: float,
    roc: float,
    flat_line_30: float,
) -> Tuple[str, str]:
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}
    results: List[Tuple[str, str]] = []

    results.append(("GOOD", "NONE"))

    if 40 <= x <= 180:
        results.append(("GOOD", "NONE"))
    elif (30 <= x < 40) or (180 < x <= 220):
        results.append(("NOISY", "NONE"))
    else:
        results.append(("BAD", "ARTIFACT"))

    if not np.isnan(roc):
        if roc <= 20:
            results.append(("GOOD", "NONE"))
        elif roc <= 35:
            results.append(("NOISY", "NONE"))
        else:
            results.append(("BAD", "ARTIFACT"))

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

        if prev_x is not None and prev_quality_flag != "BAD":
            roc = round(abs(x - prev_x), 4)
        else:
            roc = float("nan")

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
