import io
import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BRAINWAVE_BAND_COLS = [
    "delta", "lowAlpha", "highAlpha", "lowBeta",
    "highBeta", "lowGamma", "middleGamma", "theta",
]

_EEG_MNE_FUNCS = ["pow_freq_bands", "hjorth_mobility", "hjorth_complexity", "app_entropy"]
_EEG_MNE_PARAMS = {
    # delta(0.5–4), theta(4–8), alpha(8–13), beta(13–30), gamma(30–100)
    "pow_freq_bands__freq_bands": np.array([0.5, 4.0, 8.0, 13.0, 30.0, 100.0]),
    "app_entropy__emb": 2,
}


def extract_eeg_mne_features(
    eeg_arr: np.ndarray,
    entity_id: str,
    sfreq: float = 500.0,
    n_instances: int = 200,
    n_timepoints: Optional[int] = None,
    window_size_s: float = 0.3,
    tp_axis: int = 0,
    inst_axis: int = 2,
) -> Optional[pd.DataFrame]:
    """Extract PSD, Hjorth mobility/complexity, and approximate entropy in fixed-length windows.

    Expects eeg_arr shaped (n_timepoints, n_channels, n_instances) per EAV convention.
    Each trial is sliced into window_size_s windows; mne-features are computed per window.
    Returns one row per (trial, window) with entity_id, trial_id, window_id,
    window_start_s, window_end_s and all mne-features columns.
    """
    try:
        from mne_features.feature_extraction import extract_features

        if eeg_arr.ndim != 3:
            logger.warning("[EAV EEG] [%s] Unexpected array ndim=%d", entity_id, eeg_arr.ndim)
            return None

        # (n_tp, n_ch, n_inst) → (n_inst, n_ch, n_tp) as required by mne-features
        data = np.moveaxis(eeg_arr, [inst_axis, tp_axis], [0, 2])
        n_use = min(n_instances, data.shape[0])
        data = data[:n_use].astype(np.float64)
        if n_timepoints is not None:
            data = data[:, :, :n_timepoints]

        window_n = max(1, int(round(sfreq * window_size_s)))
        n_tp = data.shape[2]
        n_windows = math.ceil(n_tp / window_n)

        all_frames: List[pd.DataFrame] = []
        for inst_idx in range(n_use):
            trial = data[inst_idx]  # (n_ch, n_tp)
            for wid in range(1, n_windows + 1):
                start = (wid - 1) * window_n
                end = min(wid * window_n, n_tp)
                chunk = trial[:, start:end][np.newaxis, :, :]  # (1, n_ch, samples)
                try:
                    feat_df = extract_features(
                        chunk, sfreq, _EEG_MNE_FUNCS,
                        funcs_params=_EEG_MNE_PARAMS,
                        return_as_df=True,
                    )
                    feat_df = feat_df.reset_index(drop=True)
                    feat_df.insert(0, "entity_id", entity_id)
                    feat_df.insert(1, "trial_id", inst_idx)
                    feat_df.insert(2, "window_id", wid)
                    feat_df.insert(3, "window_start_s", round(start / sfreq, 3))
                    feat_df.insert(4, "window_end_s", round((wid * window_n) / sfreq, 3))
                    all_frames.append(feat_df)
                except Exception as e:
                    logger.warning(
                        "[EAV EEG] [%s] trial %d window %d: %s", entity_id, inst_idx, wid, e,
                    )

        if not all_frames:
            return None
        return pd.concat(all_frames, ignore_index=True)
    except ImportError:
        logger.error("mne-features not installed — skipping EAV EEG extraction")
        return None
    except Exception as e:
        logger.error("[EAV EEG] [%s] mne-features extraction failed: %s", entity_id, e)
        return None


def extract_kemocon_brainwave(
    data: bytes,
    entity_id: str,
    window_size_s: float = 0.3,
    debate_start_ms: Optional[float] = None,
    debate_end_ms: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Map NeuroSky band-power features from BrainWave.csv to fixed-length windows.

    Signal is at 1 Hz; each second's sample is broadcast to every window whose
    centre falls within that second. Returns one row per window.
    """
    try:
        df = pd.read_csv(io.BytesIO(data))
        available = [c for c in _BRAINWAVE_BAND_COLS if c in df.columns]
        if not available:
            logger.warning("[K-EmoCon EEG] [%s] No band power columns in BrainWave.csv", entity_id)
            return None
        if debate_start_ms is None or debate_end_ms is None or "timestamp" not in df.columns:
            return None

        keep = [c for c in (["timestamp"] + available) if c in df.columns]
        df = df[keep].copy()
        df = df[(df["timestamp"] >= debate_start_ms) & (df["timestamp"] <= debate_end_ms)].copy()
        if df.empty:
            return None

        debate_duration_s = (debate_end_ms - debate_start_ms) / 1000.0
        df["rel_ts_s"] = (df["timestamp"] - debate_start_ms) / 1000.0
        df["sec_idx"] = df["rel_ts_s"].apply(math.floor)
        sec_values: Dict[int, Dict[str, float]] = (
            df.groupby("sec_idx")[available].mean().to_dict("index")
        )

        total_windows = math.ceil(debate_duration_s / window_size_s)
        rows: List[Dict[str, Any]] = []
        for wid in range(1, total_windows + 1):
            center_s = (wid - 1) * window_size_s + window_size_s / 2.0
            sec_idx = math.floor(center_s)
            row: Dict[str, Any] = {
                "entity_id": entity_id,
                "window_id": wid,
                "window_start_s": round((wid - 1) * window_size_s, 3),
                "window_end_s": round(wid * window_size_s, 3),
            }
            band_vals = sec_values.get(sec_idx, {})
            for band in available:
                row[band] = band_vals.get(band, float("nan"))
            rows.append(row)

        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        logger.error("[K-EmoCon EEG] [%s] BrainWave extraction failed: %s", entity_id, e)
        return None


def extract_kemocon_scalar_windowed(
    data: bytes,
    entity_id: str,
    signal_name: str,
    debate_start_ms: Optional[float],
    debate_end_ms: Optional[float],
    window_size_s: float = 10.0,
    step_s: float = 0.3,
) -> Optional[pd.DataFrame]:
    """Extract windowed mean from a 1 Hz scalar signal (Attention / Meditation).

    Slides a window of `window_size_s` seconds with `step_s` step over the debate
    window. The single feature per window is the mean of all samples that fall
    within [window_start, window_end). Windows with no samples are skipped.
    Returns one row per window: entity_id, window_id, window_start_s,
    window_end_s, {signal_name}.
    """
    if debate_start_ms is None or debate_end_ms is None:
        return None
    try:
        df = pd.read_csv(io.BytesIO(data))
        if "timestamp" not in df.columns or "value" not in df.columns:
            logger.warning(
                "[K-EmoCon] [%s] %s.csv missing required columns", entity_id, signal_name,
            )
            return None
        df = df[["timestamp", "value"]].copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[
            (df["timestamp"] >= debate_start_ms) & (df["timestamp"] <= debate_end_ms)
        ].dropna()
        if df.empty:
            return None

        rel_s = ((df["timestamp"] - debate_start_ms) / 1000.0).values
        values = df["value"].values
        debate_duration_s = (debate_end_ms - debate_start_ms) / 1000.0

        rows: List[Dict[str, Any]] = []
        wid = 1
        t_start = 0.0
        while t_start < debate_duration_s:
            t_end = t_start + window_size_s
            mask = (rel_s >= t_start) & (rel_s < t_end)
            finite = values[mask]
            finite = finite[np.isfinite(finite)]
            if finite.size > 0:
                rows.append({
                    "entity_id": entity_id,
                    "window_id": wid,
                    "window_start_s": round(t_start, 3),
                    "window_end_s": round(t_end, 3),
                    signal_name: float(finite.mean()),
                })
            t_start = round(t_start + step_s, 9)
            wid += 1

        return pd.DataFrame(rows) if rows else None
    except Exception as e:
        logger.error(
            "[K-EmoCon] [%s] %s windowed extraction failed: %s", entity_id, signal_name, e,
        )
        return None
