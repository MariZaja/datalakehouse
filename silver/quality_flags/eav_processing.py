import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .minio_utils import download_object, upload_csv
from .signal_readers import load_mat_eeg, read_video_signal
from .audio_video_qc import _process_single_wav, _compute_video_windows

logger = logging.getLogger("silver_quality_flags")


def _max_run_length(x: np.ndarray) -> int:
    """Longest consecutive run of identical values in x."""
    if len(x) == 0:
        return 0
    changes = np.concatenate([[True], x[1:] != x[:-1], [True]])
    run_lengths = np.diff(np.where(changes)[0])
    return int(run_lengths.max())


def _compute_rNSR(x: np.ndarray, fs: float) -> float:
    """Compute P(40-250 Hz)/P(1-40 Hz) on a HP-filtered (>1 Hz) copy of x."""
    if len(x) < max(10, int(fs * 0.1)):
        return float("nan")
    try:
        from scipy import signal as sp_signal
        nyq = fs / 2.0
        sos = sp_signal.butter(4, 1.0 / nyq, btype="high", output="sos")
        x_hp = sp_signal.sosfiltfilt(sos, x)
        freqs, psd = sp_signal.welch(x_hp, fs=fs, nperseg=min(len(x_hp), int(fs)))
        P_noise = float(np.mean(psd[(freqs >= 40) & (freqs <= 250)]))
        P_signal = float(np.mean(psd[(freqs >= 1) & (freqs < 40)]))
        if P_signal == 0:
            return float("nan")
        return round(P_noise / P_signal, 6)
    except Exception:
        return float("nan")


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
    window_id, channel_id, samples, std, max_run, peak_to_peak,
    rNSR, rNSR_zscore, quality_flag, problem_flag.

    Analyses (applied per channel per window):
      1. Incomplete window — n_samples < 80% expected → BAD/ARTIFACT; < 100% → NOISY/ARTIFACT
      2. Flat line         — nan_present OR std < 0.5 µV → BAD/FLAT
      3. High-freq noise   — rNSR > 0.5 AND |z| > 3 → BAD/ARTIFACT
                             rNSR > 0.5 OR  |z| > 3 → NOISY/ARTIFACT
                             (z-score per session: robust 1.4826×MAD normalization)
      4. Clipping          — max_run ≥ 5 → BAD/CLIPPING
      5. Amplitude         — peak_to_peak > 500 µV → BAD/ARTIFACT

    Priority: problem FLAT > CLIPPING > ARTIFACT > NONE; quality BAD > NOISY > GOOD.
    rNSR z-score is computed across all windows × channels of the same session.
    """
    _q = {"GOOD": 0, "NOISY": 1, "BAD": 2}

    if eeg_arr is None or eeg_arr.ndim < 2:
        return

    declared_hz = float(eeg_sig_cfg.get("declared_hz", 500.0))
    n_instances = int(eeg_sig_cfg.get("expected_instances", 200))
    window_size_s = float(qf_cfg.get("window_size_s", 1.0))
    expected_per_window = int(round(declared_hz * window_size_s))
    incomplete_bad_thr = int(round(0.8 * expected_per_window))

    n_instances_actual = eeg_arr.shape[eeg_inst_axis] if eeg_arr.ndim == 3 else 1

    for inst_idx in range(min(n_instances, n_instances_actual)):
        session_id = f"{inst_idx:03d}"

        if (entity_id, session_id, "eeg") in miss_skip:
            logger.info("[EAV] [%s] Session %s → excluded (total_missing)", entity_id, session_id)
            continue

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
        total_windows = math.floor(n_timepoints / expected_per_window)

        # Pass 1 — collect per-(window, channel) metrics; gather rNSR values for z-score
        intermediate: List[Dict[str, Any]] = []
        rNSR_all: List[float] = []

        for wid in range(total_windows):
            start_idx = wid * expected_per_window
            end_idx = min((wid + 1) * expected_per_window, n_timepoints)
            window_data = eeg_slice[start_idx:end_idx, :]

            for ch_idx in range(n_channels):
                x_raw = window_data[:, ch_idx]
                nan_present = bool(not np.all(np.isfinite(x_raw)))
                x = x_raw[np.isfinite(x_raw)]
                n_samples = len(x)

                if n_samples < incomplete_bad_thr:
                    a1 = ("BAD", "ARTIFACT")
                elif n_samples < expected_per_window:
                    a1 = ("NOISY", "ARTIFACT")
                else:
                    a1 = ("GOOD", "NONE")

                if n_samples == 0:
                    intermediate.append({
                        "window_id": wid, "channel_id": ch_idx,
                        "nan_present": nan_present, "samples": 0,
                        "std": float("nan"), "max_run": 0,
                        "peak_to_peak": float("nan"), "rNSR": float("nan"),
                        "a1": a1,
                    })
                    continue

                std_val = round(float(np.std(x)), 6)
                max_run = _max_run_length(x)
                peak_to_peak = round(float(np.max(x) - np.min(x)), 4)
                rNSR = _compute_rNSR(x, declared_hz)
                if not np.isnan(rNSR):
                    rNSR_all.append(rNSR)

                intermediate.append({
                    "window_id": wid, "channel_id": ch_idx,
                    "nan_present": nan_present, "samples": n_samples,
                    "std": std_val, "max_run": max_run,
                    "peak_to_peak": peak_to_peak, "rNSR": rNSR,
                    "a1": a1,
                })

        # rNSR z-score parameters — computed per session across all windows × channels
        if len(rNSR_all) >= 2:
            arr = np.array(rNSR_all)
            median_rNSR = float(np.median(arr))
            MAD = float(np.median(np.abs(arr - median_rNSR)))
            rNSR_scale = 1.4826 * MAD if MAD > 0 else float("nan")
        else:
            median_rNSR = float("nan")
            rNSR_scale = float("nan")

        # Pass 2 — assign quality/problem flags
        rows: List[Dict[str, Any]] = []
        for m in intermediate:
            a1_quality, a1_problem = m["a1"]
            nan_present = m["nan_present"]
            std_val = m["std"]
            max_run = m["max_run"]
            peak_to_peak = m["peak_to_peak"]
            rNSR = m["rNSR"]

            # Analysis 2: flat line — nan_present OR std < 0.5 µV
            if nan_present or (not np.isnan(std_val) and std_val < 0.5):
                a2_quality, a2_problem = "BAD", "FLAT"
            else:
                a2_quality, a2_problem = "GOOD", "NONE"

            # Analysis 3: high-freq noise (rNSR with per-session robust z-score)
            if not np.isnan(rNSR) and not np.isnan(rNSR_scale) and rNSR_scale > 0:
                rNSR_zscore = round((rNSR - median_rNSR) / rNSR_scale, 4)
            else:
                rNSR_zscore = float("nan")

            rNSR_over = not np.isnan(rNSR) and rNSR > 0.5
            z_over = not np.isnan(rNSR_zscore) and abs(rNSR_zscore) > 3
            if rNSR_over and z_over:
                a3_quality, a3_problem = "BAD", "ARTIFACT"
            elif rNSR_over or z_over:
                a3_quality, a3_problem = "NOISY", "ARTIFACT"
            else:
                a3_quality, a3_problem = "GOOD", "NONE"

            # Analysis 4: clipping
            a4_quality = "BAD" if max_run >= 5 else "GOOD"
            a4_problem = "CLIPPING" if max_run >= 5 else "NONE"

            # Analysis 5: amplitude artifact
            if not np.isnan(peak_to_peak) and peak_to_peak > 500:
                a5_quality, a5_problem = "BAD", "ARTIFACT"
            else:
                a5_quality, a5_problem = "GOOD", "NONE"

            all_results = [
                (a1_quality, a1_problem),
                (a2_quality, a2_problem),
                (a3_quality, a3_problem),
                (a4_quality, a4_problem),
                (a5_quality, a5_problem),
            ]
            quality_flag = max((r[0] for r in all_results), key=lambda q: _q[q])
            problem_set = {r[1] for r in all_results}
            problem_flag = next(p for p in ("FLAT", "CLIPPING", "ARTIFACT", "NONE") if p in problem_set)

            rows.append({
                "window_id": m["window_id"],
                "channel_id": m["channel_id"],
                "samples": m["samples"],
                "std": std_val,
                "max_run": max_run,
                "peak_to_peak": peak_to_peak,
                "rNSR": rNSR,
                "rNSR_zscore": rNSR_zscore,
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
        trial_duration_s = float(sig_cfg.get("trial_duration_s", 20.0))
        df = _process_single_wav(data, "eav", entity_id, file_id, window_size_s,
                                 max_duration_s=trial_duration_s)
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
    total_windows = math.floor(trial_duration_s / window_size_s)

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
        video_section = qf_cfg.get("signals", {}).get("video", {})
        video_cfg = video_section.get("datasets", {}).get("eav", {})
        rows = _compute_video_windows(sd, total_windows, window_size_s, video_cfg)
        if rows:
            yield obj.object_name, pd.DataFrame(rows)
        else:
            logger.warning("[EAV] [%s] Trial %s video → no windows produced", entity_id, trial_id)


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
    modality_filter: Optional[Set[str]] = None,
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

        if modality_filter is not None:
            if signal_type.lower() not in modality_filter:
                continue
        else:
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
