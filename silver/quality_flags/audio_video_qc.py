import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .signal_readers import read_wav_signal, read_video_signal

logger = logging.getLogger("silver_quality_flags")


# ── Audio helpers ──────────────────────────────────────────────────────────────

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
    elif m["kurtosis"] > 10 and m["max_delta"] > 0.5:
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
    start_s: float = 0.0,
    max_duration_s: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    sig = read_wav_signal(data)
    if sig is None:
        return None
    samples = sig["samples"]
    sr = sig["sample_rate_hz"]
    start_sample = int(round(start_s * sr))
    if max_duration_s is not None:
        end_sample = start_sample + int(round(max_duration_s * sr))
        samples = samples[start_sample:end_sample]
    else:
        samples = samples[start_sample:]
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


# ── Video helpers ──────────────────────────────────────────────────────────────

def _video_quality_problem(
    window_blur: float,
    window_clipping: float,
    window_noise: float,
    blur_bad: float,
    blur_good: float,
    clipping_noisy: float,
    clipping_bad: float,
    noise_noisy: float = 4.0,
    noise_bad: float = 8.0,
    include_clipping: bool = True,
) -> Tuple[str, str]:
    if include_clipping and window_clipping >= clipping_bad:
        problem_flag = "CLIPPING"
    elif window_blur < blur_bad or window_noise >= noise_bad:
        problem_flag = "ARTIFACT"
    else:
        problem_flag = "NONE"

    if (window_blur < blur_bad
            or (include_clipping and window_clipping >= clipping_bad)
            or window_noise >= noise_bad):
        quality_flag = "BAD"
    elif (window_blur < blur_good
          or (include_clipping and window_clipping >= clipping_noisy)
          or window_noise >= noise_noisy):
        quality_flag = "NOISY"
    else:
        quality_flag = "GOOD"

    return quality_flag, problem_flag


def _compute_video_windows(
    sd: Dict[str, Any],
    total_windows: int,
    window_size_s: float,
    video_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    fps = sd["sample_rate_hz"]
    frames_per_window = int(round(fps * window_size_s)) if fps > 0 else 1

    lap_vars = sd["lap_vars"]
    clipped_ratios = sd["clipped_ratios"]
    noise_sigmas = sd["noise_sigmas"]
    n_frames = len(lap_vars)

    blur_cfg = video_cfg.get("blur", {})
    clip_cfg = video_cfg.get("clipping", {})
    noise_cfg = video_cfg.get("noise", {})
    blur_bad = float(blur_cfg.get("bad", 20))
    blur_good = float(blur_cfg.get("good", 40))
    clipping_noisy = float(clip_cfg.get("noisy", 0.01))
    clipping_bad = float(clip_cfg.get("bad", 0.03))
    noise_noisy = float(noise_cfg.get("noisy", 4.0))
    noise_bad = float(noise_cfg.get("bad", 8.0))
    include_clipping = bool(video_cfg.get("use_clipping", True))

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
                "quality_flag": "BAD", "problem_flag": "ARTIFACT",
            })
            continue

        window_blur = float(np.mean(lap_vars[f_start:f_end]))
        window_clipping = float(np.mean(clipped_ratios[f_start:f_end]))
        window_noise = float(np.mean(noise_sigmas[f_start:f_end]))

        quality_flag, problem_flag = _video_quality_problem(
            window_blur, window_clipping, window_noise,
            blur_bad, blur_good, clipping_noisy, clipping_bad,
            noise_noisy, noise_bad, include_clipping,
        )

        rows.append({
            "window_id": wid,
            "window_start_s": w_start,
            "window_end_s": w_end,
            "window_blur": round(window_blur, 4),
            "window_clipping": round(window_clipping, 6),
            "window_noise": round(window_noise, 4),
            "quality_flag": quality_flag,
            "problem_flag": problem_flag,
        })

    return rows


# ── K-EmoCon audio / video processors ─────────────────────────────────────────

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
    return _process_single_wav(
        file_data, "k-emocon", entity_id, file_id, window_size_s,
        max_duration_s=debate_duration_s,
    )


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
    video_section = qf_cfg.get("signals", {}).get("video", {})
    video_cfg = video_section.get("datasets", {}).get("kemocon", {})
    rows = _compute_video_windows(sd, total_windows, window_size_s, video_cfg)
    return pd.DataFrame(rows) if rows else None
