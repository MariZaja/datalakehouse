import logging
import math
from typing import Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_audio_egemaps_windowed(
    audio_bytes: bytes,
    file_id: str,
    smile: Any,
    window_size_s: float,
    max_duration_s: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """Extract eGeMAPSv02 Functionals in fixed-length windows from a WAV file.

    Clips signal to max_duration_s (debate window) before windowing.
    Returns one row per window with window_id, window_start_s, window_end_s columns.
    """
    from silver.quality_flags.signal_readers import read_wav_signal
    sig = read_wav_signal(audio_bytes)
    if sig is None:
        logger.error("eGeMAPS windowed: could not read WAV for %s", file_id)
        return None
    samples = sig["samples"]
    sr = sig["sample_rate_hz"]
    if max_duration_s is not None:
        samples = samples[:int(round(max_duration_s * sr))]
    window_n = int(round(sr * window_size_s))
    if window_n == 0 or len(samples) == 0:
        return None
    n_windows = math.ceil(len(samples) / window_n)
    frames: List[pd.DataFrame] = []
    for wid in range(1, n_windows + 1):
        start = (wid - 1) * window_n
        end = min(wid * window_n, len(samples))
        chunk = samples[start:end].astype(np.float32)
        if len(chunk) == 0:
            continue
        try:
            df = smile.process_signal(chunk, sr)
            df = df.reset_index(drop=True)
            df.insert(0, "file_id", file_id)
            df.insert(1, "window_id", wid)
            df.insert(2, "window_start_s", round(start / sr, 3))
            df.insert(3, "window_end_s", round((wid * window_n) / sr, 3))
            frames.append(df)
        except Exception as e:
            logger.warning("[audio] window %d for %s: %s", wid, file_id, e)
    if not frames:
        logger.error("eGeMAPS windowed: no windows extracted for %s", file_id)
        return None
    return pd.concat(frames, ignore_index=True)


def init_opensmile() -> Optional[Any]:
    try:
        import opensmile
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        logger.info("openSMILE eGeMAPSv02 initialized")
        return smile
    except ImportError:
        logger.warning("opensmile not installed — audio features will be skipped")
        return None
