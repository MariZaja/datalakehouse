import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("silver_quality_flags")

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

        cap.release()
        if not lap_vars:
            return None

        return {
            "lap_vars": np.array(lap_vars),
            "clipped_ratios": np.array(clipped_ratios),
            "noise_sigmas": np.array(noise_sigmas),
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
