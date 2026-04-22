import argparse
import io
import json
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
logger = logging.getLogger("silver_missingness")

NAN = float("nan")

# Minimum run lengths to avoid spurious gap detection
MIN_NAN_RUN_CSV = 3
MIN_ZERO_RUN_CSV = 5
MIN_ZERO_RUN_WAV = 2205

OUTPUT_COLUMNS = [
    "dataset", "participant_id", "unit_id", "signal_type", "device", "modality",
    "status", "expected_samples", "actual_samples", "missing_samples", "missing_pct",
    "known_in_literature", "literature_missing_pct",
]

# Signal name mappings: auxiliary CSV column → signal_type used in pipeline
AVAILABILITY_COL_TO_SIGNAL: Dict[str, str] = {
    "E4_ACC": "E4_ACC", "E4_BVP": "E4_BVP", "E4_EDA": "E4_EDA",
    "E4_HR": "E4_HR", "E4_IBI": "E4_IBI", "E4_TEMP": "E4_TEMP",
    "BrainWave": "BrainWave", "Attention": "Attention", "Meditation": "Meditation",
    "Polar_HR": "Polar_HR",
    "debate_audio": "audio", "debate_recording": "video",
}
E4_COMPLETENESS_COL_TO_SIGNAL: Dict[str, str] = {
    "ACC": "E4_ACC", "BVP": "E4_BVP", "EDA": "E4_EDA",
    "HR": "E4_HR", "IBI": "E4_IBI", "TEMP": "E4_TEMP",
}
NEURO_POLAR_COMPLETENESS_COL_TO_SIGNAL: Dict[str, str] = {
    "BrainWave": "BrainWave", "Attention": "Attention", "Meditation": "Meditation",
    "Polar_HR": "Polar_HR",
}


# MinIO helpers

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
    """List all objects under prefix and group them by entity_id path segment."""
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


def upload_json(minio_client, bucket: str, key: str, payload: dict) -> None:
    json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    minio_client.put_object(
        bucket, key,
        data=io.BytesIO(json_bytes),
        length=len(json_bytes),
        content_type="application/json",
    )


# Gap helpers

def _apply_min_run(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Zero out runs shorter than min_run samples in a boolean mask."""
    if min_run <= 1 or not mask.any():
        return mask
    result = np.zeros_like(mask)
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if (e - s) >= min_run:
            result[s:e] = True
    return result


def mask_to_gaps(is_missing: np.ndarray, sr: float) -> List[Dict[str, Any]]:
    """Convert a boolean missing-sample mask to a list of gap interval dicts."""
    gaps: List[Dict[str, Any]] = []
    if len(is_missing) == 0 or sr <= 0 or np.isnan(sr):
        return gaps
    padded = np.concatenate(([False], is_missing, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        duration_s = float(e - s) / sr
        gaps.append({
            "start_sample": int(s),
            "end_sample": int(e - 1),
            "start_s": round(float(s) / sr, 6),
            "end_s": round(float(e - 1) / sr, 6),
            "duration_s": round(duration_s, 6),
        })
    return gaps


# File scanning

def scan_csv_missingness(
    data: bytes,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    do_zero_check: bool,
) -> Dict[str, Any]:
    """Scan a K-EmoCon CSV signal file for NaN runs and zero runs."""
    try:
        df = pd.read_csv(io.BytesIO(data))
        if df.empty or timestamp_col not in df.columns:
            return {"error": "empty_or_no_timestamp"}

        actual_samples = len(df)
        signal_cols = [c for c in df.columns if c != timestamp_col]

        if not signal_cols:
            return {
                "actual_samples": actual_samples,
                "missing_mask": np.zeros(actual_samples, dtype=bool),
                "sr": NAN,
                "detection_method": "sample_count",
            }

        # Keep only numeric columns — text columns (e.g. device serial) are metadata
        # and should not be scanned for signal gaps.
        numeric_cols = [c for c in signal_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            return {
                "actual_samples": actual_samples,
                "missing_mask": np.zeros(actual_samples, dtype=bool),
                "sr": NAN,
                "detection_method": "sample_count",
            }
        sig_data = df[numeric_cols].to_numpy(dtype=float)

        # NaN mask: any signal column is NaN in that row
        nan_raw = np.isnan(sig_data).any(axis=1)
        nan_runs = _apply_min_run(nan_raw, MIN_NAN_RUN_CSV)

        # Zero mask: all signal columns are exactly 0 for sustained run
        if do_zero_check:
            zero_raw = (sig_data == 0).all(axis=1)
            zero_runs = _apply_min_run(zero_raw, MIN_ZERO_RUN_CSV)
        else:
            zero_runs = np.zeros(actual_samples, dtype=bool)

        missing_mask = nan_runs | zero_runs

        # Infer sampling rate from timestamps
        ts = df[timestamp_col].dropna().values.astype(float)
        if timestamp_unit_ms:
            ts = ts / 1000.0
        if len(ts) > 1:
            median_delta = float(np.median(np.diff(ts)))
            sr = 1.0 / median_delta if median_delta > 0 else NAN
        else:
            sr = NAN

        if nan_runs.any():
            detection = "nan_scan"
        elif zero_runs.any():
            detection = "zero_run"
        else:
            detection = "sample_count"

        return {
            "actual_samples": actual_samples,
            "missing_mask": missing_mask,
            "sr": sr,
            "detection_method": detection,
        }
    except Exception as e:
        logger.warning("CSV scan failed: %s", e)
        return {"error": str(e)}


def scan_wav_missingness(data: bytes, do_zero_check: bool) -> Dict[str, Any]:
    """Scan a WAV file for sample count and (optionally) zero-silence runs."""
    try:
        import soundfile as sf
        buf = io.BytesIO(data)
        with sf.SoundFile(buf) as f:
            sr = float(f.samplerate)
            n_samples = int(f.frames)
            if do_zero_check and n_samples > 0:
                samples = f.read(dtype="float32")
                mono = samples[:, 0] if samples.ndim > 1 else samples
                zero_raw = np.abs(mono) < 1e-7
                zero_runs = _apply_min_run(zero_raw, MIN_ZERO_RUN_WAV)
                missing_mask = zero_runs
                detection = "zero_run" if zero_runs.any() else "sample_count"
            else:
                missing_mask = np.zeros(n_samples, dtype=bool)
                detection = "sample_count"

        return {
            "actual_samples": n_samples,
            "missing_mask": missing_mask,
            "sr": sr,
            "detection_method": detection,
        }
    except Exception as e:
        logger.warning("WAV scan failed: %s", e)
        return {"error": str(e)}


def scan_mp4_frame_count(data: bytes) -> Optional[int]:
    """Extract video frame count from an MP4 file via box parsing (no temp file needed)."""
    import struct

    def find_box(buf: bytes, target: bytes, start: int = 0) -> Tuple[int, int]:
        i = start
        while i + 8 <= len(buf):
            raw_sz = struct.unpack_from(">I", buf, i)[0]
            btype = buf[i + 4:i + 8]
            if raw_sz == 1:
                if i + 16 > len(buf):
                    break
                size = struct.unpack_from(">Q", buf, i + 8)[0]
                hdr = 16
            elif raw_sz == 0:
                size = len(buf) - i
                hdr = 8
            else:
                size = raw_sz
                hdr = 8
            if size < 8:
                break
            if btype == target:
                return i + hdr, i + size
            i += size
        return -1, -1

    try:
        moov_s, moov_e = find_box(data, b"moov")
        if moov_s == -1:
            logger.warning("MP4: moov box not found (%d bytes)", len(data))
            return None
        moov = data[moov_s:moov_e]

        cursor = 0
        while True:
            trak_s, trak_e = find_box(moov, b"trak", cursor)
            if trak_s == -1:
                break
            trak = moov[trak_s:trak_e]
            cursor = trak_e

            mdia_s, mdia_e = find_box(trak, b"mdia")
            if mdia_s == -1:
                continue
            mdia = trak[mdia_s:mdia_e]

            hdlr_s, _ = find_box(mdia, b"hdlr")
            if hdlr_s == -1:
                continue
            if mdia[hdlr_s + 8:hdlr_s + 12] != b"vide":
                continue  # not a video track

            minf_s, minf_e = find_box(mdia, b"minf")
            if minf_s == -1:
                break
            minf = mdia[minf_s:minf_e]
            stbl_s, stbl_e = find_box(minf, b"stbl")
            if stbl_s == -1:
                break
            stbl = minf[stbl_s:stbl_e]
            stts_s, stts_e = find_box(stbl, b"stts")
            if stts_s == -1:
                break
            stts = stbl[stts_s:stts_e]

            entry_count = struct.unpack_from(">I", stts, 4)[0]
            total_frames = 0
            off = 8
            for _ in range(entry_count):
                if off + 8 > len(stts):
                    break
                sc = struct.unpack_from(">I", stts, off)[0]
                total_frames += sc
                off += 8
            return total_frames

    except Exception as e:
        logger.warning("MP4 frame count failed: %s", e)
    return None


def extract_mat_shape(data: bytes) -> Optional[Tuple[int, ...]]:
    """Return .mat matrix shape without loading full data (identical to time_audit approach)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        is_hdf5 = data[:8] == b"\x89HDF\r\n\x1a\n"

        if is_hdf5:
            import h5py
            with h5py.File(tmp_path, "r") as hf:
                for key in hf.keys():
                    if not key.startswith("#"):
                        return tuple(reversed(hf[key].shape))
            return None

        import scipy.io
        variables = scipy.io.whosmat(tmp_path)
        data_vars = [(n, s, d) for n, s, d in variables if not n.startswith("_")]
        if not data_vars:
            return None
        return data_vars[0][1]

    except Exception as e:
        logger.warning("MAT shape extraction failed: %s", e)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# Row builder

def make_miss_row(
    dataset: str,
    participant_id: str,
    unit_id: str,
    signal_type: str,
    device: str,
    modality: str,
    status: str,
    expected_samples: Optional[int],
    actual_samples: int,
    known_in_literature: bool,
    literature_missing_pct: Optional[float] = None,
) -> Dict[str, Any]:
    exp = expected_samples if expected_samples is not None else 0
    missing = max(0, exp - actual_samples) if exp > 0 else 0
    missing_pct = round((missing / exp) * 100.0, 4) if exp > 0 else 0.0
    if literature_missing_pct is not None and missing_pct > literature_missing_pct:
        known_in_literature = False
    return {
        "dataset": dataset,
        "participant_id": participant_id,
        "unit_id": unit_id,
        "signal_type": signal_type,
        "device": device,
        "modality": modality,
        "status": status,
        "expected_samples": expected_samples if expected_samples is not None else 0,
        "actual_samples": actual_samples,
        "missing_samples": missing,
        "missing_pct": missing_pct,
        "known_in_literature": known_in_literature or (status == "complete"),
        "literature_missing_pct": literature_missing_pct,
    }


def _total_missing_row(
    dataset: str,
    participant_id: str,
    unit_id: str,
    signal_type: str,
    device: str,
    modality: str,
    expected_samples: Optional[int],
    known_in_literature: bool,
    literature_missing_pct: Optional[float] = None,
) -> Dict[str, Any]:
    return make_miss_row(
        dataset, participant_id, unit_id, signal_type, device, modality,
        "total_missing", expected_samples, 0, known_in_literature,
        literature_missing_pct,
    )


# Known-missing lookup

def build_known_missing_lookup(known_cfg: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """Build (participant_id, signal_type) → {scope, reason, approx_missing_pct}."""
    lookup: Dict[Tuple[str, str], Dict] = {}
    for entry in known_cfg:
        for pid in entry.get("participants", []):
            for sig in entry.get("signals", []):
                lookup[(pid, sig)] = {
                    "scope": entry.get("scope", "total"),
                    "reason": entry.get("reason", ""),
                    "approx_missing_pct": entry.get("approx_missing_pct"),
                }
    return lookup


# Auxiliary-file based known-missing / completeness loaders (K-EmoCon)

def load_data_availability(
    minio_client, bucket: str, key: str,
) -> Dict[Tuple[str, str], bool]:
    data = download_object(minio_client, bucket, key)
    if data is None:
        logger.warning("Could not load data_availability from %s/%s — known_in_literature will be False for all", bucket, key)
        return {}
    df = pd.read_csv(io.BytesIO(data))
    absent: Dict[Tuple[str, str], bool] = {}
    for _, row in df.iterrows():
        try:
            pid = int(row["pid"])
        except (KeyError, ValueError):
            continue
        entity_id = f"e{pid:02d}"
        for col, signal_type in AVAILABILITY_COL_TO_SIGNAL.items():
            if col not in row.index:
                continue
            if str(row[col]).strip().upper() == "FALSE":
                absent[(entity_id, signal_type)] = True
    logger.info("data_availability: %d (entity, signal) pairs known absent", len(absent))
    return absent


def load_completeness_tables(
    minio_client, bucket: str, e4_key: str, neuro_key: str,
) -> Dict[Tuple[str, str], float]:
    """Read e4_completeness.csv + neuro_polar_completeness.csv →
    (entity_id, signal_type) → literature_missing_pct (0–100).

    Rows are positional (row i = participant i, 1-indexed).
    n/a  → 100.0 (file totally absent).
    ratio r → (1 - r) * 100.
    """
    result: Dict[Tuple[str, str], float] = {}

    def _load_one(key: str, col_map: Dict[str, str]) -> None:
        data = download_object(minio_client, bucket, key)
        if data is None:
            logger.warning("Could not load completeness table from %s/%s", bucket, key)
            return
        df = pd.read_csv(io.BytesIO(data))
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            entity_id = f"e{i:02d}"
            for col, signal_type in col_map.items():
                if col not in row.index:
                    continue
                val = str(row[col]).strip().lower()
                if val == "n/a":
                    result[(entity_id, signal_type)] = 100.0
                else:
                    try:
                        ratio = float(val)
                        result[(entity_id, signal_type)] = round((1.0 - ratio) * 100.0, 4)
                    except ValueError:
                        pass

    _load_one(e4_key, E4_COMPLETENESS_COL_TO_SIGNAL)
    _load_one(neuro_key, NEURO_POLAR_COMPLETENESS_COL_TO_SIGNAL)
    logger.info("completeness tables: %d (entity, signal) literature_missing_pct values", len(result))
    return result


# time_audit.csv lookup

def build_ta_lookup(ta_df: pd.DataFrame) -> Dict[Tuple[str, str, str, str], Optional[int]]:
    """(dataset_label, participant_id, unit_id, signal_type) → expected_samples."""
    lookup: Dict[Tuple[str, str, str, str], Optional[int]] = {}
    for _, row in ta_df.iterrows():
        key = (
            str(row["dataset"]),
            str(row["participant_id"]),
            str(row["unit_id"]),
            str(row["signal_type"]),
        )
        exp = row.get("expected_samples")
        lookup[key] = int(exp) if pd.notna(exp) else None
    return lookup


def _determine_status(
    actual_samples: int,
    expected_samples: Optional[int],
    missing_mask_count: int,
) -> str:
    if actual_samples == 0:
        return "total_missing"
    if missing_mask_count > 0:
        return "partial_missing"
    if expected_samples is not None and actual_samples < expected_samples - 1:
        return "partial_missing"
    return "complete"


# K-EmoCon audit

def audit_kemocon_missingness(
    minio_client,
    bucket: str,
    entity_objects: Dict[str, List[Any]],
    ta_df: pd.DataFrame,
    ds_cfg: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict]]:
    dataset_label = ds_cfg["dataset_label"]
    timestamp_col = ds_cfg.get("timestamp_col", "timestamp")
    ts_unit_ms = ds_cfg.get("timestamp_unit_ms", False)
    expected_signals: List[Dict] = ds_cfg.get("expected_signals", [])
    ta_lookup = build_ta_lookup(ta_df[ta_df["dataset"] == dataset_label])

    # Load known-missingness and literature completeness from auxiliary files
    avail_key = ds_cfg.get("data_availability_key")
    e4_comp_key = ds_cfg.get("e4_completeness_key")
    neuro_comp_key = ds_cfg.get("neuro_polar_completeness_key")

    known_absent: Dict[Tuple[str, str], bool] = {}
    completeness_lookup: Dict[Tuple[str, str], float] = {}

    if avail_key:
        known_absent = load_data_availability(minio_client, bucket, avail_key)
    if e4_comp_key and neuro_comp_key:
        completeness_lookup = load_completeness_tables(minio_client, bucket, e4_comp_key, neuro_comp_key)

    # All expected participants: those with files + those known absent in any signal
    known_pids = {pid for (pid, _) in known_absent.keys()}
    all_pids = sorted(set(entity_objects.keys()) | known_pids)
    logger.info("[K-EmoCon] Total participants to audit: %d", len(all_pids))

    rows: List[Dict] = []
    gaps: List[Dict] = []

    for entity_id in all_pids:
        # Index this entity's objects by filename and by extension
        obj_by_fname: Dict[str, Any] = {}
        obj_by_ext: Dict[str, List[Any]] = {}
        for obj in entity_objects.get(entity_id, []):
            fname = obj.object_name.split("/")[-1]
            obj_by_fname[fname] = obj
            ext = Path(fname).suffix.lower()
            obj_by_ext.setdefault(ext, []).append(obj)

        for sig in expected_signals:
            signal_type = sig["signal_type"]
            device = sig["device"]
            modality = sig["modality"]
            unit_id = entity_id  # K-EmoCon: one session per participant
            do_zero = sig.get("zero_check", False)
            do_gaps = sig.get("gap_detection", True)

            lit_missing_pct: Optional[float] = completeness_lookup.get((entity_id, signal_type))
            known_lit = (
                (entity_id, signal_type) in known_absent
                or (lit_missing_pct is not None and lit_missing_pct > 0)
            )

            ta_key = (dataset_label, entity_id, unit_id, signal_type)
            expected_samples = ta_lookup.get(ta_key)

            # Locate the file for this signal
            obj = None
            if sig.get("filename"):
                obj = obj_by_fname.get(sig["filename"])
            elif sig.get("ext"):
                candidates = obj_by_ext.get(sig["ext"], [])
                obj = candidates[0] if candidates else None

            if obj is None:
                if not known_lit:
                    logger.warning(
                        "[K-EmoCon] [%s] %s not found in MinIO. Available files: %s",
                        entity_id, sig.get("filename") or sig.get("ext"),
                        sorted(obj_by_fname.keys()) or "(entity has no files indexed)",
                    )
                rows.append(_total_missing_row(
                    dataset_label, entity_id, unit_id, signal_type, device, modality,
                    expected_samples, known_lit, lit_missing_pct,
                ))
                logger.info("[K-EmoCon] [%s] %-12s → total_missing%s",
                            entity_id, signal_type, " (known)" if known_lit else "")
                continue

            data = download_object(minio_client, bucket, obj.object_name)
            if data is None:
                rows.append(_total_missing_row(
                    dataset_label, entity_id, unit_id, signal_type, device, modality,
                    expected_samples, known_lit, lit_missing_pct,
                ))
                continue

            # Scan based on file type
            ext_lower = (sig.get("filename", "") or sig.get("ext", "")).lower()
            if ext_lower.endswith(".csv"):
                scan = scan_csv_missingness(data, timestamp_col, ts_unit_ms, do_zero)
            elif ext_lower.endswith(".wav"):
                scan = scan_wav_missingness(data, do_zero)
            elif ext_lower.endswith(".mp4"):
                n_frames = scan_mp4_frame_count(data)
                scan = {
                    "actual_samples": n_frames if n_frames is not None else 0,
                    "missing_mask": np.array([], dtype=bool),
                    "sr": NAN,
                    "detection_method": "frame_count",
                }
                if n_frames is None:
                    scan["error"] = "mp4_parse_failed"
            else:
                scan = scan_csv_missingness(data, timestamp_col, ts_unit_ms, do_zero)

            if "error" in scan:
                logger.error("[K-EmoCon] [%s] %s scan error (file=%s): %s",
                             entity_id, signal_type, obj.object_name, scan["error"])
                rows.append(_total_missing_row(
                    dataset_label, entity_id, unit_id, signal_type, device, modality,
                    expected_samples, known_lit, lit_missing_pct,
                ))
                continue

            actual_samples = scan["actual_samples"]
            missing_mask: np.ndarray = scan.get("missing_mask", np.array([], dtype=bool))
            sr = scan.get("sr", NAN)

            gap_list: List[Dict] = []
            if do_gaps and len(missing_mask) > 0 and not np.isnan(sr) and sr > 0:
                gap_list = mask_to_gaps(missing_mask, sr)

            masked_count = int(missing_mask.sum()) if len(missing_mask) > 0 else 0
            status = _determine_status(actual_samples, expected_samples, masked_count)

            rows.append(make_miss_row(
                dataset_label, entity_id, unit_id, signal_type, device, modality,
                status, expected_samples, actual_samples, known_lit, lit_missing_pct,
            ))

            if status == "partial_missing" and gap_list:
                gaps.append({
                    "participant_id": entity_id,
                    "unit_id": unit_id,
                    "device": device,
                    "modality": modality,
                    "intervals": gap_list,
                })

            logger.info("[K-EmoCon] [%s] %-12s → %s (actual=%d, expected=%s, gaps=%d)",
                        entity_id, signal_type, status, actual_samples,
                        expected_samples or "?", len(gap_list))

    return rows, gaps


# EAV audit

def _extract_trial_id(filename: str, pattern: str) -> Optional[str]:
    m = re.match(pattern, filename)
    return m.group(1) if m else None


def _build_expected_trial_ids(sig_cfg: Dict[str, Any]) -> List[str]:
    """Generate expected trial IDs from per-signal config (supports step)."""
    start = sig_cfg.get("trial_id_start", 1)
    step = sig_cfg.get("trial_id_step", 1)
    count = sig_cfg.get("trial_id_count", 30)
    fmt = sig_cfg.get("trial_id_format", "{:03d}")
    return [fmt.format(i) for i in range(start, start + step * count, step)]


def audit_eav_missingness(
    minio_client,
    bucket: str,
    entity_objects: Dict[str, List[Any]],
    ta_df: pd.DataFrame,
    ds_cfg: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict]]:
    dataset_label = ds_cfg["dataset_label"]
    expected_signals: List[Dict] = ds_cfg.get("expected_signals", [])
    known_lookup = build_known_missing_lookup(ds_cfg.get("known_missing", []))
    ta_lookup = build_ta_lookup(ta_df[ta_df["dataset"] == dataset_label])
    trial_id_pattern = ds_cfg.get("trial_id_pattern", r"^(\d+)_")
    eeg_label_suffix = ds_cfg.get("eeg_label_suffix", "_label")
    eeg_tp_axis = int(ds_cfg.get("eeg_timepoints_axis", 0))
    eeg_inst_axis = int(ds_cfg.get("eeg_instances_axis", 2))

    all_pids = sorted(entity_objects.keys())
    logger.info("[EAV] Total participants to audit: %d", len(all_pids))

    rows: List[Dict] = []
    gaps: List[Dict] = []

    for entity_id in all_pids:
        obj_list = entity_objects.get(entity_id, [])

        # Bucket objects by type
        mat_objs = [o for o in obj_list
                    if o.object_name.endswith(".mat")
                    and not Path(o.object_name).stem.endswith(eeg_label_suffix)]
        wav_objs = {_extract_trial_id(o.object_name.split("/")[-1], trial_id_pattern): o
                    for o in obj_list if o.object_name.endswith(".wav")}
        mp4_objs = {_extract_trial_id(o.object_name.split("/")[-1], trial_id_pattern): o
                    for o in obj_list if o.object_name.endswith(".mp4")}
        # Remove None keys (unmatched filenames)
        wav_objs = {k: v for k, v in wav_objs.items() if k is not None}
        mp4_objs = {k: v for k, v in mp4_objs.items() if k is not None}

        for sig in expected_signals:
            signal_type = sig["signal_type"]
            device = sig["device"]
            modality = sig["modality"]
            do_zero = sig.get("zero_check", False)
            do_gaps = sig.get("gap_detection", False)
            per_trial = sig.get("per_trial", False)
            ext = sig.get("ext", "")

            # EEG: single .mat per participant
            if not per_trial and ext == ".mat":
                known_info = known_lookup.get((entity_id, signal_type), {})
                known_lit = bool(known_info)
                expected_instances = sig.get("expected_instances", 30)
                expected_trial_range = [f"{i:03d}" for i in range(expected_instances)]

                def _eeg_total_missing() -> None:
                    for trial_id in expected_trial_range:
                        ta_key = (dataset_label, entity_id, trial_id, signal_type)
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            ta_lookup.get(ta_key), known_lit,
                        ))

                if not mat_objs:
                    _eeg_total_missing()
                    logger.info("[EAV] [%s] eeg → total_missing (mat absent)", entity_id)
                    continue

                data = download_object(minio_client, bucket, mat_objs[0].object_name)
                if data is None:
                    _eeg_total_missing()
                    continue

                # Shape-only check — same approach as time_audit (no full matrix load)
                shape = extract_mat_shape(data)
                if shape is None:
                    logger.error("[EAV] [%s] MAT shape extraction failed: %s",
                                 entity_id, mat_objs[0].object_name)
                    _eeg_total_missing()
                    continue

                if len(shape) == 3:
                    n_timepoints = shape[eeg_tp_axis]
                    n_instances = shape[eeg_inst_axis]
                elif len(shape) == 2:
                    n_instances = 1
                    n_timepoints = max(shape)
                else:
                    n_instances = 1
                    n_timepoints = shape[0]

                # One row per expected trial (0-indexed, matching time_audit)
                for i in range(expected_instances):
                    trial_id = f"{i:03d}"
                    ta_key = (dataset_label, entity_id, trial_id, signal_type)
                    expected_samples = ta_lookup.get(ta_key)

                    if i >= n_instances:
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            expected_samples, known_lit,
                        ))
                        continue

                    status = _determine_status(n_timepoints, expected_samples, 0)
                    rows.append(make_miss_row(
                        dataset_label, entity_id, trial_id, signal_type, device, modality,
                        status, expected_samples, n_timepoints, known_lit,
                    ))

                logger.info("[EAV] [%s] eeg → %d/%d instances",
                            entity_id, n_instances, expected_instances)

            # Audio / Video: one file per trial
            elif per_trial:
                obj_map = wav_objs if ext == ".wav" else mp4_objs
                known_info = known_lookup.get((entity_id, signal_type), {})
                known_lit = bool(known_info)
                signal_trial_ids = _build_expected_trial_ids(sig)

                found_count = 0
                missing_count = 0

                for trial_id in signal_trial_ids:
                    ta_key = (dataset_label, entity_id, trial_id, signal_type)
                    expected_samples = ta_lookup.get(ta_key)
                    obj = obj_map.get(trial_id)

                    if obj is None:
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            expected_samples, known_lit,
                        ))
                        missing_count += 1
                        continue

                    found_count += 1
                    file_data = download_object(minio_client, bucket, obj.object_name)
                    if file_data is None:
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            expected_samples, known_lit,
                        ))
                        continue

                    if ext == ".wav":
                        scan = scan_wav_missingness(file_data, do_zero)
                    else:
                        n_frames = scan_mp4_frame_count(file_data)
                        scan = {
                            "actual_samples": n_frames if n_frames is not None else 0,
                            "missing_mask": np.array([], dtype=bool),
                            "sr": NAN,
                            "detection_method": "frame_count",
                        }
                        if n_frames is None:
                            scan["error"] = "mp4_parse_failed"

                    if "error" in scan:
                        logger.error("[EAV] [%s] %s trial %s scan error (file=%s): %s",
                                     entity_id, signal_type, trial_id,
                                     obj.object_name, scan["error"])
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            expected_samples, known_lit,
                        ))
                        continue

                    actual_samples = scan["actual_samples"]
                    missing_mask_arr: np.ndarray = scan.get("missing_mask", np.array([], dtype=bool))
                    sr = scan.get("sr", NAN)

                    gap_list_trial: List[Dict] = []
                    if do_gaps and len(missing_mask_arr) > 0 and not np.isnan(sr) and sr > 0:
                        gap_list_trial = mask_to_gaps(missing_mask_arr, sr)

                    masked_c = int(missing_mask_arr.sum()) if len(missing_mask_arr) > 0 else 0
                    trial_status = _determine_status(actual_samples, expected_samples, masked_c)

                    rows.append(make_miss_row(
                        dataset_label, entity_id, trial_id, signal_type, device, modality,
                        trial_status, expected_samples, actual_samples, known_lit,
                    ))

                    if trial_status == "partial_missing" and gap_list_trial:
                        gaps.append({
                            "participant_id": entity_id,
                            "unit_id": trial_id,
                            "device": device,
                            "modality": modality,
                            "intervals": gap_list_trial,
                        })

                logger.info("[EAV] [%s] %-6s → found=%d, missing=%d / %d expected trials",
                            entity_id, signal_type, found_count, missing_count,
                            len(signal_trial_ids))

    return rows, gaps


# Orchestration

def run_missingness_detection(
    minio_client,
    silver_bucket: str,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict], List[Dict]]:
    md_cfg = cfg.get("missingness_detection", {})
    datasets_cfg = md_cfg.get("datasets", {})
    time_audit_key = md_cfg.get("time_audit_key", "02_time_audit/metadata/time_audit.csv")

    # Load time_audit.csv
    logger.info("Loading time_audit.csv from %s/%s", silver_bucket, time_audit_key)
    ta_data = download_object(minio_client, silver_bucket, time_audit_key)
    if ta_data is None:
        logger.error("Cannot load time_audit.csv — aborting.")
        sys.exit(1)
    ta_df = pd.read_csv(io.BytesIO(ta_data))
    logger.info("time_audit.csv loaded: %d rows", len(ta_df))

    all_rows: List[Dict] = []
    all_gaps: List[Dict] = []

    # K-EmoCon
    kemocon_cfg = datasets_cfg.get("kemocon")
    if kemocon_cfg:
        logger.info("=== Auditing K-EmoCon ===")
        entity_objects = _group_objects_by_entity(
            minio_client, silver_bucket, kemocon_cfg["silver_files_prefix"]
        )
        logger.info("K-EmoCon entities in MinIO: %d", len(entity_objects))
        krows, kgaps = audit_kemocon_missingness(
            minio_client, silver_bucket, entity_objects, ta_df, kemocon_cfg
        )
        logger.info("K-EmoCon: %d report rows, %d gap entries", len(krows), len(kgaps))
        all_rows.extend(krows)
        all_gaps.extend(kgaps)

    # EAV
    eav_cfg = datasets_cfg.get("eav")
    if eav_cfg:
        logger.info("=== Auditing EAV ===")
        entity_objects = _group_objects_by_entity(
            minio_client, silver_bucket, eav_cfg["silver_files_prefix"]
        )
        logger.info("EAV entities in MinIO: %d", len(entity_objects))
        erows, egaps = audit_eav_missingness(
            minio_client, silver_bucket, entity_objects, ta_df, eav_cfg
        )
        logger.info("EAV: %d report rows, %d gap entries", len(erows), len(egaps))
        all_rows.extend(erows)
        all_gaps.extend(egaps)

    return all_rows, all_gaps


# Entry point

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver — Step 03: Missingness Detection.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client, _ = project_config.config()
    silver_bucket = cfg["bucket_silver"]
    md_cfg = cfg.get("missingness_detection", {})
    output_prefix = md_cfg.get("output_prefix", "03_missingness").rstrip("/")
    report_filename = md_cfg.get("output_report_filename", "missingness_report.csv")
    gaps_filename = md_cfg.get("output_gaps_filename", "missingness_gaps.json")

    logger.info("Starting Silver — Step 03: Missingness Detection")

    rows, gaps = run_missingness_detection(minio_client, silver_bucket, cfg)

    if not rows:
        logger.error("No rows produced — check logs for errors.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    logger.info("Total report rows: %d", len(df))

    total_missing = (df["status"] == "total_missing").sum()
    partial_missing = (df["status"] == "partial_missing").sum()
    complete = (df["status"] == "complete").sum()
    logger.info("Status summary: complete=%d, partial_missing=%d, total_missing=%d",
                complete, partial_missing, total_missing)

    # Upload report CSV
    report_key = f"{output_prefix}/{report_filename}"
    upload_csv(minio_client, silver_bucket, report_key, df)
    logger.info("Uploaded report: %s/%s", silver_bucket, report_key)

    # Upload gaps JSON
    gaps_payload = {
        "dataset": "all",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gaps": gaps,
    }
    gaps_key = f"{output_prefix}/{gaps_filename}"
    upload_json(minio_client, silver_bucket, gaps_key, gaps_payload)
    logger.info("Uploaded gaps: %s/%s (%d entries)", silver_bucket, gaps_key, len(gaps))

    logger.info("Missingness Detection complete.")


if __name__ == "__main__":
    main()