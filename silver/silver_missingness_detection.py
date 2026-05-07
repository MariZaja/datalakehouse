import argparse
import io
import logging
import os
import re
import sys
import tempfile
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

OUTPUT_COLUMNS = [
    "dataset", "participant_id", "unit_id", "signal_type", "device", "modality",
    "status", "sample_rate_hz", "expected_samples", "actual_samples",
    "missing_samples", "missing_pct", "known_in_literature",
    "zero_or_null_pct", "effective_missing_samples", "effective_missing_pct",
]

AVAILABILITY_COL_TO_SIGNAL: Dict[str, str] = {
    "E4_ACC": "E4_ACC", "E4_BVP": "E4_BVP", "E4_EDA": "E4_EDA",
    "E4_HR": "E4_HR", "E4_IBI": "E4_IBI", "E4_TEMP": "E4_TEMP",
    "BrainWave": "BrainWave", "Attention": "Attention", "Meditation": "Meditation",
    "Polar_HR": "Polar_HR",
    "debate_audio": "audio", "debate_recording": "video",
}
ZERO_CHECK_SIGNALS = {
    "E4_EDA", "E4_HR", "E4_IBI", "E4_TEMP", "Polar_HR", "Attention", "Meditation",
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


# File scanners

def scan_csv_in_window(
    data: bytes,
    timestamp_col: str,
    timestamp_unit_ms: bool,
    declared_hz: Optional[float],
    window_start_s: Optional[float],
    window_end_s: Optional[float],
    check_zeros: bool = False,
) -> Dict[str, Any]:
    """Read a CSV signal file, infer sample rate, count samples and bad values within debate window."""
    try:
        df = pd.read_csv(io.BytesIO(data))
        if df.empty or timestamp_col not in df.columns:
            return {"error": "empty_or_no_timestamp"}

        ts_raw = df[timestamp_col].copy()
        if timestamp_unit_ms:
            ts_raw = ts_raw / 1000.0

        ts = ts_raw.dropna().values.astype(float)

        if declared_hz is not None:
            sample_rate_hz: Optional[float] = float(declared_hz)
        elif len(ts) > 1:
            median_delta = float(np.median(np.diff(ts)))
            sample_rate_hz = 1.0 / median_delta if median_delta > 0 else None
        else:
            sample_rate_hz = None

        if window_start_s is not None and window_end_s is not None:
            in_window_mask = ts_raw.notna() & (ts_raw >= window_start_s) & (ts_raw <= window_end_s)
            actual_samples = int(in_window_mask.sum())
            df_window = df[in_window_mask]
        else:
            actual_samples = len(ts)
            df_window = df

        zero_or_null_samples = 0
        if "value" in df_window.columns and len(df_window) > 0:
            col = pd.to_numeric(df_window["value"], errors="coerce")
            null_mask = col.isnull()
            if check_zeros:
                bad_mask = null_mask | (col == 0)
            else:
                bad_mask = null_mask
            zero_or_null_samples = int(bad_mask.sum())
        elif len(df_window) > 0:
            logger.warning("CSV has no 'value' column — skipping zero/null check. Columns: %s",
                           list(df_window.columns))

        return {
            "sample_rate_hz": sample_rate_hz,
            "actual_samples": actual_samples,
            "zero_or_null_samples": zero_or_null_samples,
        }
    except Exception as e:
        logger.warning("CSV scan failed: %s", e)
        return {"error": str(e)}


def get_wav_metadata(data: bytes) -> Optional[Dict[str, Any]]:
    try:
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(data)) as f:
            return {"sample_rate_hz": float(f.samplerate), "n_samples": int(f.frames)}
    except Exception as e:
        logger.warning("WAV metadata extraction failed: %s", e)
        return None


def get_mp4_metadata(data: bytes) -> Optional[Dict[str, Any]]:
    import struct

    def _find_box(buf: bytes, target: bytes, start: int = 0) -> Tuple[int, int]:
        i = start
        while i + 8 <= len(buf):
            raw = struct.unpack_from(">I", buf, i)[0]
            btype = buf[i + 4:i + 8]
            if raw == 1:
                if i + 16 > len(buf):
                    break
                size, hdr = struct.unpack_from(">Q", buf, i + 8)[0], 16
            elif raw == 0:
                size, hdr = len(buf) - i, 8
            else:
                size, hdr = raw, 8
            if size < 8:
                break
            if btype == target:
                return i + hdr, i + size
            i += size
        return -1, -1

    try:
        moov_s, moov_e = _find_box(data, b"moov")
        if moov_s == -1:
            logger.warning("MP4: moov box not found")
            return None
        moov = data[moov_s:moov_e]

        cursor = 0
        while True:
            trak_s, trak_e = _find_box(moov, b"trak", cursor)
            if trak_s == -1:
                break
            trak = moov[trak_s:trak_e]
            cursor = trak_e

            mdia_s, mdia_e = _find_box(trak, b"mdia")
            if mdia_s == -1:
                continue
            mdia = trak[mdia_s:mdia_e]

            hdlr_s, _ = _find_box(mdia, b"hdlr")
            if hdlr_s == -1 or mdia[hdlr_s + 8:hdlr_s + 12] != b"vide":
                continue

            mdhd_s, mdhd_e = _find_box(mdia, b"mdhd")
            media_ts = 0
            if mdhd_s != -1:
                mdhd = mdia[mdhd_s:mdhd_e]
                v = mdhd[0]
                media_ts = struct.unpack_from(">I", mdhd, 12 if v == 0 else 20)[0]

            minf_s, minf_e = _find_box(mdia, b"minf")
            if minf_s == -1:
                break
            stbl_s, stbl_e = _find_box(mdia[minf_s:minf_e], b"stbl")
            if stbl_s == -1:
                break
            stbl = mdia[minf_s:minf_e][stbl_s:stbl_e]
            stts_s, stts_e = _find_box(stbl, b"stts")
            if stts_s == -1:
                break
            stts = stbl[stts_s:stts_e]

            entry_count = struct.unpack_from(">I", stts, 4)[0]
            total_frames = total_delta = 0
            off = 8
            for _ in range(entry_count):
                if off + 8 > len(stts):
                    break
                sc = struct.unpack_from(">I", stts, off)[0]
                sd = struct.unpack_from(">I", stts, off + 4)[0]
                total_frames += sc
                total_delta += sc * sd
                off += 8

            fps: Optional[float] = None
            if total_delta > 0 and media_ts > 0:
                fps = float(total_frames) / (float(total_delta) / media_ts)
            return {"sample_rate_hz": fps, "n_samples": total_frames}

        return None
    except Exception as e:
        logger.warning("MP4 metadata extraction failed: %s", e)
        return None


def _extract_mat_shape(data: bytes) -> Optional[Tuple[int, ...]]:
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
                        return tuple(reversed(hf[key].shape))
            return None

        import scipy.io
        variables = scipy.io.whosmat(tmp_path)
        data_vars = [(n, s, d) for n, s, d in variables if not n.startswith("_")]
        return data_vars[0][1] if data_vars else None
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
    sample_rate_hz: Optional[float],
    expected_samples: Optional[int],
    actual_samples: int,
    known_in_literature: bool,
    zero_or_null_samples: int = 0,
) -> Dict[str, Any]:
    exp = expected_samples if expected_samples is not None else 0
    missing = max(0, exp - actual_samples) if exp > 0 else 0
    missing_pct = round((missing / exp) * 100.0, 4) if exp > 0 else 0.0
    zero_or_null_pct = round((zero_or_null_samples / actual_samples) * 100.0, 4) if actual_samples > 0 else 0.0
    effective_missing = missing + zero_or_null_samples
    effective_missing_pct = round((effective_missing / exp) * 100.0, 4) if exp > 0 else 0.0
    return {
        "dataset": dataset,
        "participant_id": participant_id,
        "unit_id": unit_id,
        "signal_type": signal_type,
        "device": device,
        "modality": modality,
        "status": status,
        "sample_rate_hz": sample_rate_hz,
        "expected_samples": exp if exp > 0 else 0,
        "actual_samples": actual_samples,
        "missing_samples": missing,
        "missing_pct": missing_pct,
        "known_in_literature": known_in_literature or (status == "complete"),
        "zero_or_null_pct": zero_or_null_pct,
        "effective_missing_samples": effective_missing,
        "effective_missing_pct": effective_missing_pct,
    }


def _total_missing_row(
    dataset: str,
    participant_id: str,
    unit_id: str,
    signal_type: str,
    device: str,
    modality: str,
    sample_rate_hz: Optional[float],
    expected_samples: Optional[int],
    known_in_literature: bool,
) -> Dict[str, Any]:
    return make_miss_row(
        dataset, participant_id, unit_id, signal_type, device, modality,
        "total_missing", sample_rate_hz, 0, 0,
        known_in_literature,
    )


def _determine_status(actual_samples: int, expected_samples: Optional[int]) -> str:
    if actual_samples == 0:
        return "total_missing"
    if expected_samples is not None and actual_samples < expected_samples - 1:
        return "partial_missing"
    return "complete"


# Auxiliary-file loaders (K-EmoCon)

def load_kemocon_subjects(
    minio_client, bucket: str, path: str
) -> Dict[int, Dict[str, int]]:
    data = download_object(minio_client, bucket, path)
    if data is None:
        logger.error("Cannot load subjects.csv from %s/%s", bucket, path)
        return {}
    df = pd.read_csv(io.BytesIO(data))
    result = {}
    for _, row in df.iterrows():
        pid = int(row["pid"])
        result[pid] = {
            "startTime": int(row["startTime"]),
            "endTime": int(row["endTime"]),
        }
    return result


def load_data_availability(
    minio_client, bucket: str, key: str,
) -> Dict[Tuple[str, str], bool]:
    data = download_object(minio_client, bucket, key)
    if data is None:
        logger.warning("Could not load data_availability from %s/%s", bucket, key)
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


# Known-missing lookup (EAV)

def build_known_missing_lookup(known_cfg: List[Dict]) -> Dict[Tuple[str, str], Dict]:
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


# K-EmoCon audit

def _pid_from_entity_id(entity_id: str) -> int:
    return int(entity_id[1:])


def audit_kemocon_missingness(
    minio_client,
    bucket: str,
    entity_objects: Dict[str, List[Any]],
    subjects_map: Dict[int, Dict[str, int]],
    ds_cfg: Dict[str, Any],
) -> List[Dict]:
    dataset_label = ds_cfg["dataset_label"]
    timestamp_col = ds_cfg.get("timestamp_col", "timestamp")
    ts_unit_ms = ds_cfg.get("timestamp_unit_ms", False)
    expected_signals: List[Dict] = ds_cfg.get("expected_signals", [])

    known_absent: Dict[Tuple[str, str], bool] = {}

    avail_key = ds_cfg.get("data_availability_key")
    if avail_key:
        known_absent = load_data_availability(minio_client, bucket, avail_key)

    known_pids = {pid for (pid, _) in known_absent.keys()}
    all_pids = sorted(set(entity_objects.keys()) | known_pids)
    logger.info("[K-EmoCon] Total participants to audit: %d", len(all_pids))

    rows: List[Dict] = []

    for entity_id in all_pids:
        pid = _pid_from_entity_id(entity_id)
        subject = subjects_map.get(pid, {})
        if subject:
            debate_start_s = subject["startTime"] / 1000.0
            debate_end_s = subject["endTime"] / 1000.0
            debate_duration_s = debate_end_s - debate_start_s
        else:
            debate_start_s = debate_end_s = debate_duration_s = None
            logger.warning("[K-EmoCon] [%s] No subject entry — debate window unknown", entity_id)

        obj_by_fname: Dict[str, Any] = {}
        obj_by_ext: Dict[str, List[Any]] = {}
        for obj in entity_objects.get(entity_id, []):
            fname = obj.object_name.split("/")[-1]
            obj_by_fname[fname] = obj
            obj_by_ext.setdefault(Path(fname).suffix.lower(), []).append(obj)

        for sig in expected_signals:
            signal_type = sig["signal_type"]
            device = sig["device"]
            modality = sig["modality"]
            declared_hz: Optional[float] = sig.get("declared_hz")
            sr_for_missing = float(declared_hz) if declared_hz is not None else 0.0
            exp_for_missing = (
                int(round(debate_duration_s * declared_hz))
                if debate_duration_s is not None and declared_hz is not None
                else None
            )
            sig_ref = (sig.get("filename") or sig.get("ext", "")).lower()

            known_lit = (entity_id, signal_type) in known_absent

            # Locate file
            obj = None
            if sig.get("filename"):
                obj = obj_by_fname.get(sig["filename"])
            elif sig.get("ext"):
                candidates = obj_by_ext.get(sig["ext"], [])
                obj = candidates[0] if candidates else None

            # Warn on availability mismatches
            file_present = obj is not None
            file_should_be_absent = (entity_id, signal_type) in known_absent
            if not file_present and not file_should_be_absent:
                logger.warning(
                    "[K-EmoCon] [%s] %s not found and not flagged absent in data_availability. "
                    "Available: %s",
                    entity_id, sig_ref,
                    sorted(obj_by_fname.keys()) or "(no files)",
                )
            elif file_present and file_should_be_absent:
                logger.warning(
                    "[K-EmoCon] [%s] %s present in MinIO but flagged absent in data_availability",
                    entity_id, signal_type,
                )
                known_lit = False

            if obj is None:
                rows.append(_total_missing_row(
                    dataset_label, entity_id, entity_id, signal_type, device, modality,
                    sr_for_missing, exp_for_missing, known_lit,
                ))
                logger.info("[K-EmoCon] [%s] %-12s → total_missing%s",
                            entity_id, signal_type, " (known)" if known_lit else "")
                continue

            file_data = download_object(minio_client, bucket, obj.object_name)
            if file_data is None:
                rows.append(_total_missing_row(
                    dataset_label, entity_id, entity_id, signal_type, device, modality,
                    sr_for_missing, exp_for_missing, known_lit,
                ))
                continue

            # Scan file
            check_zeros = signal_type in ZERO_CHECK_SIGNALS
            zero_or_null_samples = 0

            if sig_ref.endswith(".csv"):
                scan = scan_csv_in_window(
                    file_data, timestamp_col, ts_unit_ms, declared_hz,
                    debate_start_s, debate_end_s, check_zeros=check_zeros,
                )
                if "error" in scan:
                    logger.error("[K-EmoCon] [%s] %s scan error: %s",
                                 entity_id, signal_type, scan["error"])
                    rows.append(_total_missing_row(
                        dataset_label, entity_id, entity_id, signal_type, device, modality,
                        sr_for_missing, exp_for_missing, known_lit,
                    ))
                    continue
                sample_rate_hz = scan["sample_rate_hz"]
                actual_samples = scan["actual_samples"]
                zero_or_null_samples = scan["zero_or_null_samples"]

            elif sig_ref.endswith(".wav"):
                meta = get_wav_metadata(file_data)
                if meta is None:
                    rows.append(_total_missing_row(
                        dataset_label, entity_id, entity_id, signal_type, device, modality,
                        sr_for_missing, exp_for_missing, known_lit,
                    ))
                    continue
                sample_rate_hz = meta["sample_rate_hz"]
                actual_samples = meta["n_samples"]

            elif sig_ref.endswith(".mp4"):
                meta = get_mp4_metadata(file_data)
                if meta is None:
                    rows.append(_total_missing_row(
                        dataset_label, entity_id, entity_id, signal_type, device, modality,
                        sr_for_missing, exp_for_missing, known_lit,
                    ))
                    continue
                sample_rate_hz = meta["sample_rate_hz"]
                actual_samples = meta["n_samples"]

            else:
                scan = scan_csv_in_window(
                    file_data, timestamp_col, ts_unit_ms, declared_hz,
                    debate_start_s, debate_end_s, check_zeros=check_zeros,
                )
                if "error" in scan:
                    rows.append(_total_missing_row(
                        dataset_label, entity_id, entity_id, signal_type, device, modality,
                        sr_for_missing, exp_for_missing, known_lit,
                    ))
                    continue
                sample_rate_hz = scan["sample_rate_hz"]
                actual_samples = scan["actual_samples"]
                zero_or_null_samples = scan["zero_or_null_samples"]

            # Expected samples from debate duration × sample rate
            if sample_rate_hz is not None and debate_duration_s is not None:
                expected_samples: Optional[int] = int(round(debate_duration_s * sample_rate_hz))
            else:
                expected_samples = None

            status = _determine_status(actual_samples, expected_samples)
            if status == "complete" and zero_or_null_samples > 0:
                status = "partial_missing"
            rows.append(make_miss_row(
                dataset_label, entity_id, entity_id, signal_type, device, modality,
                status, sample_rate_hz, expected_samples, actual_samples,
                known_lit, zero_or_null_samples,
            ))
            logger.info("[K-EmoCon] [%s] %-12s → %s (actual=%d, expected=%s)",
                        entity_id, signal_type, status, actual_samples,
                        expected_samples if expected_samples is not None else "?")

    return rows


# EAV audit

def _extract_trial_id(filename: str, pattern: str) -> Optional[str]:
    m = re.match(pattern, filename)
    return m.group(1) if m else None


def _build_expected_trial_ids(sig_cfg: Dict[str, Any]) -> List[str]:
    start = sig_cfg.get("trial_id_start", 1)
    step = sig_cfg.get("trial_id_step", 1)
    count = sig_cfg.get("trial_id_count", 30)
    fmt = sig_cfg.get("trial_id_format", "{:03d}")
    return [fmt.format(i) for i in range(start, start + step * count, step)]


def audit_eav_missingness(
    minio_client,
    bucket: str,
    entity_objects: Dict[str, List[Any]],
    ds_cfg: Dict[str, Any],
) -> List[Dict]:
    dataset_label = ds_cfg["dataset_label"]
    expected_signals: List[Dict] = ds_cfg.get("expected_signals", [])
    known_lookup = build_known_missing_lookup(ds_cfg.get("known_missing", []))
    trial_id_pattern = ds_cfg.get("trial_id_pattern", r"^(\d+)_")
    eeg_label_suffix = ds_cfg.get("eeg_label_suffix", "_label")
    eeg_tp_axis = int(ds_cfg.get("eeg_timepoints_axis", 0))
    eeg_inst_axis = int(ds_cfg.get("eeg_instances_axis", 2))

    all_pids = sorted(entity_objects.keys())
    logger.info("[EAV] Total participants to audit: %d", len(all_pids))

    rows: List[Dict] = []

    for entity_id in all_pids:
        obj_list = entity_objects.get(entity_id, [])

        mat_objs = [o for o in obj_list
                    if o.object_name.endswith(".mat")
                    and not Path(o.object_name).stem.endswith(eeg_label_suffix)]
        wav_objs = {_extract_trial_id(o.object_name.split("/")[-1], trial_id_pattern): o
                    for o in obj_list if o.object_name.endswith(".wav")}
        mp4_objs = {_extract_trial_id(o.object_name.split("/")[-1], trial_id_pattern): o
                    for o in obj_list if o.object_name.endswith(".mp4")}
        wav_objs = {k: v for k, v in wav_objs.items() if k is not None}
        mp4_objs = {k: v for k, v in mp4_objs.items() if k is not None}

        for sig in expected_signals:
            signal_type = sig["signal_type"]
            device = sig["device"]
            modality = sig["modality"]
            per_trial = sig.get("per_trial", False)
            ext = sig.get("ext", "")
            known_lit = bool(known_lookup.get((entity_id, signal_type), {}))

            # EEG: single .mat per participant
            if not per_trial and ext == ".mat":
                declared_eeg_hz = float(sig.get("declared_hz", 500.0))
                expected_instances = sig.get("expected_instances", 30)
                expected_timepoints: Optional[int] = sig.get("expected_timepoints")
                min_mat_size_bytes = int(sig.get("min_size_bytes", 0))

                if not mat_objs:
                    for i in range(expected_instances):
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, f"{i:03d}", signal_type, device, modality,
                            declared_eeg_hz, expected_timepoints, known_lit,
                        ))
                    logger.info("[EAV] [%s] eeg → total_missing (mat absent)", entity_id)
                    continue

                mat_obj = mat_objs[0]
                if min_mat_size_bytes > 0 and mat_obj.size < min_mat_size_bytes:
                    logger.error(
                        "[EAV] [%s] MAT file too small: %d bytes (< %d) — likely corrupt",
                        entity_id, mat_obj.size, min_mat_size_bytes,
                    )
                    for i in range(expected_instances):
                        rows.append(make_miss_row(
                            dataset_label, entity_id, f"{i:03d}", signal_type, device, modality,
                            "size_anomaly", declared_eeg_hz, expected_timepoints, 0, known_lit,
                        ))
                    continue

                file_data = download_object(minio_client, bucket, mat_obj.object_name)
                if file_data is None:
                    for i in range(expected_instances):
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, f"{i:03d}", signal_type, device, modality,
                            declared_eeg_hz, expected_timepoints, known_lit,
                        ))
                    continue

                shape = _extract_mat_shape(file_data)
                if shape is None or len(shape) < 2:
                    logger.error("[EAV] [%s] MAT shape extraction failed", entity_id)
                    for i in range(expected_instances):
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, f"{i:03d}", signal_type, device, modality,
                            declared_eeg_hz, expected_timepoints, known_lit,
                        ))
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

                for i in range(expected_instances):
                    trial_id = f"{i:03d}"
                    if i >= n_instances:
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            declared_eeg_hz, expected_timepoints, known_lit,
                        ))
                        continue
                    status = _determine_status(n_timepoints, expected_timepoints)
                    rows.append(make_miss_row(
                        dataset_label, entity_id, trial_id, signal_type, device, modality,
                        status, declared_eeg_hz, expected_timepoints, n_timepoints, known_lit,
                    ))

                logger.info("[EAV] [%s] eeg → %d/%d instances", entity_id, n_instances, expected_instances)

            # Audio / Video: one file per trial
            elif per_trial:
                obj_map = wav_objs if ext == ".wav" else mp4_objs
                signal_trial_ids = _build_expected_trial_ids(sig)
                trial_duration_s: Optional[float] = sig.get("trial_duration_s")
                found_count = missing_count = 0

                for trial_id in signal_trial_ids:
                    obj = obj_map.get(trial_id)
                    if obj is None:
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            None, None, known_lit,
                        ))
                        missing_count += 1
                        continue

                    found_count += 1
                    file_data = download_object(minio_client, bucket, obj.object_name)
                    if file_data is None:
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            None, None, known_lit,
                        ))
                        continue

                    if ext == ".wav":
                        meta = get_wav_metadata(file_data)
                    else:
                        meta = get_mp4_metadata(file_data)

                    if meta is None:
                        rows.append(_total_missing_row(
                            dataset_label, entity_id, trial_id, signal_type, device, modality,
                            None, None, known_lit,
                        ))
                        continue

                    sample_rate_hz = meta["sample_rate_hz"]
                    actual_samples = meta["n_samples"]
                    expected_samples = (
                        int(round(trial_duration_s * sample_rate_hz))
                        if trial_duration_s is not None and sample_rate_hz is not None
                        else None
                    )
                    status = _determine_status(actual_samples, expected_samples)
                    rows.append(make_miss_row(
                        dataset_label, entity_id, trial_id, signal_type, device, modality,
                        status, sample_rate_hz, expected_samples, actual_samples, known_lit,
                    ))

                logger.info("[EAV] [%s] %-6s → found=%d, missing=%d / %d",
                            entity_id, signal_type, found_count, missing_count, len(signal_trial_ids))

    return rows


# Orchestration

def run_missingness_detection(
    minio_client,
    silver_bucket: str,
    cfg: Dict[str, Any],
) -> List[Dict]:
    md_cfg = cfg.get("missingness_detection", {})
    datasets_cfg = md_cfg.get("datasets", {})

    all_rows: List[Dict] = []

    kemocon_cfg = datasets_cfg.get("kemocon")
    if kemocon_cfg:
        logger.info("=== Auditing K-EmoCon ===")
        subjects_map = load_kemocon_subjects(
            minio_client,
            kemocon_cfg["subjects_bucket"],
            kemocon_cfg["subjects_path"],
        )
        logger.info("K-EmoCon subjects loaded: %d", len(subjects_map))

        entity_objects = _group_objects_by_entity(
            minio_client, silver_bucket, kemocon_cfg["silver_files_prefix"]
        )
        logger.info("K-EmoCon entities in MinIO: %d", len(entity_objects))

        krows = audit_kemocon_missingness(
            minio_client, silver_bucket, entity_objects, subjects_map, kemocon_cfg,
        )
        logger.info("K-EmoCon: %d report rows", len(krows))
        all_rows.extend(krows)

    eav_cfg = datasets_cfg.get("eav")
    if eav_cfg:
        logger.info("=== Auditing EAV ===")
        entity_objects = _group_objects_by_entity(
            minio_client, silver_bucket, eav_cfg["silver_files_prefix"]
        )
        logger.info("EAV entities in MinIO: %d", len(entity_objects))

        erows = audit_eav_missingness(
            minio_client, silver_bucket, entity_objects, eav_cfg,
        )
        logger.info("EAV: %d report rows", len(erows))
        all_rows.extend(erows)

    return all_rows


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

    logger.info("Starting Silver — Step 03: Missingness Detection")

    rows = run_missingness_detection(minio_client, silver_bucket, cfg)

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

    report_key = f"{output_prefix}/{report_filename}"
    upload_csv(minio_client, silver_bucket, report_key, df)
    logger.info("Uploaded: %s/%s", silver_bucket, report_key)

    logger.info("Missingness Detection complete.")


if __name__ == "__main__":
    main()
