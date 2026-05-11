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
logger = logging.getLogger("silver_time_audit")

OUTPUT_COLUMNS = [
    "dataset", "participant_id", "unit_id", "signal_type", "device", "modality",
    "start_rel_s", "end_rel_s", "duration_s",
]

# File extractors

def extract_csv_timing(
    data: bytes,
    timestamp_col: str,
    timestamp_unit_ms: bool = False,
) -> Optional[Dict[str, float]]:
    try:
        df = pd.read_csv(io.BytesIO(data), usecols=[timestamp_col])
        ts = df[timestamp_col].dropna().values.astype(float)
        if len(ts) == 0:
            return None
        if timestamp_unit_ms:
            ts = ts / 1000.0
        return {"first_ts_s": float(ts[0]), "last_ts_s": float(ts[-1])}
    except Exception as e:
        logger.warning("CSV timing extraction failed: %s", e)
        return None


def extract_wav_duration(data: bytes) -> Optional[float]:
    try:
        import soundfile as sf
        with sf.SoundFile(io.BytesIO(data)) as f:
            return float(f.frames) / f.samplerate if f.samplerate > 0 else None
    except Exception as e:
        logger.warning("WAV duration extraction failed: %s", e)
        return None


def extract_mp4_duration(data: bytes) -> Optional[float]:
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
        mvhd_s, mvhd_e = _find_box(moov, b"mvhd")
        if mvhd_s == -1:
            return None
        payload = moov[mvhd_s:mvhd_e]
        version = payload[0]
        if version == 0:
            timescale = struct.unpack_from(">I", payload, 12)[0]
            duration = struct.unpack_from(">I", payload, 16)[0]
        else:
            timescale = struct.unpack_from(">I", payload, 20)[0]
            duration = struct.unpack_from(">Q", payload, 24)[0]
        return float(duration) / timescale if timescale > 0 else None
    except Exception as e:
        logger.warning("MP4 duration extraction failed: %s", e)
        return None


def extract_mat_shape(data: bytes) -> Optional[Tuple[int, ...]]:
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


# MinIO helper

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


# Row builder

def make_row(
    dataset: str,
    participant_id: str,
    unit_id: str,
    signal_type: str,
    device: str,
    modality: str,
    start_rel_s: Optional[float],
    end_rel_s: Optional[float],
    duration_s: Optional[float],
) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "participant_id": participant_id,
        "unit_id": unit_id,
        "signal_type": signal_type,
        "device": device,
        "modality": modality,
        "start_rel_s": start_rel_s,
        "end_rel_s": end_rel_s,
        "duration_s": duration_s,
    }


# K-EmoCon

def load_kemocon_subjects(
    minio_client, bucket: str, path: str
) -> Dict[int, Dict[str, int]]:
    data = download_object(minio_client, bucket, path)
    if data is None:
        logger.error("Cannot load K-EmoCon subjects.csv from %s/%s", bucket, path)
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


def _pid_from_entity_id(entity_id: str) -> int:
    return int(entity_id[1:])


def audit_kemocon_entity(
    minio_client,
    bucket: str,
    entity_id: str,
    objects: List[Any],
    subjects_map: Dict[int, Dict[str, int]],
    ta_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dataset_label = ta_cfg["dataset_label"]
    signal_map = ta_cfg.get("signals", {})
    timestamp_col = ta_cfg.get("timestamp_col", "timestamp")
    timestamp_unit_ms = ta_cfg.get("timestamp_unit_ms", False)
    audio_sig = ta_cfg.get("audio_signal", {})
    video_sig = ta_cfg.get("video_signal", {})

    pid = _pid_from_entity_id(entity_id)
    subject = subjects_map.get(pid, {})
    start_time_s = subject["startTime"] / 1000.0 if subject else None

    rows = []

    for obj in objects:
        filename = obj.object_name.split("/")[-1]
        key = obj.object_name

        try:
            if filename.endswith(".csv") and filename in signal_map:
                sig = signal_map[filename]
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                timing = extract_csv_timing(data, timestamp_col, timestamp_unit_ms)
                if timing is None:
                    logger.warning("[K-EmoCon] [%s] No timing from %s", entity_id, filename)
                    continue

                if start_time_s is not None:
                    s_rel = timing["first_ts_s"] - start_time_s
                    e_rel = timing["last_ts_s"] - start_time_s
                else:
                    s_rel = e_rel = None
                duration = timing["last_ts_s"] - timing["first_ts_s"]

                rows.append(make_row(
                    dataset_label, entity_id, entity_id,
                    sig["signal_type"], sig["device"], sig["modality"],
                    s_rel, e_rel, duration,
                ))

            elif filename.endswith(".wav"):
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                duration = extract_wav_duration(data)
                if duration is None:
                    continue
                rows.append(make_row(
                    dataset_label, entity_id, entity_id,
                    audio_sig.get("signal_type", "audio"),
                    audio_sig.get("device", "unknown"),
                    audio_sig.get("modality", "audio"),
                    0.0, duration, duration,
                ))

            elif filename.endswith(".mp4"):
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                duration = extract_mp4_duration(data)
                if duration is None:
                    continue
                rows.append(make_row(
                    dataset_label, entity_id, entity_id,
                    video_sig.get("signal_type", "video"),
                    video_sig.get("device", "unknown"),
                    video_sig.get("modality", "video"),
                    0.0, duration, duration,
                ))

            elif filename.endswith(".csv"):
                logger.debug("[K-EmoCon] [%s] Unmapped CSV, skipping: %s", entity_id, filename)

        except Exception as e:
            logger.error("[K-EmoCon] [%s] Error processing %s: %s", entity_id, filename, e)

    return rows


# EAV

def _extract_trial_id(filename: str, pattern: str) -> str:
    m = re.match(pattern, filename)
    return m.group(1) if m else Path(filename).stem


def audit_eav_entity(
    minio_client,
    bucket: str,
    entity_id: str,
    objects: List[Any],
    ta_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    dataset_label = ta_cfg["dataset_label"]
    eeg_sig = ta_cfg.get("eeg_signal", {})
    audio_sig = ta_cfg.get("audio_signal", {})
    video_sig = ta_cfg.get("video_signal", {})
    trial_id_pattern = ta_cfg.get("trial_id_pattern", r"^(\d+)_")
    label_suffix = eeg_sig.get("label_suffix", "_label")
    declared_eeg_hz = float(eeg_sig.get("declared_hz", 500.0))
    timepoints_axis = int(eeg_sig.get("timepoints_axis", 0))
    instances_axis = int(eeg_sig.get("instances_axis", 2))
    min_mat_size_bytes = int(eeg_sig.get("min_size_bytes", 0))

    rows = []

    for obj in objects:
        filename = obj.object_name.split("/")[-1]
        key = obj.object_name

        try:
            if filename.endswith(".mat"):
                if Path(filename).stem.endswith(label_suffix):
                    logger.debug("[EAV] [%s] Skipping label file: %s", entity_id, filename)
                    continue

                if min_mat_size_bytes > 0 and obj.size < min_mat_size_bytes:
                    logger.error(
                        "[EAV] [%s] MAT file too small: %s (%d bytes < %d) — likely corrupt",
                        entity_id, filename, obj.size, min_mat_size_bytes,
                    )
                    continue

                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue

                shape = extract_mat_shape(data)
                if shape is None or len(shape) < 2:
                    logger.warning("[EAV] [%s] Cannot extract shape from %s", entity_id, filename)
                    continue

                if len(shape) == 3:
                    n_timepoints = shape[timepoints_axis]
                    n_instances = shape[instances_axis]
                elif len(shape) == 2:
                    n_instances = 1
                    n_timepoints = max(shape)
                else:
                    n_instances = 1
                    n_timepoints = shape[0]

                duration_per_instance = n_timepoints / declared_eeg_hz
                logger.info(
                    "[EAV] [%s] %s: shape=%s → %d instances × %d timepoints",
                    entity_id, filename, shape, n_instances, n_timepoints,
                )

                for i in range(n_instances):
                    rows.append(make_row(
                        dataset_label, entity_id, f"{i:03d}",
                        eeg_sig.get("signal_type", "eeg"),
                        eeg_sig.get("device", "BrainAmp"),
                        eeg_sig.get("modality", "eeg"),
                        0.0, duration_per_instance, duration_per_instance,
                    ))

            elif filename.endswith(".wav"):
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                duration = extract_wav_duration(data)
                if duration is None:
                    continue
                trial_id = _extract_trial_id(filename, trial_id_pattern)
                rows.append(make_row(
                    dataset_label, entity_id, trial_id,
                    audio_sig.get("signal_type", "audio"),
                    audio_sig.get("device", "microphone"),
                    audio_sig.get("modality", "audio"),
                    0.0, duration, duration,
                ))

            elif filename.endswith(".mp4"):
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                duration = extract_mp4_duration(data)
                if duration is None:
                    continue
                trial_id = _extract_trial_id(filename, trial_id_pattern)
                rows.append(make_row(
                    dataset_label, entity_id, trial_id,
                    video_sig.get("signal_type", "video"),
                    video_sig.get("device", "webcam"),
                    video_sig.get("modality", "video"),
                    0.0, duration, duration,
                ))

        except Exception as e:
            logger.error("[EAV] [%s] Error processing %s: %s", entity_id, filename, e)

    return rows


# Grouping helper

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


# Orchestration

def run_time_audit(
    minio_client,
    silver_bucket: str,
    ta_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    datasets_cfg = ta_cfg.get("datasets", {})

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
        logger.info("K-EmoCon entities: %d", len(entity_objects))

        for entity_id in sorted(entity_objects):
            logger.info("[K-EmoCon] Auditing %s (%d files)", entity_id, len(entity_objects[entity_id]))
            rows = audit_kemocon_entity(
                minio_client, silver_bucket, entity_id,
                entity_objects[entity_id], subjects_map, kemocon_cfg,
            )
            logger.info("[K-EmoCon] [%s] → %d rows", entity_id, len(rows))
            all_rows.extend(rows)

    eav_cfg = datasets_cfg.get("eav")
    if eav_cfg:
        logger.info("=== Auditing EAV ===")
        entity_objects = _group_objects_by_entity(
            minio_client, silver_bucket, eav_cfg["silver_files_prefix"]
        )
        logger.info("EAV entities: %d", len(entity_objects))

        for entity_id in sorted(entity_objects):
            logger.info("[EAV] Auditing %s (%d files)", entity_id, len(entity_objects[entity_id]))
            rows = audit_eav_entity(
                minio_client, silver_bucket, entity_id,
                entity_objects[entity_id], eav_cfg,
            )
            logger.info("[EAV] [%s] → %d rows", entity_id, len(rows))
            all_rows.extend(rows)

    return all_rows


# ── Output ────────────────────────────────────────────────────────────────────

def upload_csv(minio_client, bucket: str, key: str, df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    minio_client.put_object(
        bucket, key,
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="text/csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver — Step 02: Time Audit.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client, _ = project_config.config()

    silver_bucket = cfg["bucket_silver"]
    ta_cfg = cfg.get("time_audit", {})

    logger.info("Starting Silver — Step 02: Time Audit")

    rows = run_time_audit(minio_client, silver_bucket, ta_cfg)

    if not rows:
        logger.error("No rows produced — check logs for errors.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    logger.info("Total rows: %d", len(df))

    output_prefix = ta_cfg.get("output_prefix", "02_time_audit/metadata")
    output_filename = ta_cfg.get("output_filename", "time_audit.csv")
    minio_key = f"{output_prefix.rstrip('/')}/{output_filename}"
    upload_csv(minio_client, silver_bucket, minio_key, df)
    logger.info("Uploaded: %s/%s", silver_bucket, minio_key)

    logger.info("Time Audit complete.")


if __name__ == "__main__":
    main()
