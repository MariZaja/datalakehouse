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
logger = logging.getLogger("silver_time_audit")

NAN = float("nan")

OUTPUT_COLUMNS = [
    "dataset", "participant_id", "unit_id", "signal_type", "device", "modality",
    "sampling_rate_hz", "n_samples", "start_rel_s", "end_rel_s", "duration_s",
    "expected_samples", "coverage_pct",
]


# File extractors

def extract_csv_timing(
    data: bytes,
    timestamp_col: str,
    timestamp_unit_ms: bool = False,
) -> Optional[Dict[str, Any]]:

    try:
        df = pd.read_csv(io.BytesIO(data), usecols=[timestamp_col])
        ts = df[timestamp_col].dropna().values.astype(float)
        if len(ts) == 0:
            return None
        if timestamp_unit_ms:
            ts = ts / 1000.0
        n = len(ts)
        first_ts = float(ts[0])
        last_ts = float(ts[-1])
        if n > 1:
            deltas = np.diff(ts)
            median_delta = float(np.median(deltas))
            measured_hz = 1.0 / median_delta if median_delta > 0 else NAN
        else:
            measured_hz = NAN
        return {
            "first_ts_s": first_ts,
            "last_ts_s": last_ts,
            "n_samples": n,
            "measured_hz": measured_hz,
        }
    except Exception as e:
        logger.warning("CSV timing extraction failed: %s", e)
        return None


def extract_wav_metadata(data: bytes) -> Optional[Dict[str, Any]]:
    try:
        import soundfile as sf
        buf = io.BytesIO(data)
        with sf.SoundFile(buf) as f:
            sr = float(f.samplerate)
            n_samples = int(f.frames)
            duration_s = n_samples / sr if sr > 0 else NAN
        return {"sampling_rate_hz": sr, "n_samples": n_samples, "duration_s": duration_s}
    except Exception as e:
        logger.warning("WAV metadata extraction failed: %s", e)
        return None


def _mp4_find_box(
    data: bytes, target: bytes, start: int = 0
) -> Tuple[int, int]:
    import struct
    i = start
    while i + 8 <= len(data):
        raw_size = struct.unpack_from(">I", data, i)[0]
        btype = data[i + 4 : i + 8]
        if raw_size == 1:
            if i + 16 > len(data):
                break
            size = struct.unpack_from(">Q", data, i + 8)[0]
            hdr = 16
        elif raw_size == 0:
            size = len(data) - i
            hdr = 8
        else:
            size = raw_size
            hdr = 8
        if size < 8:
            break
        if btype == target:
            return i + hdr, i + size
        i += size
    return -1, -1


def _mp4_read_fullbox_timescale_duration(payload: bytes) -> Tuple[int, int]:
    import struct
    version = payload[0]
    if version == 0:
        ts = struct.unpack_from(">I", payload, 12)[0]
        dur = struct.unpack_from(">I", payload, 16)[0]
    else:
        ts = struct.unpack_from(">I", payload, 20)[0]
        dur = struct.unpack_from(">Q", payload, 24)[0]
    return ts, dur


def extract_mp4_metadata(data: bytes) -> Optional[Dict[str, Any]]:
    import struct

    moov_s, moov_e = _mp4_find_box(data, b"moov")
    if moov_s == -1:
        logger.warning("MP4: moov box not found (%d bytes)", len(data))
        return None
    moov = data[moov_s:moov_e]

    mvhd_s, mvhd_e = _mp4_find_box(moov, b"mvhd")
    if mvhd_s == -1:
        logger.warning("MP4: mvhd box not found in moov")
        return None
    movie_ts, movie_dur = _mp4_read_fullbox_timescale_duration(moov[mvhd_s:mvhd_e])
    duration_s = float(movie_dur) / movie_ts if movie_ts > 0 else NAN

    fps: Optional[float] = None
    n_frames: Optional[int] = None

    trak_cursor = 0
    while True:
        trak_s, trak_e = _mp4_find_box(moov, b"trak", trak_cursor)
        if trak_s == -1:
            break
        trak = moov[trak_s:trak_e]
        trak_cursor = trak_e

        mdia_s, mdia_e = _mp4_find_box(trak, b"mdia")
        if mdia_s == -1:
            continue
        mdia = trak[mdia_s:mdia_e]

        hdlr_s, hdlr_e = _mp4_find_box(mdia, b"hdlr")
        if hdlr_s == -1:
            continue
        hdlr = mdia[hdlr_s:hdlr_e]
        if hdlr[8:12] != b"vide":
            continue

        mdhd_s, mdhd_e = _mp4_find_box(mdia, b"mdhd")
        media_ts = 0
        if mdhd_s != -1:
            media_ts, _ = _mp4_read_fullbox_timescale_duration(mdia[mdhd_s:mdhd_e])

        minf_s, minf_e = _mp4_find_box(mdia, b"minf")
        if minf_s == -1:
            break
        minf = mdia[minf_s:minf_e]
        stbl_s, stbl_e = _mp4_find_box(minf, b"stbl")
        if stbl_s == -1:
            break
        stbl = minf[stbl_s:stbl_e]
        stts_s, stts_e = _mp4_find_box(stbl, b"stts")
        if stts_s == -1:
            break
        stts = stbl[stts_s:stts_e]
        entry_count = struct.unpack_from(">I", stts, 4)[0]
        total_samples = 0
        total_delta = 0
        off = 8
        for _ in range(entry_count):
            if off + 8 > len(stts):
                break
            sc = struct.unpack_from(">I", stts, off)[0]
            sd = struct.unpack_from(">I", stts, off + 4)[0]
            total_samples += sc
            total_delta += sc * sd
            off += 8
        n_frames = total_samples
        if total_delta > 0 and media_ts > 0:
            fps = float(total_samples) / (float(total_delta) / media_ts)
        break

    if fps is None:
        fps = NAN
    if n_frames is None or n_frames == 0:
        n_frames = int(fps * duration_s) if not np.isnan(fps) and not np.isnan(duration_s) else 0

    return {"sampling_rate_hz": fps, "n_samples": n_frames, "duration_s": duration_s}


def extract_mat_shape(data: bytes) -> Optional[Tuple[int, ...]]:
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
                        shape = hf[key].shape
                        return tuple(reversed(shape))
            return None

        import scipy.io
        variables = scipy.io.whosmat(tmp_path)
        data_vars = [(name, shape, dtype) for name, shape, dtype in variables
                     if not name.startswith("_")]
        if not data_vars:
            return None
        return data_vars[0][1]

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
    sampling_rate_hz: Optional[float],
    n_samples: int,
    start_rel_s: Optional[float],
    end_rel_s: Optional[float],
    duration_s: Optional[float],
) -> Dict[str, Any]:
    sr = sampling_rate_hz if sampling_rate_hz is not None else NAN
    dur = duration_s if duration_s is not None else NAN
    start = start_rel_s if start_rel_s is not None else NAN
    end = end_rel_s if end_rel_s is not None else NAN

    if not (np.isnan(sr) or np.isnan(dur)) and sr > 0 and dur > 0:
        expected = sr * dur
    else:
        expected = NAN

    if not np.isnan(expected) and expected > 0:
        coverage = (n_samples / expected) * 100.0
    else:
        coverage = NAN

    return {
        "dataset": dataset,
        "participant_id": participant_id,
        "unit_id": unit_id,
        "signal_type": signal_type,
        "device": device,
        "modality": modality,
        "sampling_rate_hz": None if np.isnan(sr) else sr,
        "n_samples": n_samples,
        "start_rel_s": None if np.isnan(start) else start,
        "end_rel_s": None if np.isnan(end) else end,
        "duration_s": None if np.isnan(dur) else dur,
        "expected_samples": None if np.isnan(expected) else int(round(expected)),
        "coverage_pct": None if np.isnan(coverage) else coverage,
    }


# K-EmoCon dataset audit
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
            "initTime": int(row["initTime"]),
            "startTime": int(row["startTime"]),
            "endTime": int(row["endTime"]),
        }
    return result


def _pid_from_entity_id(entity_id: str) -> int:
    """e01 → 1, e02 → 2, ..."""
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
                meta = extract_csv_timing(data, timestamp_col, timestamp_unit_ms)
                if meta is None:
                    logger.warning("[K-EmoCon] [%s] No timing extracted from %s", entity_id, filename)
                    continue

                if start_time_s is not None:
                    s_rel = meta["first_ts_s"] - start_time_s
                    e_rel = meta["last_ts_s"] - start_time_s
                else:
                    s_rel = e_rel = None

                dur = (meta["last_ts_s"] - meta["first_ts_s"]) if meta["n_samples"] > 1 else 0.0

                rows.append(make_row(
                    dataset=dataset_label,
                    participant_id=entity_id,
                    unit_id=entity_id,
                    signal_type=sig["signal_type"],
                    device=sig["device"],
                    modality=sig["modality"],
                    sampling_rate_hz=sig.get("declared_hz") or meta["measured_hz"],
                    n_samples=meta["n_samples"],
                    start_rel_s=s_rel,
                    end_rel_s=e_rel,
                    duration_s=dur,
                ))

            elif filename.endswith(".wav"):
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                meta = extract_wav_metadata(data)
                if meta is None:
                    continue
                rows.append(make_row(
                    dataset=dataset_label,
                    participant_id=entity_id,
                    unit_id=entity_id,
                    signal_type=audio_sig.get("signal_type", "audio"),
                    device=audio_sig.get("device", "unknown"),
                    modality=audio_sig.get("modality", "audio"),
                    sampling_rate_hz=meta["sampling_rate_hz"],
                    n_samples=meta["n_samples"],
                    start_rel_s=0.0,
                    end_rel_s=meta["duration_s"],
                    duration_s=meta["duration_s"],
                ))

            elif filename.endswith(".mp4"):
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                meta = extract_mp4_metadata(data)
                if meta is None:
                    continue
                rows.append(make_row(
                    dataset=dataset_label,
                    participant_id=entity_id,
                    unit_id=entity_id,
                    signal_type=video_sig.get("signal_type", "video"),
                    device=video_sig.get("device", "unknown"),
                    modality=video_sig.get("modality", "video"),
                    sampling_rate_hz=meta["sampling_rate_hz"],
                    n_samples=meta["n_samples"],
                    start_rel_s=0.0,
                    end_rel_s=meta["duration_s"],
                    duration_s=meta["duration_s"],
                ))

            elif filename.endswith(".csv"):
                logger.debug("[K-EmoCon] [%s] Unmapped CSV, skipping: %s", entity_id, filename)

        except Exception as e:
            logger.error("[K-EmoCon] [%s] Error processing %s: %s", entity_id, filename, e)

    return rows


# EAV dataset audit
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

    rows = []

    for obj in objects:
        filename = obj.object_name.split("/")[-1]
        key = obj.object_name

        try:
            if filename.endswith(".mat"):
                stem = Path(filename).stem
                if stem.endswith(label_suffix):
                    logger.debug("[EAV] [%s] Skipping label file: %s", entity_id, filename)
                    continue

                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue

                shape = extract_mat_shape(data)
                if shape is None:
                    logger.warning("[EAV] [%s] Cannot extract shape from %s", entity_id, filename)
                    continue

                if len(shape) < 2:
                    logger.warning("[EAV] [%s] Unexpected MAT shape %s in %s — skipping",
                                   entity_id, shape, filename)
                    continue

                if len(shape) == 3:
                    n_timepoints = shape[timepoints_axis]
                    n_instances = shape[instances_axis]
                elif len(shape) == 2:
                    n_instances = 1
                    n_timepoints = shape[0] if shape[0] > shape[1] else shape[1]
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
                        dataset=dataset_label,
                        participant_id=entity_id,
                        unit_id=f"{i:03d}",
                        signal_type=eeg_sig.get("signal_type", "eeg"),
                        device=eeg_sig.get("device", "BrainAmp"),
                        modality=eeg_sig.get("modality", "eeg"),
                        sampling_rate_hz=declared_eeg_hz,
                        n_samples=n_timepoints,
                        start_rel_s=0.0,
                        end_rel_s=duration_per_instance,
                        duration_s=duration_per_instance,
                    ))

            elif filename.endswith(".wav"):
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                meta = extract_wav_metadata(data)
                if meta is None:
                    continue
                trial_id = _extract_trial_id(filename, trial_id_pattern)
                rows.append(make_row(
                    dataset=dataset_label,
                    participant_id=entity_id,
                    unit_id=trial_id,
                    signal_type=audio_sig.get("signal_type", "audio"),
                    device=audio_sig.get("device", "microphone"),
                    modality=audio_sig.get("modality", "audio"),
                    sampling_rate_hz=meta["sampling_rate_hz"],
                    n_samples=meta["n_samples"],
                    start_rel_s=0.0,
                    end_rel_s=meta["duration_s"],
                    duration_s=meta["duration_s"],
                ))

            elif filename.endswith(".mp4"):
                data = download_object(minio_client, bucket, key)
                if data is None:
                    continue
                meta = extract_mp4_metadata(data)
                if meta is None:
                    continue
                trial_id = _extract_trial_id(filename, trial_id_pattern)
                rows.append(make_row(
                    dataset=dataset_label,
                    participant_id=entity_id,
                    unit_id=trial_id,
                    signal_type=video_sig.get("signal_type", "video"),
                    device=video_sig.get("device", "webcam"),
                    modality=video_sig.get("modality", "video"),
                    sampling_rate_hz=meta["sampling_rate_hz"],
                    n_samples=meta["n_samples"],
                    start_rel_s=0.0,
                    end_rel_s=meta["duration_s"],
                    duration_s=meta["duration_s"],
                ))

        except Exception as e:
            logger.error("[EAV] [%s] Error processing %s: %s", entity_id, filename, e)

    return rows


def _group_objects_by_entity(minio_client, bucket: str, prefix: str) -> Dict[str, List[Any]]:
    """List all objects under prefix and group them by entity_id."""
    entity_objects: Dict[str, List[Any]] = {}
    full_prefix = prefix.rstrip("/") + "/"
    for obj in minio_client.list_objects(bucket, prefix=full_prefix, recursive=True):
        for seg in obj.object_name.split("/"):
            if seg.startswith("entity="):
                eid = seg[len("entity="):]
                entity_objects.setdefault(eid, []).append(obj)
                break
    return entity_objects


def run_time_audit(
    minio_client,
    silver_bucket: str,
    bronze_bucket: str,
    ta_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    datasets_cfg = ta_cfg.get("datasets", {})

    # --- K-EmoCon ---
    kemocon_cfg = datasets_cfg.get("kemocon")
    if kemocon_cfg:
        logger.info("=== Auditing K-EmoCon ===")
        subjects_map = load_kemocon_subjects(
            minio_client,
            kemocon_cfg["subjects_bucket"],
            kemocon_cfg["subjects_path"],
        )
        logger.info("Loaded K-EmoCon subjects.csv: %d participants", len(subjects_map))

        entity_objects = _group_objects_by_entity(
            minio_client, silver_bucket, kemocon_cfg["silver_files_prefix"]
        )
        logger.info("K-EmoCon entities found: %d", len(entity_objects))

        for entity_id in sorted(entity_objects):
            logger.info("[K-EmoCon] Auditing %s (%d files)", entity_id, len(entity_objects[entity_id]))
            rows = audit_kemocon_entity(
                minio_client, silver_bucket, entity_id,
                entity_objects[entity_id], subjects_map, kemocon_cfg,
            )
            logger.info("[K-EmoCon] [%s] → %d rows", entity_id, len(rows))
            all_rows.extend(rows)

    # --- EAV ---
    eav_cfg = datasets_cfg.get("eav")
    if eav_cfg:
        logger.info("=== Auditing EAV ===")
        entity_objects = _group_objects_by_entity(
            minio_client, silver_bucket, eav_cfg["silver_files_prefix"]
        )
        logger.info("EAV entities found: %d", len(entity_objects))

        for entity_id in sorted(entity_objects):
            logger.info("[EAV] Auditing %s (%d files)", entity_id, len(entity_objects[entity_id]))
            rows = audit_eav_entity(
                minio_client, silver_bucket, entity_id,
                entity_objects[entity_id], eav_cfg,
            )
            logger.info("[EAV] [%s] → %d rows", entity_id, len(rows))
            all_rows.extend(rows)

    return all_rows


# Output helpers
def upload_csv(minio_client, bucket: str, key: str, df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    minio_client.put_object(
        bucket, key,
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="text/csv",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver — Step 2: Time Audit.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client, _ = project_config.config()

    silver_bucket = cfg["bucket_silver"]
    bronze_bucket = cfg["bucket"]
    ta_cfg = cfg.get("time_audit", {})

    logger.info("Starting Silver — Step 02: Time Audit")

    rows = run_time_audit(minio_client, silver_bucket, bronze_bucket, ta_cfg)

    if not rows:
        logger.error("No rows produced — check logs for errors.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    logger.info("Total rows: %d", len(df))

    # Upload to MinIO
    output_prefix = ta_cfg.get("output_prefix", "02_time_audit/metadata")
    output_filename = ta_cfg.get("output_filename", "time_audit.csv")
    minio_key = f"{output_prefix.rstrip('/')}/{output_filename}"
    upload_csv(minio_client, silver_bucket, minio_key, df)
    logger.info("Uploaded to MinIO: %s/%s", silver_bucket, minio_key)

    logger.info("Time Audit complete.")


if __name__ == "__main__":
    main()