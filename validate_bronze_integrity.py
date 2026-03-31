import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import yaml
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_bronze_integrity")


# Configuration helpers
def _env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Required environment variable {key!r} is not set.")
    return val


def _get_minio_client() -> Minio:
    return Minio(
        _env("MINIO_ENDPOINT"),
        access_key=_env("MINIO_ACCESS_KEY"),
        secret_key=_env("MINIO_SECRET_KEY"),
        secure=False,
    )


# File entry
class FileEntry(NamedTuple):
    local_path: str
    s3_path: str
    dataset: str
    entity_id: str
    modality: str
    file_name: str
    table: str


# Manifest builders
def _build_eav_manifest(eav_root: str, base_path: str) -> List[FileEntry]:
    entries: List[FileEntry] = []
    modality_map = {"Audio": "audio", "Video": "video", "EEG": "eeg"}

    for item in sorted(os.listdir(eav_root)):
        subject_path = os.path.join(eav_root, item)
        if not item.startswith("subject") or not os.path.isdir(subject_path):
            continue
        for src_mod, dst_mod in modality_map.items():
            src_dir = os.path.join(subject_path, src_mod)
            if not os.path.isdir(src_dir):
                continue
            for file_name in sorted(os.listdir(src_dir)):
                if file_name.startswith("."):
                    continue
                local_file = os.path.join(src_dir, file_name)
                if not os.path.isfile(local_file):
                    continue
                s3_path = f"{base_path}/files/entity={item}/modality={dst_mod}/{file_name}"
                entries.append(FileEntry(
                    local_path=local_file, s3_path=s3_path,
                    dataset="EAV", entity_id=item,
                    modality=dst_mod, file_name=file_name, table="files",
                ))

    # Auxiliary files are one level above eav_root (e.g. data/EAV/, not data/EAV/EAV/)
    aux_root = os.path.dirname(eav_root.rstrip("/\\"))
    aux_files = {
        "meta_data.csv": "metadata",
        "subjects.csv": "metadata",
        "questionnaire.xlsx": "annotations",
    }
    for file_name, category in aux_files.items():
        local_file = os.path.join(aux_root, file_name)
        if not os.path.exists(local_file):
            continue
        s3_path = f"{base_path}/auxiliary/{category}/{file_name}"
        entries.append(FileEntry(
            local_path=local_file, s3_path=s3_path,
            dataset="EAV", entity_id="global",
            modality=category, file_name=file_name, table="aux",
        ))

    return entries


def _build_kemocon_manifest(kemocon_root: str, base_path: str) -> List[FileEntry]:
    entries: List[FileEntry] = []

    audio_dir = os.path.join(kemocon_root, "debate_audios")
    if os.path.isdir(audio_dir):
        for file_name in sorted(os.listdir(audio_dir)):
            if not file_name.endswith(".wav") or file_name.startswith("."):
                continue
            entity_id = file_name.replace(".wav", "").replace(".", "_")
            local_file = os.path.join(audio_dir, file_name)
            s3_path = f"{base_path}/files/entity={entity_id}/modality=audio/{file_name}"
            entries.append(FileEntry(
                local_path=local_file, s3_path=s3_path,
                dataset="K-EmoCon", entity_id=entity_id,
                modality="audio", file_name=file_name, table="files",
            ))

    video_dir = os.path.join(kemocon_root, "debate_recordings")
    if os.path.isdir(video_dir):
        for file_name in sorted(os.listdir(video_dir)):
            if not file_name.endswith(".mp4") or file_name.startswith("."):
                continue
            entity_id = file_name.split("_")[0]
            local_file = os.path.join(video_dir, file_name)
            s3_path = f"{base_path}/files/entity={entity_id}/modality=video/{file_name}"
            entries.append(FileEntry(
                local_path=local_file, s3_path=s3_path,
                dataset="K-EmoCon", entity_id=entity_id,
                modality="video", file_name=file_name, table="files",
            ))

    e4_dir = os.path.join(kemocon_root, "e4_data")
    if os.path.isdir(e4_dir):
        for participant_id in sorted(os.listdir(e4_dir)):
            part_dir = os.path.join(e4_dir, participant_id)
            if not os.path.isdir(part_dir) or participant_id.startswith("."):
                continue
            for file_name in sorted(os.listdir(part_dir)):
                if file_name.startswith("."):
                    continue
                local_file = os.path.join(part_dir, file_name)
                if not os.path.isfile(local_file):
                    continue
                s3_path = f"{base_path}/files/entity={participant_id}/modality=biosignal/{file_name}"
                entries.append(FileEntry(
                    local_path=local_file, s3_path=s3_path,
                    dataset="K-EmoCon", entity_id=participant_id,
                    modality="biosignal", file_name=file_name, table="files",
                ))

    eeg_dir = os.path.join(kemocon_root, "neurosky_polar_data")
    if os.path.isdir(eeg_dir):
        for participant_id in sorted(os.listdir(eeg_dir)):
            part_dir = os.path.join(eeg_dir, participant_id)
            if not os.path.isdir(part_dir) or participant_id.startswith("."):
                continue
            for file_name in sorted(os.listdir(part_dir)):
                if file_name.startswith("."):
                    continue
                local_file = os.path.join(part_dir, file_name)
                if not os.path.isfile(local_file):
                    continue
                s3_path = f"{base_path}/files/entity={participant_id}/modality=eeg/{file_name}"
                entries.append(FileEntry(
                    local_path=local_file, s3_path=s3_path,
                    dataset="K-EmoCon", entity_id=participant_id,
                    modality="eeg", file_name=file_name, table="files",
                ))

    aux_dirs = {
        "emotion_annotations": "annotations",
        "data_quality_tables": "quality",
        "metadata": "metadata",
    }
    for src_dir_name, category in aux_dirs.items():
        full_dir = os.path.join(kemocon_root, src_dir_name)
        if not os.path.isdir(full_dir):
            continue
        for root, _, files in os.walk(full_dir):
            for file_name in sorted(files):
                if file_name.startswith("."):
                    continue
                local_file = os.path.join(root, file_name)
                relative_path = os.path.relpath(local_file, kemocon_root)
                s3_path = f"{base_path}/auxiliary/{category}/{relative_path}"
                entries.append(FileEntry(
                    local_path=local_file, s3_path=s3_path,
                    dataset="K-EmoCon", entity_id="global",
                    modality=category, file_name=file_name, table="aux",
                ))

    return entries


# Validation rules lookup
def _get_rule_for_entry(entry: FileEntry, validation_cfg: Dict) -> Optional[Dict]:
    """Return the validation rule dict for this entry, or None."""
    ds_key = "eav" if entry.dataset == "EAV" else "kemocon"
    ds_cfg = validation_cfg.get("datasets", {}).get(ds_key, {})

    if entry.dataset == "EAV":
        # Modality → reverse map from dst_modality back to config key
        mod_map = {"audio": "Audio", "video": "Video", "eeg": "EEG"}
        cfg_key = mod_map.get(entry.modality)
        if cfg_key:
            return ds_cfg.get("modalities", {}).get(cfg_key)
    else:
        # K-EmoCon: map modality to the right config section
        flat_map = {"audio": "debate_audios", "video": "debate_recordings"}
        part_map = {"biosignal": "e4_data", "eeg": "neurosky_polar_data"}
        key = flat_map.get(entry.modality) or part_map.get(entry.modality)
        if key:
            return (ds_cfg.get("flat_dirs", {}).get(key)
                    or ds_cfg.get("participant_dirs", {}).get(key))
    return None


# Hash helpers
def _md5_local(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_hash(local_path: str, etag: str) -> Optional[str]:
    """Compare local MD5 with MinIO ETag. Skips multipart ETags (contain '-')."""
    etag = etag.strip('"').lower()
    if "-" in etag:
        return None
    try:
        local_md5 = _md5_local(local_path)
    except OSError as e:
        return f"Cannot read local file for hash check: {e}"
    if local_md5 != etag:
        return f"MD5 mismatch: local={local_md5}, MinIO ETag={etag}"
    return None


# Content checks
def _parse_wav_duration(header_bytes: bytes) -> float:
    import struct
    if len(header_bytes) < 12 or header_bytes[:4] != b"RIFF" or header_bytes[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file")

    sample_rate = channels = bits_per_sample = None
    offset = 12
    while offset + 8 <= len(header_bytes):
        chunk_id = header_bytes[offset:offset + 4]
        chunk_size = struct.unpack_from("<I", header_bytes, offset + 4)[0]

        if chunk_id == b"fmt ":
            if offset + 8 + 16 > len(header_bytes):
                raise ValueError("fmt chunk truncated")
            channels = struct.unpack_from("<H", header_bytes, offset + 10)[0]
            sample_rate = struct.unpack_from("<I", header_bytes, offset + 12)[0]
            bits_per_sample = struct.unpack_from("<H", header_bytes, offset + 22)[0]
        elif chunk_id == b"data":
            if sample_rate is None or channels is None or bits_per_sample is None:
                raise ValueError("data chunk found before fmt chunk")
            bytes_per_sample = bits_per_sample // 8
            if sample_rate == 0 or channels == 0 or bytes_per_sample == 0:
                raise ValueError("Invalid audio parameters in fmt chunk")
            return chunk_size / (sample_rate * channels * bytes_per_sample)

        offset += 8 + chunk_size
        if chunk_size % 2 == 1:
            offset += 1  # WAV chunks are word-aligned

    raise ValueError("data chunk not found in header bytes")


def _check_audio_duration(
    minio_client: Minio, bucket: str, s3_path: str, rule: Dict
) -> Optional[str]:
    min_dur = rule.get("min_duration_s")
    max_dur = rule.get("max_duration_s")
    if min_dur is None and max_dur is None:
        return None

    try:
        response = minio_client.get_object(bucket, s3_path, offset=0, length=256)
        header_bytes = response.read()
        response.close()
        response.release_conn()
        duration = _parse_wav_duration(header_bytes)
    except Exception as e:
        return f"Cannot read audio header: {e}"

    if min_dur is not None and duration < min_dur:
        return f"Duration {duration:.2f}s < min {min_dur}s"
    if max_dur is not None and duration > max_dur:
        return f"Duration {duration:.2f}s > max {max_dur}s"
    return None


def _check_video_size(size_in_minio: int, rule: Dict) -> Optional[str]:
    """Returns error message if video size is below threshold, else None."""
    min_size = rule.get("min_size_bytes")
    if min_size is not None and size_in_minio < min_size:
        return f"Object size {size_in_minio:,} bytes < min {min_size:,} bytes (truncation)"
    return None


# Core integrity check
def check_integrity(
    minio_client: Minio,
    entries: List[FileEntry],
    bucket: str,
    validation_cfg: Dict,
) -> Tuple[List[Dict], List[str]]:
    # Build inventory once (size check and presence check)
    prefixes = sorted({e.s3_path.split("/")[0] + "/" for e in entries})
    inventory: Dict[str, Dict] = {}
    for prefix in prefixes:
        try:
            for obj in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
                inventory[obj.object_name] = {"size": obj.size, "etag": obj.etag or ""}
        except Exception as e:
            logger.warning("Cannot list objects under %s/%s: %s", bucket, prefix, e)

    failures: List[Dict] = []
    faulty: List[str] = []

    for entry in entries:
        reasons = []

        # Check 1: object exists
        if entry.s3_path not in inventory:
            reasons.append("Object not found in MinIO")
        else:
            obj_info = inventory[entry.s3_path]
            minio_size = obj_info["size"]
            minio_etag = obj_info["etag"]

            # Check 2: size matches local source
            try:
                local_size = os.path.getsize(entry.local_path)
                if minio_size != local_size:
                    reasons.append(
                        f"Size mismatch: MinIO={minio_size:,}, local={local_size:,} bytes"
                    )
            except OSError as e:
                reasons.append(f"Cannot stat local file: {e}")

            # Check 3: MD5 hash vs MinIO ETag
            if not reasons:
                err = _check_hash(entry.local_path, minio_etag)
                if err:
                    reasons.append(err)

            # Check 4: content validation per type
            rule = _get_rule_for_entry(entry, validation_cfg)
            if rule and not reasons:
                file_type = rule.get("file_type")
                if file_type == "audio" and entry.file_name.lower().endswith(".wav"):
                    err = _check_audio_duration(minio_client, bucket, entry.s3_path, rule)
                    if err:
                        reasons.append(err)
                elif file_type == "video":
                    err = _check_video_size(minio_size, rule)
                    if err:
                        reasons.append(err)

        if reasons:
            logger.error("FAIL [%s] %s — %s", entry.dataset, entry.s3_path, "; ".join(reasons))
            failures.append({
                "s3_path": entry.s3_path,
                "local_path": entry.local_path,
                "dataset": entry.dataset,
                "entity": entry.entity_id,
                "reasons": reasons,
            })
            faulty.append(entry.s3_path)

    return failures, faulty


def delete_faulty_objects(
    minio_client: Minio, bucket: str, faulty_paths: List[str]
) -> List[str]:
    delete_errors = []
    for s3_path in faulty_paths:
        try:
            minio_client.remove_object(bucket, s3_path)
            logger.info("DELETED faulty object: %s", s3_path)
        except Exception as e:
            logger.error("Failed to delete %s: %s", s3_path, e)
            delete_errors.append(s3_path)
    return delete_errors


# Report helpers
def build_report(
    dataset: str,
    total_checked: int,
    failures: List[Dict],
    deleted: List[str],
    delete_errors: List[str],
) -> Dict[str, Any]:
    overall = "FAIL" if failures else "PASS"
    return {
        "stage": "validate_bronze_integrity",
        "dataset": dataset,
        "overall": overall,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "total_checked": total_checked,
            "passed": total_checked - len(failures),
            "failed": len(failures),
            "deleted_from_minio": len(deleted) - len(delete_errors),
            "delete_errors": len(delete_errors),
        },
        "failures": failures,
        "delete_errors": delete_errors,
    }


def save_report(report: Dict, output_dir: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"validate_bronze_integrity_{report['overall']}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report saved: %s", path)
    return str(path)


def print_summary(report: Dict) -> None:
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"  validate_bronze_integrity  |  {report['overall']}")
    print(f"{'='*60}")
    print(f"  Files checked       : {s['total_checked']}")
    print(f"  Passed              : {s['passed']}")
    print(f"  Failed              : {s['failed']}")
    print(f"  Deleted from MinIO  : {s['deleted_from_minio']}")
    if s["delete_errors"]:
        print(f"  Delete errors       : {s['delete_errors']}  (manual cleanup needed)")
    if report["failures"]:
        print(f"\n  Failures (first 10):")
        for f in report["failures"][:10]:
            print(f"    {f['s3_path']}")
            for r in f["reasons"]:
                print(f"      → {r}")
    print(f"{'='*60}\n")


# Entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bronze layer integrity check.")
    parser.add_argument("--dataset", choices=["eav", "kemocon", "all"], default="all",
                        help="Dataset to check (default: all)")
    parser.add_argument("--config", default="pipeline_config.yaml",
                        help="Path to pipeline_config.yaml")
    parser.add_argument("--output", default=".", help="Directory for JSON report output")
    parser.add_argument("--no-delete", action="store_true",
                        help="Do not delete faulty objects from MinIO (dry-run mode)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    validation_cfg = cfg.get("bronze_validation", {})
    bucket = _env("BUCKET_BRONZE")
    minio_client = _get_minio_client()

    all_entries: List[FileEntry] = []

    if args.dataset in ("eav", "all"):
        eav_root = _env("LOCAL_EAV_PATH")
        base_path_eav = _env("BASE_PATH_EAV")
        all_entries += _build_eav_manifest(eav_root, base_path_eav)

    if args.dataset in ("kemocon", "all"):
        kemocon_root = _env("LOCAL_KEMOCON_PATH")
        base_path_kemocon = _env("BASE_PATH_KEMOCON")
        all_entries += _build_kemocon_manifest(kemocon_root, base_path_kemocon)

    logger.info("Checking integrity of %d expected files in MinIO...", len(all_entries))

    failures, faulty_paths = check_integrity(minio_client, all_entries, bucket, validation_cfg)

    delete_errors: List[str] = []
    if faulty_paths:
        if args.no_delete:
            logger.warning("--no-delete: skipping deletion of %d faulty objects.", len(faulty_paths))
        else:
            logger.info("Deleting %d faulty objects from MinIO...", len(faulty_paths))
            delete_errors = delete_faulty_objects(minio_client, bucket, faulty_paths)

    report = build_report(
        dataset=args.dataset,
        total_checked=len(all_entries),
        failures=failures,
        deleted=faulty_paths,
        delete_errors=delete_errors,
    )
    print_summary(report)
    print(json.dumps(report, indent=2))
    save_report(report, args.output)

    sys.exit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
