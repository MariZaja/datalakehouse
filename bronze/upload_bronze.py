import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from minio import Minio

from bronze_common import (
    FileEntry,
    build_eav_manifest,
    build_kemocon_manifest,
    env,
    get_minio_client,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("upload_bronze")


# MinIO inventory fetch (one list_objects call per top-level prefix)
def _fetch_inventory(minio_client: Minio, bucket: str, prefixes: List[str]) -> Dict[str, int]:
    inventory: Dict[str, int] = {}
    for prefix in prefixes:
        try:
            for obj in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
                inventory[obj.object_name] = obj.size
        except Exception as e:
            logger.warning("Could not list objects under %s/%s: %s", bucket, prefix, e)
    return inventory


# Upload with idempotency
def _upload_entries(
    minio_client: Minio,
    entries: List[FileEntry],
    bucket: str,
) -> Tuple[List[FileEntry], List[FileEntry], List[Dict]]:
    prefixes = sorted({e.s3_path.split("/")[0] + "/" for e in entries})
    inventory = _fetch_inventory(minio_client, bucket, prefixes)

    uploaded: List[FileEntry] = []
    skipped: List[FileEntry] = []
    failed: List[Dict] = []

    for entry in entries:
        try:
            local_size = os.path.getsize(entry.local_path)
        except OSError as e:
            logger.error("Cannot stat local file %s: %s", entry.local_path, e)
            failed.append({"path": entry.s3_path, "local": entry.local_path, "error": str(e)})
            continue

        existing_size = inventory.get(entry.s3_path)
        if existing_size is not None and existing_size == local_size:
            logger.debug("SKIP  %s (size=%d already in MinIO)", entry.s3_path, existing_size)
            skipped.append(entry)
            continue

        try:
            minio_client.fput_object(bucket, entry.s3_path, entry.local_path)
            logger.info("UPLOAD %s (%d bytes)", entry.s3_path, local_size)
            uploaded.append(entry)
        except Exception as e:
            logger.error("FAIL  %s: %s", entry.s3_path, e)
            failed.append({"path": entry.s3_path, "local": entry.local_path, "error": str(e)})

    return uploaded, skipped, failed


# Delta table writes (Spark — only started if new records exist)
def _write_files_delta(spark, entries: List[FileEntry], bucket: str,
                       base_path: str, delta_table_name: str) -> None:
    from pyspark.sql.functions import current_timestamp
    records = [
        (str(uuid.uuid4()), e.dataset, e.entity_id, e.modality,
         f"s3://{bucket}/{e.s3_path}", e.file_name, datetime.utcnow())
        for e in entries
    ]
    df = spark.createDataFrame(records, [
        "file_id", "dataset", "entity_id", "modality",
        "file_path", "file_name", "ingestion_time",
    ])
    (
        df.withColumn("ingest_ts", current_timestamp())
        .write.format("delta").mode("append")
        .partitionBy("entity_id", "modality")
        .save(f"s3a://{bucket}/{base_path}/delta/{delta_table_name}")
    )
    logger.info("Wrote %d records to delta/%s", len(records), delta_table_name)


def _write_aux_delta(spark, entries: List[FileEntry], bucket: str, base_path: str) -> None:
    from pyspark.sql.functions import current_timestamp
    records = [
        (str(uuid.uuid4()), e.dataset, e.modality,
         f"s3://{bucket}/{e.s3_path}", e.file_name, "global", datetime.utcnow())
        for e in entries
    ]
    df = spark.createDataFrame(records, [
        "record_id", "dataset", "category",
        "source_path", "file_name", "related_entity", "ingestion_time",
    ])
    (
        df.withColumn("ingest_ts", current_timestamp())
        .write.format("delta").mode("append")
        .partitionBy("dataset", "category")
        .save(f"s3a://{bucket}/{base_path}/delta/auxiliary_metadata")
    )
    logger.info("Wrote %d records to delta/auxiliary_metadata", len(records))


def _write_delta_tables(
    uploaded: List[FileEntry],
    bucket: str,
    base_path_eav: Optional[str],
    base_path_kemocon: Optional[str],
) -> None:
    import config as project_config
    _, spark = project_config.config()

    eav_files = [e for e in uploaded if e.dataset == "EAV" and e.table == "files"]
    eav_aux = [e for e in uploaded if e.dataset == "EAV" and e.table == "aux"]
    kemo_files = [e for e in uploaded if e.dataset == "K-EmoCon" and e.table == "files"]
    kemo_aux = [e for e in uploaded if e.dataset == "K-EmoCon" and e.table == "aux"]

    if eav_files and base_path_eav:
        _write_files_delta(spark, eav_files, bucket, base_path_eav, "eav_files_metadata")
    if eav_aux and base_path_eav:
        _write_aux_delta(spark, eav_aux, bucket, base_path_eav)
    if kemo_files and base_path_kemocon:
        _write_files_delta(spark, kemo_files, bucket, base_path_kemocon, "k_emocon_files_metadata")
    if kemo_aux and base_path_kemocon:
        _write_aux_delta(spark, kemo_aux, bucket, base_path_kemocon)


# Main upload orchestration
def run_upload(dataset: str) -> Dict[str, Any]:
    bucket = env("BUCKET_BRONZE")
    base_path_eav: Optional[str] = None
    base_path_kemocon: Optional[str] = None
    all_entries: List[FileEntry] = []

    if dataset in ("eav", "all"):
        eav_root = env("LOCAL_EAV_PATH")
        base_path_eav = env("BASE_PATH_EAV")
        all_entries += build_eav_manifest(eav_root, base_path_eav)
        logger.info("EAV manifest: %d files", sum(1 for e in all_entries if e.dataset == "EAV"))

    if dataset in ("kemocon", "all"):
        kemocon_root = env("LOCAL_KEMOCON_PATH")
        base_path_kemocon = env("BASE_PATH_KEMOCON")
        all_entries += build_kemocon_manifest(kemocon_root, base_path_kemocon)
        logger.info("K-EmoCon manifest: %d files",
                    sum(1 for e in all_entries if e.dataset == "K-EmoCon"))

    logger.info("Total manifest: %d files", len(all_entries))

    minio_client = get_minio_client()
    uploaded, skipped, failed = _upload_entries(minio_client, all_entries, bucket)

    logger.info("Upload complete — uploaded=%d, skipped=%d, failed=%d",
                len(uploaded), len(skipped), len(failed))

    # Write Delta records only for newly uploaded files
    if uploaded:
        logger.info("Writing Delta table records for %d new files...", len(uploaded))
        try:
            _write_delta_tables(uploaded, bucket, base_path_eav, base_path_kemocon)
        except Exception as e:
            logger.error("Delta table write failed: %s", e)
            # Add all as failed delta writes (files are in MinIO, just metadata missing)
            for entry in uploaded:
                failed.append({
                    "path": entry.s3_path,
                    "local": entry.local_path,
                    "error": f"Delta write failed: {e}",
                })
    else:
        logger.info("No new files uploaded — skipping Spark/Delta startup.")

    total_in_bronze = len(uploaded) + len(skipped)
    return {
        "stage": "upload",
        "dataset": dataset,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "total_manifest": len(all_entries),
            "uploaded": len(uploaded),
            "skipped_already_present": len(skipped),
            "failed": len(failed),
            "total_in_bronze": total_in_bronze,
        },
        "failures": failed,
    }


# Report helpers
def save_report(report: Dict, output_dir: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    overall = "PASS" if report["summary"]["failed"] == 0 else (
        "FAIL" if report["summary"]["total_in_bronze"] == 0 else "PARTIAL"
    )
    path = Path(output_dir) / f"upload_bronze_{overall}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report saved: %s", path)
    return str(path)


def print_summary(report: Dict) -> None:
    s = report["summary"]
    total_in_bronze = s["total_in_bronze"]
    status = "PASS" if s["failed"] == 0 else (
        "FATAL" if total_in_bronze == 0 else "PARTIAL"
    )
    print(f"\n{'='*60}")
    print(f"  upload_bronze  |  {status}")
    print(f"{'='*60}")
    print(f"  Manifest total    : {s['total_manifest']}")
    print(f"  Uploaded (new)    : {s['uploaded']}")
    print(f"  Skipped (present) : {s['skipped_already_present']}")
    print(f"  Failed            : {s['failed']}")
    print(f"  Total in bronze   : {total_in_bronze}")
    if report["failures"]:
        print(f"\n  Failed files (first 10):")
        for f in report["failures"][:10]:
            print(f"    {f['path']} — {f['error']}")
    print(f"{'='*60}\n")


# Entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Idempotent bronze upload stage.")
    parser.add_argument("--dataset", choices=["eav", "kemocon", "all"], default="all",
                        help="Dataset to upload (default: all)")
    parser.add_argument("--output", default=".", help="Directory for JSON report output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_upload(args.dataset)
    print_summary(report)
    print(json.dumps(report, indent=2))
    save_report(report, args.output)

    total_in_bronze = report["summary"]["total_in_bronze"]
    failed = report["summary"]["failed"]

    if total_in_bronze == 0:
        logger.error("FATAL: zero files in bronze after upload run.")
        sys.exit(1)
    if failed > 0:
        logger.warning("Partial failure: %d files failed, %d in bronze.", failed, total_in_bronze)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
