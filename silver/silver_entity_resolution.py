import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from pyspark.sql import Row
from pyspark.sql.functions import current_timestamp

import config as project_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("silver_entity_resolution")

def _pass(check: str, dataset: str, message: str) -> Dict[str, Any]:
    logger.info("[PASS] [%s] %s — %s", dataset, check, message)
    return {"check": check, "dataset": dataset, "status": "PASS", "message": message, "details": []}


def _fail(check: str, dataset: str, message: str, details: List[str] = None) -> Dict[str, Any]:
    logger.error("[FAIL] [%s] %s — %s", dataset, check, message)
    return {"check": check, "dataset": dataset, "status": "FAIL", "message": message, "details": details or []}


def _s3a(bucket: str, path: str) -> str:
    return f"s3a://{bucket}/{path}"


def to_canonical(number: int) -> str:
    return f"e{number:02d}"


def extract_number(raw_id: str, pattern: str) -> Optional[int]:
    m = re.match(pattern, raw_id, re.IGNORECASE)
    return int(m.group(1)) if m else None


def resolve_id(raw_id: str, id_pattern: str, dyadic_separator: Optional[str]) -> List[str]:
    """Return list of canonical IDs for a raw entity_id.

    Single:  "p3"     -> ["e03"]
    Dyadic:  "p1_p2"  -> ["e01", "e02"]
    """
    if dyadic_separator:
        parts = raw_id.split(dyadic_separator)
        if len(parts) >= 2 and all(
            re.match(id_pattern, p, re.IGNORECASE) for p in parts
        ):
            numbers = [extract_number(p, id_pattern) for p in parts]
            if all(n is not None for n in numbers):
                return [to_canonical(n) for n in numbers]

    number = extract_number(raw_id, id_pattern)
    if number is not None:
        return [to_canonical(number)]

    return []


def build_mapping(
    entity_ids: List[str], id_pattern: str, dyadic_separator: Optional[str]
) -> Dict[str, List[str]]:
    """Return {raw_id: [canonical_id, ...]} for every raw entity_id."""
    mapping = {}
    for raw in entity_ids:
        canonical_list = resolve_id(raw, id_pattern, dyadic_separator)
        if canonical_list:
            mapping[raw] = canonical_list
        else:
            logger.warning("Cannot map entity_id '%s' — no match for pattern '%s'", raw, id_pattern)
            mapping[raw] = []
    return mapping

# Phase 1 — ID audit
def audit_entity_ids(
    minio_client,
    bucket: str,
    dataset: str,
    files_prefix: str,
    id_pattern: str,
    dyadic_separator: Optional[str],
) -> Tuple[Dict[str, Any], List[str]]:
    """Scan entity= folders and classify every ID found.

    Returns (audit_dict, sorted_entity_id_list).
    """
    canonical_re = re.compile(r"^e\d{2}$")
    single_re = re.compile(id_pattern, re.IGNORECASE)

    prefix = files_prefix.rstrip("/") + "/"
    entity_ids = set()

    for obj in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
        # extract entity=XXX segment from path
        for segment in obj.object_name.split("/"):
            if segment.startswith("entity="):
                entity_ids.add(segment[len("entity="):])

    entity_ids = sorted(entity_ids)

    formats: Dict[str, List[str]] = {
        "canonical_eXX": [],
        "single_raw": [],
        "dyadic_raw": [],
        "unknown": [],
    }

    for eid in entity_ids:
        if canonical_re.match(eid):
            formats["canonical_eXX"].append(eid)
        elif dyadic_separator and dyadic_separator in eid:
            parts = eid.split(dyadic_separator)
            if len(parts) >= 2 and all(single_re.match(p) for p in parts):
                formats["dyadic_raw"].append(eid)
            else:
                formats["unknown"].append(eid)
        elif single_re.match(eid):
            formats["single_raw"].append(eid)
        else:
            formats["unknown"].append(eid)

    audit = {
        "dataset": dataset,
        "total_entity_ids": len(entity_ids),
        "formats": {k: {"count": len(v), "examples": v[:10]} for k, v in formats.items() if v},
        "dyadic_ids": formats["dyadic_raw"],
    }

    logger.info(
        "[%s] Audit — total=%d  canonical=%d  single_raw=%d  dyadic_raw=%d  unknown=%d",
        dataset,
        len(entity_ids),
        len(formats["canonical_eXX"]),
        len(formats["single_raw"]),
        len(formats["dyadic_raw"]),
        len(formats["unknown"]),
    )

    return audit, entity_ids

# Phase 2 — transformation
def copy_object(minio_client, src_bucket: str, src_key: str, dst_bucket: str, dst_key: str) -> None:
    from minio.commonconfig import CopySource
    minio_client.copy_object(
        dst_bucket,
        dst_key,
        CopySource(src_bucket, src_key),
    )


def transform_dataset(
    minio_client,
    spark,
    bronze_bucket: str,
    silver_bucket: str,
    dataset: str,
    dcfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Copy Bronze files to Silver with canonical entity IDs.

    Returns list of result dicts (PASS/FAIL per step).
    """
    results = []
    id_pattern = dcfg["id_pattern"]
    dyadic_separator = dcfg.get("dyadic_separator")
    bronze_files_prefix = dcfg["files_prefix"].rstrip("/") + "/"
    silver_base = dcfg["silver_base_prefix"]

    # --- collect all entity_ids from bronze Delta table ---
    try:
        bronze_df = (
            spark.read.format("delta")
            .load(_s3a(bronze_bucket, dcfg["files_table"]))
        )
        raw_entity_ids = [
            row[0] for row in bronze_df.select("entity_id").distinct().collect()
        ]
    except Exception as e:
        results.append(_fail("read_bronze_delta", dataset, f"Cannot read Bronze Delta: {e}"))
        return results

    mapping = build_mapping(raw_entity_ids, id_pattern, dyadic_separator)
    unmappable = [rid for rid, cids in mapping.items() if not cids]
    if unmappable:
        results.append(_fail(
            "build_mapping", dataset,
            f"{len(unmappable)} entity_id(s) could not be mapped.",
            [f"unmappable: {u}" for u in unmappable],
        ))
        return results

    logger.info("[%s] Mapping built — %d raw IDs → Silver", dataset, len(mapping))
    results.append(_pass("build_mapping", dataset, f"{len(mapping)} entity_id(s) mapped successfully."))

    # --- copy files ---
    copied = 0
    errors = []

    for obj in minio_client.list_objects(bronze_bucket, prefix=bronze_files_prefix, recursive=True):
        src_key = obj.object_name

        # extract entity_id from path segment "entity=XXX"
        raw_entity_id = None
        remaining_after_entity = None
        parts = src_key.split("/")
        for i, part in enumerate(parts):
            if part.startswith("entity="):
                raw_entity_id = part[len("entity="):]
                # everything after entity=XXX/ (modality=.../filename)
                remaining_after_entity = "/".join(parts[i + 1:])
                break

        if raw_entity_id is None:
            logger.warning("Cannot extract entity_id from path: %s — skipping", src_key)
            continue

        canonical_ids = mapping.get(raw_entity_id, [])
        if not canonical_ids:
            errors.append(f"No canonical ID for '{raw_entity_id}' at {src_key}")
            continue

        for canonical_id in canonical_ids:
            dst_key = f"{silver_base}/files/entity={canonical_id}/{remaining_after_entity}"
            try:
                copy_object(minio_client, bronze_bucket, src_key, silver_bucket, dst_key)
                copied += 1
            except Exception as e:
                errors.append(f"Copy failed {src_key} → {dst_key}: {e}")

    if errors:
        results.append(_fail(
            "copy_files", dataset,
            f"Copied {copied} file(s) with {len(errors)} error(s).",
            errors[:20],
        ))
        return results

    results.append(_pass("copy_files", dataset, f"Copied {copied} file(s) to Silver."))

    # --- write Silver Delta table ---
    try:
        silver_rows = []
        bronze_rows = bronze_df.collect()

        for row in bronze_rows:
            raw_eid = row["entity_id"]
            canonical_ids = mapping.get(raw_eid, [])
            for canonical_id in canonical_ids:
                old_path = row["file_path"]
                # replace bronze entity segment with canonical silver path
                new_path = re.sub(
                    r"entity=[^/]+",
                    f"entity={canonical_id}",
                    old_path.replace(
                        f"s3://{bronze_bucket}/{dcfg['base_prefix']}",
                        f"s3://{silver_bucket}/{silver_base}",
                        1,
                    ),
                )
                silver_rows.append(Row(
                    file_id=row["file_id"],
                    dataset=row["dataset"],
                    entity_id=canonical_id,
                    modality=row["modality"],
                    file_path=new_path,
                    file_name=row["file_name"],
                    ingestion_time=row["ingestion_time"],
                    source_entity_id=raw_eid,
                ))

        silver_df = spark.createDataFrame(silver_rows)

        (
            silver_df
            .withColumn("ingest_ts", current_timestamp())
            .write
            .format("delta")
            .mode("overwrite")
            .partitionBy("entity_id", "modality")
            .save(_s3a(silver_bucket, dcfg["silver_files_table"]))
        )

        results.append(_pass(
            "write_silver_delta", dataset,
            f"Silver Delta table written: {dcfg['silver_files_table']} ({len(silver_rows)} records).",
        ))

    except Exception as e:
        results.append(_fail("write_silver_delta", dataset, f"Cannot write Silver Delta: {e}"))

    return results


# Phase 3 — post-audit validation
def validate_post_transform(
    minio_client,
    bucket: str,
    dataset: str,
    silver_files_prefix: str,
) -> Dict[str, Any]:
    """Verify that ALL entity IDs in Silver are in canonical eXX format."""
    check = "post_transform_ids_canonical"
    canonical_re = re.compile(r"^e\d{2}$")

    prefix = silver_files_prefix.rstrip("/") + "/"
    non_canonical = set()

    for obj in minio_client.list_objects(bucket, prefix=prefix, recursive=True):
        for segment in obj.object_name.split("/"):
            if segment.startswith("entity="):
                eid = segment[len("entity="):]
                if not canonical_re.match(eid):
                    non_canonical.add(eid)

    if non_canonical:
        return _fail(
            check, dataset,
            f"{len(non_canonical)} non-canonical entity ID(s) found in Silver.",
            [f"non-canonical: {eid}" for eid in sorted(non_canonical)],
        )
    return _pass(check, dataset, "All entity IDs in Silver are in canonical eXX format.")


def build_report(
    phase1_audits: List[Dict[str, Any]],
    transform_results: List[Dict[str, Any]],
    phase3_audits: List[Dict[str, Any]],
    phase3_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    all_checks = transform_results + phase3_results
    fails = [r for r in all_checks if r["status"] == "FAIL"]
    overall = "FAIL" if fails else "PASS"
    return {
        "layer": "silver",
        "step": "01_entity_resolution",
        "overall": overall,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {
            "total": len(all_checks),
            "pass": len(all_checks) - len(fails),
            "fail": len(fails),
        },
        "phase1_id_audit_before": phase1_audits,
        "phase2_transform": transform_results,
        "phase3_id_audit_after": phase3_audits,
        "phase3_validation": phase3_results,
    }


def save_report(report: Dict[str, Any], output_dir: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"silver_entity_resolution_{report['overall']}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report saved: %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver — Step 1: Entity Resolution.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to YAML config.")
    parser.add_argument("--output", default=".", help="Directory for JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client, spark = project_config.config()

    bronze_bucket = cfg["bucket"]
    silver_bucket = cfg["bucket_silver"]

    common = cfg["datasets"].get("common", {})
    datasets = {k: {**common, **v} for k, v in cfg["datasets"].items() if k != "common"}

    phase1_audits = []
    transform_results = []
    phase3_audits = []
    phase3_results = []

    logger.info("PHASE 1 — ID Audit (Bronze, before transformation)")

    for dataset, dcfg in datasets.items():
        logger.info("--- %s ---", dataset)
        audit, _ = audit_entity_ids(
            minio_client,
            bronze_bucket,
            dataset,
            dcfg["files_prefix"],
            dcfg["id_pattern"],
            dcfg.get("dyadic_separator"),
        )
        phase1_audits.append(audit)

    logger.info("PHASE 2 — Transformation (Bronze → Silver)")

    for dataset, dcfg in datasets.items():
        logger.info("--- %s ---", dataset)
        results = transform_dataset(
            minio_client, spark,
            bronze_bucket, silver_bucket,
            dataset, dcfg,
        )
        transform_results.extend(results)

    logger.info("PHASE 3 — ID Audit (Silver, after transformation)")

    for dataset, dcfg in datasets.items():
        logger.info("--- %s ---", dataset)
        silver_files_prefix = f"{dcfg['silver_base_prefix']}/files/"
        audit, _ = audit_entity_ids(
            minio_client,
            silver_bucket,
            dataset,
            silver_files_prefix,
            dcfg["id_pattern"],
            dcfg.get("dyadic_separator"),
        )
        phase3_audits.append(audit)

        validation = validate_post_transform(
            minio_client,
            silver_bucket,
            dataset,
            silver_files_prefix,
        )
        phase3_results.append(validation)

    report = build_report(phase1_audits, transform_results, phase3_audits, phase3_results)
    print(json.dumps(report, indent=2))
    save_report(report, args.output)

    if report["overall"] == "FAIL":
        logger.error("Entity Resolution FAILED — check report for details.")
        sys.exit(1)

    logger.info("Entity Resolution PASSED — Silver layer ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
