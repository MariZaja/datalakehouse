import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv
from pyspark.sql.functions import countDistinct

import config as project_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pre_validate_bronze")

def _pass(check: str, dataset: str, message: str) -> Dict[str, Any]:
    logger.info("[PASS] [%s] %s — %s", dataset, check, message)
    return {"check": check, "dataset": dataset, "status": "PASS", "message": message, "details": []}


def _fail(check: str, dataset: str, message: str, details: List[str] = None) -> Dict[str, Any]:
    logger.error("[FAIL] [%s] %s — %s", dataset, check, message)
    return {"check": check, "dataset": dataset, "status": "FAIL", "message": message, "details": details or []}


def _s3(bucket: str, path: str) -> str:
    return f"s3a://{bucket}/{path}"

# Check 1 — bucket exists
def check_bucket_exists(minio_client, bucket: str) -> Dict[str, Any]:
    check = "bucket_exists"
    try:
        exists = minio_client.bucket_exists(bucket)
    except Exception as e:
        return _fail(check, "global", f"MinIO error: {e}")

    if not exists:
        return _fail(check, "global", f"Bucket '{bucket}' does not exist.")
    return _pass(check, "global", f"Bucket '{bucket}' exists.")

# Check 2 — expected prefixes exist
def check_expected_prefixes(minio_client, bucket: str, dataset: str, expected_prefixes: List[str]) -> Dict[str, Any]:
    check = "expected_prefixes"
    missing = []
    for prefix in expected_prefixes:
        p = prefix.rstrip("/") + "/"
        objects = list(minio_client.list_objects(bucket, prefix=p, recursive=False))
        if not objects:
            missing.append(prefix)

    if missing:
        return _fail(check, dataset, f"{len(missing)} expected prefix(es) missing.", [f"Missing: {p}" for p in missing])
    return _pass(check, dataset, f"All {len(expected_prefixes)} expected prefixes present.")

# Check 3 — no unknown top-level sub-prefixes
def check_no_unknown_prefixes(minio_client, bucket: str, dataset: str, base_prefix: str, allowed: List[str]) -> Dict[str, Any]:
    check = "no_unknown_prefixes"
    base = base_prefix.rstrip("/") + "/"
    found = set()
    for obj in minio_client.list_objects(bucket, prefix=base, recursive=False):
        relative = obj.object_name[len(base):]
        first = relative.split("/")[0]
        if first:
            found.add(first)

    unknown = sorted(found - set(allowed))
    if unknown:
        return _fail(check, dataset, f"Unknown sub-prefix(es) found under '{base}'.", [f"Unknown: {base}{u}/" for u in unknown])
    return _pass(check, dataset, f"No unknown sub-prefixes under '{base}'.")

# Check 4 — Delta tables readable
def check_delta_tables_readable(spark, bucket: str, dataset: str, files_table: str, aux_table: str) -> Dict[str, Any]:
    check = "delta_tables_readable"
    errors = []
    for table_path in [files_table, aux_table]:
        try:
            spark.read.format("delta").load(_s3(bucket, table_path)).limit(1).collect()
        except Exception as e:
            errors.append(f"{table_path}: {e}")

    if errors:
        return _fail(check, dataset, f"{len(errors)} Delta table(s) not readable.", errors)
    return _pass(check, dataset, "Both Delta tables readable.")

# Check 5 — Delta schema matches expected columns
def check_delta_schema(spark, bucket: str, dataset: str, files_table: str, aux_table: str,
                       files_expected_columns: List[str], aux_expected_columns: List[str]) -> Dict[str, Any]:
    check = "delta_schema"
    details = []

    for table_path, expected in [(files_table, files_expected_columns), (aux_table, aux_expected_columns)]:
        try:
            actual = set(spark.read.format("delta").load(_s3(bucket, table_path)).columns)
        except Exception as e:
            details.append(f"{table_path}: cannot read — {e}")
            continue
        missing = set(expected) - actual
        if missing:
            details += [f"{table_path}: missing column '{c}'" for c in sorted(missing)]

    if details:
        return _fail(check, dataset, "Schema mismatch in one or more Delta tables.", details)
    return _pass(check, dataset, "All expected columns present in both Delta tables.")

# Check 6 — entity count above minimum
def check_entity_count(spark, bucket: str, dataset: str, files_table: str,
                       entity_id_column: str, entity_count_min: int) -> Dict[str, Any]:
    check = "entity_count"
    try:
        actual = (
            spark.read.format("delta").load(_s3(bucket, files_table))
            .select(countDistinct(entity_id_column))
            .collect()[0][0]
        )
    except Exception as e:
        return _fail(check, dataset, f"Cannot count entities: {e}")

    if actual < entity_count_min:
        return _fail(check, dataset, f"Found {actual} entity/ies — expected >= {entity_count_min}.",
                     [f"actual={actual}, min={entity_count_min}"])
    return _pass(check, dataset, f"Entity count: {actual} (min={entity_count_min}).")

# Check 7 — all expected modalities have at least one file
def check_modalities_have_files(spark, bucket: str, dataset: str, files_table: str,
                                modality_column: str, expected_modalities: List[str]) -> Dict[str, Any]:
    check = "modalities_have_files"
    try:
        actual = {
            row[0]
            for row in spark.read.format("delta").load(_s3(bucket, files_table))
            .select(modality_column).distinct().collect()
        }
    except Exception as e:
        return _fail(check, dataset, f"Cannot read modalities: {e}")

    missing = sorted(set(expected_modalities) - actual)
    unknown = sorted(actual - set(expected_modalities))
    details = ([f"Missing modality: {m}" for m in missing] +
               [f"Unknown modality: {m}" for m in unknown])

    if missing or unknown:
        return _fail(check, dataset, f"Modality mismatch — missing={missing}, unknown={unknown}.", details)
    return _pass(check, dataset, f"All modalities present: {sorted(actual)}.")


def run_all_checks(minio_client, spark, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    bucket = cfg["bucket"]
    results = []

    result = check_bucket_exists(minio_client, bucket)
    results.append(result)
    if result["status"] == "FAIL":
        logger.error("Bucket missing — skipping all remaining checks.")
        return results

    for dataset, dcfg in cfg["datasets"].items():
        logger.info("--- %s ---", dataset)
        results += [
            check_expected_prefixes(minio_client, bucket, dataset, dcfg["expected_prefixes"]),
            check_no_unknown_prefixes(minio_client, bucket, dataset, dcfg["base_prefix"], dcfg["allowed_sub_prefixes"]),
            check_delta_tables_readable(spark, bucket, dataset, dcfg["files_table"], dcfg["aux_table"]),
            check_delta_schema(spark, bucket, dataset, dcfg["files_table"], dcfg["aux_table"],
                               dcfg["files_expected_columns"], dcfg["aux_expected_columns"]),
            check_entity_count(spark, bucket, dataset, dcfg["files_table"],
                               dcfg["entity_id_column"], dcfg["entity_count_min"]),
            check_modalities_have_files(spark, bucket, dataset, dcfg["files_table"],
                                        dcfg["modality_column"], dcfg["expected_modalities"]),
        ]

    return results

# Report
def build_report(results: List[Dict[str, Any]], layer: str) -> Dict[str, Any]:
    fails = [r for r in results if r["status"] == "FAIL"]
    overall = "FAIL" if fails else "PASS"
    return {
        "layer": layer,
        "overall": overall,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": {"total": len(results), "pass": len(results) - len(fails), "fail": len(fails)},
        "results": results,
    }


def save_report(report: Dict[str, Any], output_dir: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"pre_validate_bronze_{report['overall']}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report saved: %s", path)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-Validation Gate — Bronze layer.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to YAML config.")
    parser.add_argument("--output", default=".", help="Directory for JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client, spark = project_config.config()

    results = run_all_checks(minio_client, spark, cfg)
    report = build_report(results, layer="bronze")

    print(json.dumps(report, indent=2))
    save_report(report, args.output)

    if report["overall"] == "FAIL":
        logger.error("Pre-validation FAILED — pipeline should not proceed.")
        sys.exit(1)

    logger.info("Pre-validation PASSED — pipeline may proceed.")
    sys.exit(0)


if __name__ == "__main__":
    main()