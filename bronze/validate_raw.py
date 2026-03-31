import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("validate_raw")


def _validate_audio(path: str, rule: Dict) -> List[Dict]:
    issues = []
    try:
        import soundfile as sf
        info = sf.info(path)
        duration = info.frames / info.samplerate
        sr = info.samplerate
    except Exception as e:
        return [{"level": "ERROR", "msg": f"Cannot open audio file: {e}"}]

    min_dur = rule.get("min_duration_s")
    max_dur = rule.get("max_duration_s")
    expected_sr = rule.get("expected_sr")

    if min_dur is not None and duration < min_dur:
        issues.append({"level": "ERROR", "msg": f"Duration {duration:.2f}s < min {min_dur}s, info: {info}"})
    if max_dur is not None and duration > max_dur:
        issues.append({"level": "ERROR", "msg": f"Duration {duration:.2f}s > max {max_dur}s"})
    if expected_sr is not None and sr != expected_sr:
        issues.append({"level": "ERROR", "msg": f"Sample rate {sr} Hz != expected {expected_sr} Hz"})
    return issues


def _validate_video(path: str, rule: Dict) -> List[Dict]:
    size = os.path.getsize(path)
    min_size = rule.get("min_size_bytes")
    if min_size is not None and size < min_size:
        return [{"level": "ERROR",
                 "msg": f"File size {size:,} bytes < min {min_size:,} bytes (possible truncation)"}]
    return []


_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _detect_mat_format(path: str):
    try:
        with open(path, "rb") as f:
            header = f.read(136)
    except OSError:
        return "scipy"

    if header[:8] == _HDF5_MAGIC:
        return "hdf5"
    if header[:4] == b"MATL" and header[128:136] == _HDF5_MAGIC:
        return "matlab_v73"
    return "scipy"


def _open_mat_h5py(path: str) -> List[Dict]:
    try:
        import h5py
        with h5py.File(path, "r") as f:
            keys = [k for k in f.keys() if not k.startswith("#")]
            if not keys:
                return [{"level": "ERROR", "msg": "HDF5 .mat file has no data datasets"}]
        return []
    except Exception as e:
        return [{"level": "ERROR", "msg": f"Cannot open HDF5 .mat file: {e}"}]


def _validate_mat(path: str, rule: Dict) -> List[Dict]:
    fmt = _detect_mat_format(path)

    if fmt in ("hdf5", "matlab_v73"):
        return _open_mat_h5py(path)

    try:
        import scipy.io
        variables = scipy.io.whosmat(path)
        data_vars = [(n, s, d) for n, s, d in variables if not n.startswith("_")]
        if not data_vars:
            return [{"level": "ERROR", "msg": "No data variables found in .mat file"}]
        return []
    except Exception as e:
        return [{"level": "ERROR", "msg": f"Cannot open .mat file: {e}"}]


def _validate_csv(path: str, rule: Dict) -> List[Dict]:
    try:
        import pandas as pd
        pd.read_csv(path, nrows=1)
        return []
    except Exception as e:
        return [{"level": "ERROR", "msg": f"Cannot open CSV: {e}"}]


def _validate_xlsx(path: str, rule: Dict) -> List[Dict]:
    try:
        import pandas as pd
        pd.read_excel(path, nrows=1)
        return []
    except Exception as e:
        return [{"level": "ERROR", "msg": f"Cannot open xlsx: {e}"}]


_VALIDATORS = {
    "audio": _validate_audio,
    "video": _validate_video,
    "mat": _validate_mat,
    "csv": _validate_csv,
    "xlsx": _validate_xlsx,
}


def validate_file(path: str, rule: Dict) -> List[Dict]:
    if not os.path.isfile(path):
        return [{"level": "ERROR", "msg": "File not found"}]
    if os.path.getsize(path) == 0:
        return [{"level": "ERROR", "msg": "Zero-byte file"}]

    validator = _VALIDATORS.get(rule.get("file_type"))
    return validator(path, rule) if validator else []


# File collection helpers
def collect_eav_files(local_root: str, dataset_cfg: Dict) -> Tuple[List[Tuple], Dict]:
    files = []
    missing: Dict[str, List[str]] = {}
    subject_pat = re.compile(dataset_cfg.get("subject_pattern", r"^subject\d+$"))
    modalities_cfg = dataset_cfg.get("modalities", {})

    for item in sorted(os.listdir(local_root)):
        subject_path = os.path.join(local_root, item)
        if not subject_pat.match(item) or not os.path.isdir(subject_path):
            continue

        for mod_dir, rule in modalities_cfg.items():
            src_dir = os.path.join(subject_path, mod_dir)
            if not os.path.isdir(src_dir):
                missing.setdefault(item, []).append(mod_dir)
                continue
            extensions = [e.lower() for e in rule.get("extensions", [])]
            for file_name in sorted(os.listdir(src_dir)):
                if file_name.startswith("."):
                    continue
                if extensions and not any(file_name.lower().endswith(e) for e in extensions):
                    continue
                files.append((os.path.join(src_dir, file_name), item, mod_dir, rule))

    return files, missing


def collect_kemocon_files(local_root: str, dataset_cfg: Dict) -> List[Tuple]:
    files = []

    for dir_key, rule in dataset_cfg.get("flat_dirs", {}).items():
        src_dir = os.path.join(local_root, dir_key)
        if not os.path.isdir(src_dir):
            continue
        extensions = [e.lower() for e in rule.get("extensions", [])]
        for file_name in sorted(os.listdir(src_dir)):
            if file_name.startswith("."):
                continue
            if extensions and not any(file_name.lower().endswith(e) for e in extensions):
                continue
            files.append((os.path.join(src_dir, file_name), dir_key, "flat", rule))

    for dir_key, rule in dataset_cfg.get("participant_dirs", {}).items():
        src_dir = os.path.join(local_root, dir_key)
        if not os.path.isdir(src_dir):
            continue
        extensions = [e.lower() for e in rule.get("extensions", [])]
        for participant_id in sorted(os.listdir(src_dir)):
            part_dir = os.path.join(src_dir, participant_id)
            if not os.path.isdir(part_dir) or participant_id.startswith("."):
                continue
            for file_name in sorted(os.listdir(part_dir)):
                if file_name.startswith("."):
                    continue
                if extensions and not any(file_name.lower().endswith(e) for e in extensions):
                    continue
                files.append((os.path.join(part_dir, file_name), dir_key, participant_id, rule))

    for dir_key, rule in dataset_cfg.get("auxiliary_dirs", {}).items():
        full_dir = os.path.join(local_root, dir_key)
        if not os.path.isdir(full_dir):
            continue
        extensions = [e.lower() for e in rule.get("extensions", [])]
        for root, _, fnames in os.walk(full_dir):
            for file_name in sorted(fnames):
                if file_name.startswith("."):
                    continue
                if extensions and not any(file_name.lower().endswith(e) for e in extensions):
                    continue
                files.append((os.path.join(root, file_name), dir_key, "auxiliary", rule))

    return files


# Dataset-level validation
def validate_dataset(dataset_key: str, dataset_cfg: Dict) -> Dict[str, Any]:
    local_root_env = dataset_cfg.get("local_root_env", "")
    local_root = os.getenv(local_root_env, "") if local_root_env else ""

    if not local_root or not os.path.isdir(local_root):
        return {
            "dataset": dataset_key,
            "status": "FAIL",
            "error": f"Local root not found: {local_root!r} (env var: {local_root_env})",
            "total_files": 0,
            "errors": 1,
            "warnings": 0,
            "missing_modalities_info": {},
            "file_issues": [],
        }

    missing_modalities: Dict[str, List[str]] = {}
    if "modalities" in dataset_cfg:
        all_files, missing_modalities = collect_eav_files(local_root, dataset_cfg)
    else:
        all_files = collect_kemocon_files(local_root, dataset_cfg)

    aux_rel = dataset_cfg.get("auxiliary_root_relative")
    aux_root = os.path.normpath(os.path.join(local_root, aux_rel)) if aux_rel else local_root
    for aux_cfg in dataset_cfg.get("auxiliary_files", []):
        fpath = os.path.join(aux_root, aux_cfg["filename"])
        if os.path.exists(fpath):
            all_files.append((fpath, "global", "auxiliary", aux_cfg))
        else:
            logger.info("[%s] Optional auxiliary file not found, skipping: %s",
                        dataset_key, aux_cfg["filename"])

    error_count = 0
    warn_count = 0
    file_issues = []

    for path, entity_id, source_key, rule in all_files:
        issues = validate_file(path, rule)
        if not issues:
            continue
        for issue in issues:
            if issue["level"] == "ERROR":
                error_count += 1
                logger.error("[%s] %s | %s: %s", dataset_key, entity_id, path, issue["msg"])
            else:
                warn_count += 1
                logger.warning("[%s] %s | %s: %s", dataset_key, entity_id, path, issue["msg"])
        file_issues.append({
            "path": path,
            "entity": entity_id,
            "source": source_key,
            "issues": issues,
        })

    for entity_id, missing in sorted(missing_modalities.items()):
        logger.info("[%s] %s: missing modalities (not an error) — %s",
                    dataset_key, entity_id, missing)

    status = "FAIL" if error_count > 0 else "PASS"
    logger.info("[%s] %s — scanned %d files, %d errors, %d warnings",
                dataset_key, status, len(all_files), error_count, warn_count)

    return {
        "dataset": dataset_key,
        "status": status,
        "total_files": len(all_files),
        "errors": error_count,
        "warnings": warn_count,
        "missing_modalities_info": missing_modalities,
        "file_issues": file_issues,
    }


# Report helpers
def build_report(dataset_results: List[Dict], datasets_requested: List[str]) -> Dict[str, Any]:
    overall = "FAIL" if any(r["status"] == "FAIL" for r in dataset_results) else "PASS"
    total_errors = sum(r.get("errors", 0) for r in dataset_results)
    total_warnings = sum(r.get("warnings", 0) for r in dataset_results)
    total_files = sum(r.get("total_files", 0) for r in dataset_results)
    return {
        "stage": "validate_raw",
        "overall": overall,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "datasets_requested": datasets_requested,
        "summary": {
            "total_files_scanned": total_files,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
        },
        "datasets": dataset_results,
    }


def save_report(report: Dict, output_dir: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(output_dir) / f"validate_raw_{report['overall']}_{ts}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report saved: %s", path)
    return str(path)


def print_summary(report: Dict) -> None:
    overall = report["overall"]
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"  validate_raw  |  {overall}")
    print(f"{'='*60}")
    print(f"  Files scanned : {s['total_files_scanned']}")
    print(f"  Errors        : {s['total_errors']}")
    print(f"  Warnings      : {s['total_warnings']}")
    for ds in report["datasets"]:
        mm = ds.get("missing_modalities_info", {})
        mm_summary = f"  ({len(mm)} entities with missing modalities — expected)" if mm else ""
        print(f"  [{ds['status']:4s}] {ds['dataset']} — {ds.get('total_files', 0)} files, "
              f"{ds.get('errors', 0)} errors, {ds.get('warnings', 0)} warnings{mm_summary}")
    print(f"{'='*60}\n")


# Entry point
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-upload raw data validation.")
    parser.add_argument("--dataset", choices=["eav", "kemocon", "all"], default="all",
                        help="Dataset to validate (default: all)")
    parser.add_argument("--config", default="pipeline_config.yaml",
                        help="Path to pipeline_config.yaml")
    parser.add_argument("--output", default=".", help="Directory for JSON report output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    validation_cfg = cfg.get("bronze_validation", {})
    all_datasets = validation_cfg.get("datasets", {})

    if args.dataset == "all":
        datasets_to_run = list(all_datasets.keys())
    else:
        datasets_to_run = [args.dataset]

    results = []
    for ds_key in datasets_to_run:
        if ds_key not in all_datasets:
            logger.error("Dataset %r not found in bronze_validation config", ds_key)
            results.append({"dataset": ds_key, "status": "FAIL",
                             "error": "Not in config", "total_files": 0,
                             "errors": 1, "warnings": 0,
                             "missing_modalities_info": {}, "file_issues": []})
            continue
        results.append(validate_dataset(ds_key, all_datasets[ds_key]))

    report = build_report(results, datasets_to_run)
    print_summary(report)
    print(json.dumps(report, indent=2))
    save_report(report, args.output)

    sys.exit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
