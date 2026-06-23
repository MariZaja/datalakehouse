"""Validate and fix window counts in 04_quality_flags CSVs.

Expected window count = floor(debate_duration_s / 0.3):
  - EAV: floor(20.0 / 0.3) = 66 for every file
  - EAV EEG: rows are (window x channel) — filter on window_id
  - K-EmoCon: per-entity duration from subjects.csv

Files with more windows than expected are trimmed and re-uploaded.
Files with fewer windows are only reported (missing data cannot be recovered).
"""

import io
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as project_config
from minio_utils import download_object

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("check_qf_window_counts")

_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")
_QF_PREFIX = "04_quality_flags"
_WINDOW_SIZE_S = 0.3
_EAV_DURATION_S = 20.0


def _expected_windows(duration_s: float) -> int:
    return math.floor(duration_s / _WINDOW_SIZE_S)


def _load_kemocon_expected(minio_client, bucket: str, path: str) -> Dict[str, int]:
    """Return {entity_id -> expected_window_count} for every K-EmoCon participant."""
    data = download_object(minio_client, bucket, path)
    if data is None:
        logger.error("Cannot load subjects.csv from %s/%s", bucket, path)
        return {}
    df = pd.read_csv(io.BytesIO(data))
    result: Dict[str, int] = {}
    for _, row in df.iterrows():
        pid = int(row["pid"])
        duration_s = (int(row["endTime"]) - int(row["startTime"])) / 1000.0
        result[f"p{pid}"] = _expected_windows(duration_s)
    return result


def _count_windows(df: pd.DataFrame, is_eeg: bool) -> int:
    if is_eeg and "window_id" in df.columns:
        return int(df["window_id"].nunique())
    return len(df)


def _trim(df: pd.DataFrame, expected: int, is_eeg: bool) -> pd.DataFrame:
    """Return df with windows beyond expected removed."""
    if is_eeg and "window_id" in df.columns:
        return df[df["window_id"] < expected].reset_index(drop=True)
    return df.iloc[:expected].reset_index(drop=True)


def _upload_csv(minio_client, bucket: str, key: str, df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    minio_client.put_object(
        bucket, key,
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="text/csv",
    )


def main() -> None:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    bucket: str = cfg["bucket_silver"]

    qf_datasets = cfg.get("quality_flags", {}).get("datasets", {})
    kemocon_cfg = qf_datasets.get("kemocon", {})
    subjects_bucket = kemocon_cfg.get("subjects_bucket", "silver")
    subjects_path = kemocon_cfg.get(
        "subjects_path",
        "01_entity_resolution/k-emocon/auxiliary/metadata/metadata/subjects.csv",
    )

    kemocon_expected = _load_kemocon_expected(minio_client, subjects_bucket, subjects_path)
    eav_expected = _expected_windows(_EAV_DURATION_S)

    logger.info(
        "EAV expected: %d windows (floor(%.1fs / %.3fs))",
        eav_expected, _EAV_DURATION_S, _WINDOW_SIZE_S,
    )
    logger.info("K-EmoCon subjects loaded: %d", len(kemocon_expected))

    prefix = _QF_PREFIX.rstrip("/") + "/"
    csv_objects = [
        o for o in minio_client.list_objects(bucket, prefix=prefix, recursive=True)
        if o.object_name.endswith(".csv")
    ]
    logger.info("Found %d CSV files under %s", len(csv_objects), _QF_PREFIX)

    ok = 0
    trimmed = 0
    too_few: List[dict] = []
    errors: List[str] = []

    for obj in csv_objects:
        key = obj.object_name
        parts = key.split("/")

        dataset: Optional[str] = None
        entity_id: Optional[str] = None
        modality: Optional[str] = None
        for part in parts:
            if part in ("eav", "k-emocon"):
                dataset = part
            elif part.startswith("entity="):
                entity_id = part[len("entity="):]
            elif part.startswith("modality="):
                modality = part[len("modality="):]

        if dataset is None or entity_id is None:
            logger.warning("Cannot parse path — skipping: %s", key)
            continue

        if dataset == "eav":
            expected = eav_expected
        elif dataset == "k-emocon":
            expected = kemocon_expected.get(entity_id)
            if expected is None:
                logger.warning("[k-emocon] No subject entry for %s — skipping %s", entity_id, key)
                continue
        else:
            continue

        is_eeg = (dataset == "eav" and modality == "eeg")

        try:
            response = minio_client.get_object(bucket, key)
            raw = response.read()
            response.close()
            response.release_conn()
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            logger.error("Failed to read %s: %s", key, e)
            errors.append(key)
            continue

        actual = _count_windows(df, is_eeg)

        if actual == expected:
            ok += 1
        elif actual > expected:
            df_fixed = _trim(df, expected, is_eeg)
            try:
                _upload_csv(minio_client, bucket, key, df_fixed)
                trimmed += 1
                logger.info(
                    "TRIMMED [%s] %s | %s | %d -> %d windows",
                    dataset, entity_id, modality, actual, expected,
                )
            except Exception as e:
                logger.error("Failed to re-upload %s: %s", key, e)
                errors.append(key)
        else:
            too_few.append(dict(
                file=key, dataset=dataset, entity_id=entity_id,
                modality=modality, expected=expected, actual=actual,
            ))
            logger.warning(
                "TOO FEW [%s] %s | %s | expected=%d actual=%d",
                dataset, entity_id, modality, expected, actual,
            )

    total = ok + trimmed + len(too_few) + len(errors)
    logger.info("=" * 60)
    logger.info(
        "Result: %d ok, %d trimmed, %d too few, %d errors  (total %d files)",
        ok, trimmed, len(too_few), len(errors), total,
    )

    if too_few:
        logger.warning("Files with fewer windows than expected (not fixable here):")
        for iss in too_few:
            logger.warning(
                "  [%s] %-20s %-12s expected=%d actual=%d",
                iss["dataset"], iss["entity_id"], iss["modality"],
                iss["expected"], iss["actual"],
            )
    if errors:
        logger.error("Files that could not be processed:")
        for p in errors:
            logger.error("  %s", p)


if __name__ == "__main__":
    main()
