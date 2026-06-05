"""Attention & Meditation Value Statistics.

Reads raw Attention.csv and Meditation.csv for every K-EmoCon entity from MinIO,
computes per-entity value distribution and run-length statistics, and uploads two
result CSVs (one per signal) to MinIO.

Output columns per entity:
  entity_id, count_0, count_1_20, count_21_40, count_41_60, count_61_80,
  count_81_99, count_100, longest_run, longest_run_value, n_runs_over_10
"""

import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as project_config
from minio_utils import _group_objects_by_entity, download_object, upload_csv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("attention_meditation_stats")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent / "pipeline_config.yaml")
_DEFAULT_OUTPUT_PREFIX = "attention_meditation_stats"

_EMPTY_STATS: Dict[str, Any] = {
    "count_0": None,
    "count_1_20": None,
    "count_21_40": None,
    "count_41_60": None,
    "count_61_80": None,
    "count_81_99": None,
    "count_100": None,
    "longest_run": None,
    "longest_run_value": None,
    "n_runs_over_10": None,
    "jumps_over_10": None,
    "jumps_over_20": None,
    "jumps_over_30": None,
    "jumps_over_40": None,
}


def _compute_run_stats(values: np.ndarray) -> Tuple[int, float, int]:
    """Return (longest_run_length, longest_run_value, n_runs_over_10).

    A 'run' is an uninterrupted sequence of the same integer value.
    n_runs_over_10 counts distinct runs whose length exceeds 10.
    """
    if len(values) == 0:
        return 0, float("nan"), 0

    longest = 1
    longest_val = values[0]
    cur_len = 1
    cur_val = values[0]
    n_runs_over_10 = 0

    for v in values[1:]:
        if v == cur_val:
            cur_len += 1
        else:
            if cur_len > 10:
                n_runs_over_10 += 1
            if cur_len > longest:
                longest = cur_len
                longest_val = cur_val
            cur_val = v
            cur_len = 1

    # flush last run
    if cur_len > 10:
        n_runs_over_10 += 1
    if cur_len > longest:
        longest = cur_len
        longest_val = cur_val

    return longest, float(longest_val), n_runs_over_10


def _compute_stats(values: np.ndarray) -> Dict[str, Any]:
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return dict(_EMPTY_STATS)

    longest_run, longest_run_value, n_runs_over_10 = _compute_run_stats(vals)

    diffs = np.abs(np.diff(vals))

    return {
        "count_0":     int(np.sum(vals == 0)),
        "count_1_20":  int(np.sum((vals >= 1)  & (vals <= 20))),
        "count_21_40": int(np.sum((vals >= 21) & (vals <= 40))),
        "count_41_60": int(np.sum((vals >= 41) & (vals <= 60))),
        "count_61_80": int(np.sum((vals >= 61) & (vals <= 80))),
        "count_81_99": int(np.sum((vals >= 81) & (vals <= 99))),
        "count_100":   int(np.sum(vals == 100)),
        "longest_run":       longest_run,
        "longest_run_value": longest_run_value,
        "n_runs_over_10":    n_runs_over_10,
        "jumps_over_10": int(np.sum(diffs > 10)),
        "jumps_over_20": int(np.sum(diffs > 20)),
        "jumps_over_30": int(np.sum(diffs > 30)),
        "jumps_over_40": int(np.sum(diffs > 40)),
    }


def _read_values(data: bytes) -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(io.BytesIO(data))
        if "value" not in df.columns:
            logger.warning("CSV has no 'value' column. Available: %s", list(df.columns))
            return None
        return pd.to_numeric(df["value"], errors="coerce").values
    except Exception as exc:
        logger.warning("CSV read error: %s", exc)
        return None


def _process_signal(
    entity_id: str,
    signal_filename: str,
    obj_by_fname: Dict[str, Any],
    minio_client,
    bucket: str,
) -> Dict[str, Any]:
    base: Dict[str, Any] = {"entity_id": entity_id}

    obj = obj_by_fname.get(signal_filename)
    if obj is None:
        logger.warning("[%s] %s not found", entity_id, signal_filename)
        return {**base, **_EMPTY_STATS}

    data = download_object(minio_client, bucket, obj.object_name)
    if data is None:
        return {**base, **_EMPTY_STATS}

    values = _read_values(data)
    if values is None:
        return {**base, **_EMPTY_STATS}

    stats = _compute_stats(values)
    logger.info("[%s] %-15s %d values, longest_run=%d (val=%.0f), runs>10=%d",
                entity_id, signal_filename,
                len(values[np.isfinite(values)]),
                stats["longest_run"], stats["longest_run_value"] or 0,
                stats["n_runs_over_10"])
    return {**base, **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention & Meditation Value Statistics.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument("--output-prefix", default=_DEFAULT_OUTPUT_PREFIX,
                        help="MinIO prefix for output CSVs (default: attention_meditation_stats).")
    parser.add_argument("--test", action="store_true",
                        help="Process only the first 3 entities (smoke test).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    silver_bucket = cfg["bucket_silver"]

    kemocon_md = cfg.get("missingness_detection", {}).get("datasets", {}).get("kemocon", {})
    silver_files_prefix = kemocon_md.get("silver_files_prefix", "01_entity_resolution/k-emocon/files")

    entity_objects = _group_objects_by_entity(minio_client, silver_bucket, silver_files_prefix)
    entity_ids = sorted(entity_objects.keys())
    logger.info("Found %d K-EmoCon entities", len(entity_ids))

    if args.test:
        entity_ids = entity_ids[:3]
        logger.info("TEST MODE: %s", entity_ids)

    attention_rows: List[Dict[str, Any]] = []
    meditation_rows: List[Dict[str, Any]] = []

    for entity_id in entity_ids:
        obj_by_fname = {
            obj.object_name.split("/")[-1]: obj
            for obj in entity_objects[entity_id]
        }
        attention_rows.append(
            _process_signal(entity_id, "Attention.csv", obj_by_fname, minio_client, silver_bucket)
        )
        meditation_rows.append(
            _process_signal(entity_id, "Meditation.csv", obj_by_fname, minio_client, silver_bucket)
        )

    output_prefix = args.output_prefix.rstrip("/")
    attn_key = f"{output_prefix}/attention_stats.csv"
    medi_key = f"{output_prefix}/meditation_stats.csv"

    attention_df = pd.DataFrame(attention_rows)
    meditation_df = pd.DataFrame(meditation_rows)

    upload_csv(minio_client, silver_bucket, attn_key, attention_df)
    logger.info("Attention stats → silver/%s  (%d entities)", attn_key, len(attention_df))

    upload_csv(minio_client, silver_bucket, medi_key, meditation_df)
    logger.info("Meditation stats → silver/%s  (%d entities)", medi_key, len(meditation_df))


if __name__ == "__main__":
    main()
