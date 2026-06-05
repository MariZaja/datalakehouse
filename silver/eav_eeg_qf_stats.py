"""EAV EEG Quality Flags Statistics.

Reads all EAV EEG quality flag CSVs from MinIO and computes:
  1. Global rNSR distribution  — percentiles [50, 75, 90, 95, 99]
  2. Global rNSR_zscore distribution — percentiles [50, 75, 90, 95, 99]
  3. Bad rate per entity       — % non-GOOD windows per entity
  4. Bad rate per channel      — % non-GOOD windows per channel, macro-averaged across entities

Uploads four result CSVs to MinIO under --output-prefix.
"""

import argparse
import io
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as project_config
from minio_utils import upload_csv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("eav_eeg_qf_stats")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")
_DEFAULT_OUTPUT_PREFIX = "eav_eeg_qf_stats"
_PERCENTILES = [50, 75, 90, 95, 99]


def _read_csv(minio_client, bucket: str, key: str) -> Optional[pd.DataFrame]:
    try:
        response = minio_client.get_object(bucket, key)
        data = response.read()
        response.close()
        response.release_conn()
    except Exception as e:
        logger.error("Download failed %s: %s", key, e)
        return None
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception as e:
        logger.error("Parse failed %s: %s", key, e)
        return None


def _entity_from_path(obj_name: str) -> Optional[str]:
    for part in obj_name.split("/"):
        if part.startswith("entity="):
            return part[len("entity="):]
    return None


def run_stats(
    minio_client,
    bucket: str,
    qf_prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (rNSR_pct_df, rNSR_zscore_pct_df, bad_rate_entity_df, bad_rate_channel_df)."""

    eeg_prefix = f"{qf_prefix}/eav/files/"

    all_rNSR: List[float] = []
    all_rNSR_zscore: List[float] = []

    # entity_id → (total_windows, non_good_windows)
    entity_totals: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    # entity_id → channel_id → (total_windows, non_good_windows)
    entity_channel: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    n_files = 0
    n_errors = 0

    for obj in minio_client.list_objects(bucket, prefix=eeg_prefix, recursive=True):
        key = obj.object_name
        if not key.endswith("_biosignal_eeg_quality_flags.csv"):
            continue

        entity_id = _entity_from_path(key)
        if entity_id is None:
            logger.warning("Cannot extract entity from %s — skipping", key)
            continue

        df = _read_csv(minio_client, bucket, key)
        if df is None:
            n_errors += 1
            continue

        required = {"rNSR", "rNSR_zscore", "quality_flag", "channel_id"}
        if not required.issubset(df.columns):
            logger.warning("Missing columns in %s: %s", key, required - set(df.columns))
            n_errors += 1
            continue

        # --- 1 & 2: accumulate rNSR / rNSR_zscore values ---
        rnsr_vals = pd.to_numeric(df["rNSR"], errors="coerce").dropna().values
        zscore_vals = pd.to_numeric(df["rNSR_zscore"], errors="coerce").dropna().values
        all_rNSR.extend(rnsr_vals.tolist())
        all_rNSR_zscore.extend(zscore_vals.tolist())

        # --- 3: bad rate per entity ---
        n_total = len(df)
        n_non_good = int((df["quality_flag"] != "GOOD").sum())
        entity_totals[entity_id][0] += n_total
        entity_totals[entity_id][1] += n_non_good

        # --- 4: bad rate per channel per entity ---
        ch_col = pd.to_numeric(df["channel_id"], errors="coerce")
        for ch_id, grp in df.groupby(ch_col):
            ch_int = int(ch_id)
            entity_channel[entity_id][ch_int][0] += len(grp)
            entity_channel[entity_id][ch_int][1] += int((grp["quality_flag"] != "GOOD").sum())

        n_files += 1
        if n_files % 50 == 0:
            logger.info("Processed %d files...", n_files)

    logger.info("Processed %d EEG files (%d errors)", n_files, n_errors)

    # --- Build output 1: rNSR percentiles ---
    if all_rNSR:
        arr = np.array(all_rNSR)
        pcts = np.percentile(arr, _PERCENTILES)
        rNSR_df = pd.DataFrame(
            [{"n_values": len(arr), **{f"p{p}": round(float(v), 6) for p, v in zip(_PERCENTILES, pcts)}}]
        )
        logger.info(
            "rNSR global (n=%d)  p50=%.4f  p75=%.4f  p90=%.4f  p95=%.4f  p99=%.4f",
            len(arr), *pcts,
        )
    else:
        rNSR_df = pd.DataFrame(columns=["n_values"] + [f"p{p}" for p in _PERCENTILES])
        logger.warning("No rNSR values found")

    # --- Build output 2: rNSR_zscore percentiles ---
    if all_rNSR_zscore:
        arr_z = np.array(all_rNSR_zscore)
        pcts_z = np.percentile(arr_z, _PERCENTILES)
        rNSR_zscore_df = pd.DataFrame(
            [{"n_values": len(arr_z), **{f"p{p}": round(float(v), 4) for p, v in zip(_PERCENTILES, pcts_z)}}]
        )
        logger.info(
            "rNSR_zscore global (n=%d)  p50=%.4f  p75=%.4f  p90=%.4f  p95=%.4f  p99=%.4f",
            len(arr_z), *pcts_z,
        )
    else:
        rNSR_zscore_df = pd.DataFrame(columns=["n_values"] + [f"p{p}" for p in _PERCENTILES])
        logger.warning("No rNSR_zscore values found")

    # --- Build output 3: bad rate per entity ---
    entity_rows = []
    for entity_id, (total, non_good) in sorted(entity_totals.items()):
        bad_rate = round(100.0 * non_good / total, 2) if total > 0 else float("nan")
        entity_rows.append({
            "entity_id": entity_id,
            "total_windows": total,
            "non_good_windows": non_good,
            "bad_rate_pct": bad_rate,
        })
    entity_df = pd.DataFrame(entity_rows)
    if not entity_df.empty:
        logger.info(
            "Bad rate per entity: min=%.1f%%  median=%.1f%%  max=%.1f%%  (n=%d entities)",
            entity_df["bad_rate_pct"].min(),
            entity_df["bad_rate_pct"].median(),
            entity_df["bad_rate_pct"].max(),
            len(entity_df),
        )

    # --- Build output 4: bad rate per channel (macro-averaged across entities) ---
    # For each channel: mean over entities of (non_good / total) for that channel
    all_channels = sorted({ch for e in entity_channel.values() for ch in e})
    channel_rows = []
    for ch_id in all_channels:
        per_entity_rates = []
        for entity_id, ch_map in entity_channel.items():
            if ch_id in ch_map:
                total, non_good = ch_map[ch_id]
                if total > 0:
                    per_entity_rates.append(100.0 * non_good / total)
        if per_entity_rates:
            channel_rows.append({
                "channel_id": ch_id,
                "n_entities": len(per_entity_rates),
                "bad_rate_pct": round(float(np.mean(per_entity_rates)), 2),
                "bad_rate_std": round(float(np.std(per_entity_rates)), 2),
            })
    channel_df = pd.DataFrame(channel_rows)
    if not channel_df.empty:
        logger.info(
            "Bad rate per channel (macro-avg across entities): min=%.1f%%  max=%.1f%%  (n=%d channels)",
            channel_df["bad_rate_pct"].min(),
            channel_df["bad_rate_pct"].max(),
            len(channel_df),
        )

    return rNSR_df, rNSR_zscore_df, entity_df, channel_df


def main() -> None:
    parser = argparse.ArgumentParser(description="EAV EEG Quality Flags Statistics.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument(
        "--output-prefix", default=_DEFAULT_OUTPUT_PREFIX,
        help=f"MinIO prefix for output CSVs (default: {_DEFAULT_OUTPUT_PREFIX}).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    silver_bucket = cfg["bucket_silver"]

    qf_prefix = cfg.get("quality_flags", {}).get("output_prefix", "04_quality_flags").rstrip("/")

    logger.info("Starting EAV EEG Quality Flags Statistics (source: %s/eav/)", qf_prefix)

    rNSR_df, rNSR_zscore_df, entity_df, channel_df = run_stats(minio_client, silver_bucket, qf_prefix)

    out = args.output_prefix.rstrip("/")

    upload_csv(minio_client, silver_bucket, f"{out}/rNSR_percentiles.csv", rNSR_df)
    logger.info("rNSR percentiles → silver/%s/rNSR_percentiles.csv", out)

    upload_csv(minio_client, silver_bucket, f"{out}/rNSR_zscore_percentiles.csv", rNSR_zscore_df)
    logger.info("rNSR_zscore percentiles → silver/%s/rNSR_zscore_percentiles.csv", out)

    upload_csv(minio_client, silver_bucket, f"{out}/bad_rate_per_entity.csv", entity_df)
    logger.info("Bad rate per entity → silver/%s/bad_rate_per_entity.csv  (%d rows)", out, len(entity_df))

    upload_csv(minio_client, silver_bucket, f"{out}/bad_rate_per_channel.csv", channel_df)
    logger.info("Bad rate per channel → silver/%s/bad_rate_per_channel.csv  (%d rows)", out, len(channel_df))

    logger.info("EAV EEG Quality Flags Statistics complete.")


if __name__ == "__main__":
    main()
