import argparse
import io
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as project_config

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("silver_quality_flags_summary")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")

_QUALITY_FLAGS = ["GOOD", "NOISY", "BAD"]

# Reverse of signal_readers._SIGNAL_TO_FILESTEM — used to recover signal type from filename stem
_FILESTEM_TO_SIGNAL = {
    "biosignal_bvp":       "E4_BVP",
    "biosignal_eda":       "E4_EDA",
    "biosignal_acc":       "E4_ACC",
    "biosignal_hr":        "E4_HR",
    "biosignal_ibi":       "E4_IBI",
    "biosignal_temp":      "E4_TEMP",
    "biosignal_brainwave": "BrainWave",
    "biosignal_attention": "Attention",
    "biosignal_meditation":"Meditation",
    "biosignal_polar_hr":  "Polar_HR",
    "biosignal_eeg":       "eeg",
    "audio":               "audio",
    "video":               "video",
}


def _signal_type_from_path(obj_name: str) -> str:
    parts = obj_name.split("/")

    modality = None
    for part in parts:
        if part.startswith("modality="):
            modality = part[len("modality="):]
            break

    filename = parts[-1]
    stem = filename.replace("_quality_flags.csv", "")

    for file_stem, signal_type in _FILESTEM_TO_SIGNAL.items():
        if stem == file_stem or stem.endswith(f"_{file_stem}"):
            return signal_type

    return modality or "unknown"


def _dataset_from_path(obj_name: str, output_prefix: str) -> str:
    """Extract dataset name from path (segment right after output_prefix)."""
    remainder = obj_name[len(output_prefix):].lstrip("/")
    return remainder.split("/")[0]


def run_summary(
    minio_client,
    silver_bucket: str,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    qf_cfg = cfg.get("quality_flags", {})
    output_prefix = qf_cfg.get("output_prefix", "04_quality_flags").rstrip("/")

    # counts[(dataset, signal_type)][flag] → window count
    counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {f: 0 for f in _QUALITY_FLAGS}
    )

    n_files = 0
    n_errors = 0

    for obj in minio_client.list_objects(silver_bucket, prefix=output_prefix + "/", recursive=True):
        key = obj.object_name
        if not key.endswith("_quality_flags.csv"):
            continue

        dataset = _dataset_from_path(key, output_prefix)
        signal_type = _signal_type_from_path(key)

        try:
            response = minio_client.get_object(silver_bucket, key)
            data = response.read()
            response.close()
            response.release_conn()
        except Exception as e:
            logger.error("Failed to download %s: %s", key, e)
            n_errors += 1
            continue

        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception as e:
            logger.error("Failed to parse %s: %s", key, e)
            n_errors += 1
            continue

        if "quality_flag" not in df.columns:
            logger.warning("No quality_flag column in %s — skipping", key)
            continue

        vc = df["quality_flag"].value_counts()
        grp = counts[(dataset, signal_type)]
        for flag in _QUALITY_FLAGS:
            grp[flag] += int(vc.get(flag, 0))

        if dataset == "k-emocon" and signal_type == "video":
            n_bad = int(vc.get("NOISY", 0))
            if n_bad > 0:
                entity = next(
                    (p[len("entity="):] for p in key.split("/") if p.startswith("entity=")),
                    "unknown",
                )
                logger.info("[K-EmoCon][video] %d NOISY window(s) in %s (entity: %s)", n_bad, key.split("/")[-1], entity)

        n_files += 1
        if n_files % 100 == 0:
            logger.info("Processed %d files...", n_files)

    logger.info("Processed %d files total (%d errors)", n_files, n_errors)

    rows = []
    for (dataset, signal_type), flag_counts in sorted(counts.items()):
        total = sum(flag_counts.values())
        if total == 0:
            continue
        row: Dict[str, Any] = {
            "dataset": dataset,
            "signal_type": signal_type,
            "total_windows": total,
        }
        for flag in _QUALITY_FLAGS:
            row[flag.lower()] = flag_counts[flag]
            row[f"{flag.lower()}_pct"] = round(100.0 * flag_counts[flag] / total, 2)
        rows.append(row)

    return pd.DataFrame(rows)


def _upload_csv(minio_client, bucket: str, key: str, df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    minio_client.put_object(
        bucket, key,
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="text/csv",
    )


def _print_summary(df: pd.DataFrame) -> None:
    logger.info("=== Quality Flags Summary ===")
    for _, row in df.iterrows():
        logger.info(
            "%-12s  %-20s  GOOD: %5.1f%%  NOISY: %5.1f%%  BAD: %5.1f%%  (windows: %d)",
            row["dataset"], row["signal_type"],
            row["good_pct"], row["noisy_pct"], row["bad_pct"],
            row["total_windows"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silver — Step 04a: Quality Flags Summary."
    )
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    silver_bucket = cfg["bucket_silver"]

    qf_cfg = cfg.get("quality_flags", {})
    output_prefix = qf_cfg.get("output_prefix", "04_quality_flags").rstrip("/")
    output_key = f"{output_prefix}/quality_flags_summary.csv"

    logger.info("Starting Silver — Step 04a: Quality Flags Summary")

    summary_df = run_summary(minio_client, silver_bucket, cfg)

    if summary_df.empty:
        logger.warning("No quality_flags files found — summary is empty")
        return

    _print_summary(summary_df)

    _upload_csv(minio_client, silver_bucket, output_key, summary_df)
    logger.info("Summary uploaded to %s/%s", silver_bucket, output_key)

    logger.info("Quality Flags Summary complete.")


if __name__ == "__main__":
    main()
