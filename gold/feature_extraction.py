"""Gold — Step 02: Feature Extraction.

Extracts features from silver files and saves to gold/feature_extraction/ in MinIO.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
sys.path.insert(0, str(Path(__file__).resolve().parent))          # gold/ (for features.*)

import config as project_config

from features.audio import init_opensmile
from features.entity_eav import process_eav_entity
from features.entity_kemocon import process_kemocon_entity
from features.video import init_openface

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("gold_feature_extraction")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")


def _list_entities(minio_client, bucket: str, prefix: str) -> List[str]:
    entities: set = set()
    full_prefix = prefix.rstrip("/") + "/"
    for obj in minio_client.list_objects(bucket, prefix=full_prefix, recursive=True):
        for seg in obj.object_name.split("/"):
            if seg.startswith("entity="):
                entities.add(seg[len("entity="):])
                break
    return sorted(entities)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold — Step 02: Feature Extraction.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument(
        "--dataset", choices=["eav", "k-emocon", "all"], default="all",
        help="Dataset to process.",
    )
    parser.add_argument(
        "--modality",
        choices=["audio", "video", "eeg", "bvp", "acc", "hr", "ibi", "eda", "temp", "all"],
        default="all",
        help="Modality to extract (default: all). bvp/acc/hr/ibi/eda/temp apply only to K-EmoCon.",
    )
    parser.add_argument(
        "--entity",
        help="Process only this entity ID (e.g. S01 or p01). Applies to the selected dataset(s).",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Process only the first entity per dataset (smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    silver_bucket = cfg["bucket_silver"]
    gold_bucket = cfg["bucket_gold"]
    feat_cfg = cfg.get("feature_extraction", {})
    output_prefix = feat_cfg.get("output_prefix", "feature_extraction")
    window_size_s = float(feat_cfg.get("window_size_s", 0.3))
    hr_ibi_window_size_s = float(feat_cfg.get("hr_ibi_window_size_s", 10.0))
    hr_ibi_step_s = float(feat_cfg.get("hr_ibi_step_s", 0.3))
    eda_temp_window_size_s = float(feat_cfg.get("eda_temp_window_size_s", 30.0))
    eda_temp_step_s = float(feat_cfg.get("eda_temp_step_s", 0.3))

    smile = init_opensmile()
    openface_available = init_openface()

    datasets_cfg = cfg.get("datasets", {})
    eav_cfg = datasets_cfg.get("eav", {})
    kemocon_cfg = datasets_cfg.get("kemocon", {})

    eav_silver_files_prefix = (
        eav_cfg.get("silver_base_prefix", "01_entity_resolution/eav").rstrip("/") + "/files"
    )
    kemocon_silver_files_prefix = (
        kemocon_cfg.get("silver_base_prefix", "01_entity_resolution/k-emocon").rstrip("/") + "/files"
    )

    # Collect EEG signal config from time_audit (declared_hz, axes) and
    # missingness_detection (expected_instances, expected_timepoints).
    ta_eav_eeg = (
        cfg.get("time_audit", {}).get("datasets", {}).get("eav", {}).get("eeg_signal", {})
    )
    miss_eav_signals = (
        cfg.get("missingness_detection", {}).get("datasets", {}).get("eav", {})
        .get("expected_signals", [])
    )
    miss_eav_eeg = next(
        (s for s in miss_eav_signals if s.get("signal_type") == "eeg"), {}
    )
    eeg_sig_cfg: Dict = {
        "declared_hz": ta_eav_eeg.get("declared_hz", 500.0),
        "timepoints_axis": ta_eav_eeg.get("timepoints_axis", 0),
        "instances_axis": ta_eav_eeg.get("instances_axis", 2),
        "expected_instances": miss_eav_eeg.get("expected_instances", 200),
        "expected_timepoints": miss_eav_eeg.get("expected_timepoints", 10000),
    }

    # Load K-EmoCon subjects (debate start/end timestamps) — same source as quality_flags.
    from silver.quality_flags.kemocon_processing import load_kemocon_subjects
    kemocon_md = cfg.get("missingness_detection", {}).get("datasets", {}).get("kemocon", {})
    subjects_map = load_kemocon_subjects(
        minio_client,
        kemocon_md.get("subjects_bucket", "silver"),
        kemocon_md.get("subjects_path", "01_entity_resolution/k-emocon/auxiliary/metadata/metadata/subjects.csv"),
    )
    logger.info("K-EmoCon subjects loaded: %d", len(subjects_map))

    modalities: Set[str] = (
        {"audio", "video", "eeg", "bvp", "acc", "hr", "ibi", "eda", "temp"}
        if args.modality == "all" else {args.modality}
    )

    logger.info(
        "Starting Gold — Step 02: Feature Extraction (window_size_s=%.3f, modalities=%s)",
        window_size_s, sorted(modalities),
    )

    if args.dataset in ("eav", "all"):
        entities = _list_entities(minio_client, silver_bucket, eav_silver_files_prefix)
        if args.entity:
            entities = [e for e in entities if e == args.entity]
        elif args.test:
            entities = entities[:1]
        logger.info("[EAV] %d entities%s", len(entities), " (test mode)" if args.test else "")
        for entity_id in entities:
            logger.info("[EAV] Processing %s", entity_id)
            try:
                process_eav_entity(
                    minio_client, silver_bucket, gold_bucket,
                    entity_id, eav_silver_files_prefix, output_prefix,
                    smile=smile,
                    eeg_sig_cfg=eeg_sig_cfg,
                    window_size_s=window_size_s,
                    openface_available=openface_available,
                    modalities=modalities,
                )
            except Exception as e:
                logger.error("[EAV] [%s] Processing failed: %s", entity_id, e)

    if args.dataset in ("k-emocon", "all"):
        entities = _list_entities(minio_client, silver_bucket, kemocon_silver_files_prefix)
        if args.entity:
            entities = [e for e in entities if e == args.entity]
        elif args.test:
            entities = entities[:1]
        logger.info("[K-EmoCon] %d entities%s", len(entities), " (test mode)" if args.test else "")
        for entity_id in entities:
            logger.info("[K-EmoCon] Processing %s", entity_id)
            try:
                process_kemocon_entity(
                    minio_client, silver_bucket, gold_bucket,
                    entity_id, kemocon_silver_files_prefix, output_prefix,
                    smile=smile,
                    subjects_map=subjects_map,
                    window_size_s=window_size_s,
                    hr_ibi_window_size_s=hr_ibi_window_size_s,
                    hr_ibi_step_s=hr_ibi_step_s,
                    eda_temp_window_size_s=eda_temp_window_size_s,
                    eda_temp_step_s=eda_temp_step_s,
                    openface_available=openface_available,
                    modalities=modalities,
                )
            except Exception as e:
                logger.error("[K-EmoCon] [%s] Processing failed: %s", entity_id, e)

    logger.info("Gold — Step 02: Feature Extraction complete.")


if __name__ == "__main__":
    main()
