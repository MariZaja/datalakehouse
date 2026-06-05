import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

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
logger = logging.getLogger("silver_quality_flags")

from minio_utils import load_missingness_report, _group_objects_by_entity
from quality_flags.kemocon_processing import load_kemocon_subjects, process_kemocon_entity
from quality_flags.eav_processing import process_eav_entity

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")


def run_quality_flags(
    minio_client,
    silver_bucket: str,
    cfg: Dict[str, Any],
    dataset_filter: str,
    test_mode: bool,
    skip_video: bool = False,
    video_only: bool = False,
    modality_filter=None,
) -> None:
    qf_cfg = cfg.get("quality_flags", {})
    output_prefix = qf_cfg.get("output_prefix", "04_quality_flags").rstrip("/")
    md_cfg = cfg.get("missingness_detection", {})

    miss_report_prefix = md_cfg.get("output_prefix", "03_missingness").rstrip("/")
    miss_report_file = md_cfg.get("output_report_filename", "missingness_report.csv")
    miss_lookup, miss_skip = load_missingness_report(
        minio_client, silver_bucket, f"{miss_report_prefix}/{miss_report_file}"
    )
    logger.info(
        "Missingness report: %d sample-rate entries, %d total_missing entries",
        len(miss_lookup), len(miss_skip),
    )

    run_kemocon = dataset_filter in ("all", "k-emocon")
    run_eav = dataset_filter in ("all", "eav")

    if run_kemocon:
        kemocon_md = md_cfg.get("datasets", {}).get("kemocon", {})
        kemocon_qf = qf_cfg.get("datasets", {}).get("kemocon", {})
        if kemocon_md:
            logger.info("=== Processing K-EmoCon ===")
            subjects_bucket = kemocon_qf.get("subjects_bucket") or kemocon_md.get("subjects_bucket", "silver")
            subjects_path = kemocon_qf.get("subjects_path") or kemocon_md.get("subjects_path", "")
            silver_files_prefix = kemocon_qf.get("silver_files_prefix") or kemocon_md.get("silver_files_prefix", "")

            subjects_map = load_kemocon_subjects(minio_client, subjects_bucket, subjects_path)
            logger.info("K-EmoCon subjects loaded: %d", len(subjects_map))

            entity_objects = _group_objects_by_entity(minio_client, silver_bucket, silver_files_prefix)
            logger.info("K-EmoCon entities: %d", len(entity_objects))

            entity_ids = sorted(entity_objects.keys())
            if test_mode and entity_ids:
                entity_ids = entity_ids[:1]
                logger.info("TEST MODE: processing only %s", entity_ids[0])

            for entity_id in entity_ids:
                logger.info("[K-EmoCon] Processing %s", entity_id)
                try:
                    process_kemocon_entity(
                        minio_client, silver_bucket, entity_id,
                        entity_objects[entity_id], subjects_map,
                        kemocon_md, qf_cfg, miss_lookup, miss_skip,
                        output_prefix, skip_video=skip_video, video_only=video_only,
                        modality_filter=modality_filter,
                    )
                except Exception as e:
                    logger.error("[K-EmoCon] [%s] Fatal error: %s", entity_id, e)

    if run_eav:
        eav_md = md_cfg.get("datasets", {}).get("eav", {})
        eav_qf = qf_cfg.get("datasets", {}).get("eav", {})
        if eav_md:
            logger.info("=== Processing EAV ===")
            silver_files_prefix = eav_qf.get("silver_files_prefix") or eav_md.get("silver_files_prefix", "")

            entity_objects = _group_objects_by_entity(minio_client, silver_bucket, silver_files_prefix)
            logger.info("EAV entities: %d", len(entity_objects))

            entity_ids = sorted(entity_objects.keys())
            if test_mode and entity_ids:
                entity_ids = entity_ids[:1]
                logger.info("TEST MODE: processing only %s", entity_ids[0])

            for entity_id in entity_ids:
                logger.info("[EAV] Processing %s", entity_id)
                try:
                    process_eav_entity(
                        minio_client, silver_bucket, entity_id,
                        entity_objects[entity_id], eav_md, qf_cfg,
                        miss_lookup, miss_skip, output_prefix, skip_video=skip_video, video_only=video_only,
                        modality_filter=modality_filter,
                    )
                except Exception as e:
                    logger.error("[EAV] [%s] Fatal error: %s", entity_id, e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver — Step 04: Noise Detection & Quality Flags.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument(
        "--dataset", default="all", choices=["k-emocon", "eav", "all"],
        help="Dataset to process (default: all).",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Process only the first entity of k-emocon (for development/testing).",
    )
    parser.add_argument(
        "--skip-video", action="store_true",
        help="Skip video signal processing (faster runs without CV-heavy checks).",
    )
    parser.add_argument(
        "--video-only", action="store_true",
        help="Process only video signals, skip all other modalities.",
    )
    parser.add_argument(
        "--modality", nargs="+", metavar="SIGNAL_TYPE",
        help=(
            "Process only the listed signal types, e.g. --modality eeg audio E4_BVP. "
            "When given, overrides --skip-video and --video-only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    silver_bucket = cfg["bucket_silver"]

    logger.info("Starting Silver — Step 04: Noise Detection & Quality Flags")

    modality_filter = {m.lower() for m in args.modality} if args.modality else None

    run_quality_flags(
        minio_client, silver_bucket, cfg,
        dataset_filter=args.dataset,
        test_mode=args.test,
        skip_video=args.skip_video,
        video_only=args.video_only,
        modality_filter=modality_filter,
    )

    logger.info("Quality Flags complete.")


if __name__ == "__main__":
    main()
