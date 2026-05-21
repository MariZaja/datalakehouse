import argparse
import io
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as project_config
from minio_utils import download_object, _group_objects_by_entity, upload_parquet

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("gold_data_quality")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")

_SEVERITY = {"GOOD": 0, "NOISY": 1, "BAD": 2}


def _worst_flag(flags: List[str]) -> Optional[str]:
    valid = [f for f in flags if f in _SEVERITY]
    if not valid:
        return None
    return max(valid, key=lambda f: _SEVERITY[f])


def _signal_type_from_path(path: str) -> str:
    m = re.search(r'biosignal_(.*?)_quality_flags\.csv', path)
    if m:
        return m.group(1)
    if "modality=video" in path:
        return "video"
    if "modality=audio" in path:
        return "audio"
    return "unknown"


def _load_csv(minio_client, bucket: str, obj_name: str) -> Optional[pd.DataFrame]:
    data = download_object(minio_client, bucket, obj_name)
    if data is None:
        return None
    try:
        return pd.read_csv(io.BytesIO(data))
    except Exception as e:
        logger.error("Failed to parse %s: %s", obj_name, e)
        return None


def _build_eav_report(
    minio_client,
    silver_bucket: str,
    entity_id: str,
    qf_prefix: str,
    annotation_prefix: str,
    flag_columns: Dict[str, str],
    n_trials: int = 200,
    windows_per_trial: int = 20,
) -> pd.DataFrame:
    skeleton = pd.DataFrame({
        "trial_id": [t for t in range(1, n_trials + 1) for _ in range(windows_per_trial)],
        "win": [w for _ in range(n_trials) for w in range(windows_per_trial)],
    })
    skeleton["window_id"] = skeleton.apply(
        lambda r: f"{int(r.trial_id):03d}_{int(r.win):02d}", axis=1
    )

    # signal_type → {(trial_id, win_within_trial) → flag}
    sig_flags: Dict[str, Dict[Tuple[int, int], str]] = {}
    entity_prefix = f"{qf_prefix}/entity={entity_id}/"

    for obj in minio_client.list_objects(silver_bucket, prefix=entity_prefix, recursive=True):
        path = obj.object_name
        filename = path.split("/")[-1]
        if not filename.endswith("_quality_flags.csv"):
            continue

        signal_type = _signal_type_from_path(path)
        if flag_columns.get(signal_type) is None:
            continue

        df = _load_csv(minio_client, silver_bucket, path)
        if df is None or "quality_flag" not in df.columns:
            continue

        if signal_type not in sig_flags:
            sig_flags[signal_type] = {}

        if signal_type == "eeg":
            # filename: {entity_id}_{NNN}_biosignal_eeg_quality_flags.csv, NNN is 0-based session index
            m = re.search(r'_(\d{3})_biosignal_eeg_quality_flags', filename)
            if not m or "window_id" not in df.columns:
                continue
            trial_id = int(m.group(1)) + 1
            for win_id, grp in df.groupby("window_id"):
                flag = _worst_flag(grp["quality_flag"].tolist())
                if flag:
                    sig_flags[signal_type][(trial_id, int(win_id))] = flag
        else:
            # audio/video: filename starts with trial_id number, windows indexed by window_start_s
            m = re.match(r'^(\d+)', filename)
            if not m or "window_start_s" not in df.columns:
                continue
            trial_id = int(m.group(1))
            for _, row in df.iterrows():
                win = int(row["window_start_s"])
                sig_flags[signal_type][(trial_id, win)] = row["quality_flag"]

    for signal_type, flags_dict in sig_flags.items():
        col_name = flag_columns.get(signal_type)
        if col_name is None:
            continue
        skeleton[col_name] = skeleton.apply(
            lambda r, fd=flags_dict: fd.get((int(r.trial_id), int(r.win))),
            axis=1,
        )

    annot_key = f"{annotation_prefix}/{entity_id}_annotation_quality.csv"
    annot_df = _load_csv(minio_client, silver_bucket, annot_key)
    if annot_df is not None and "trial_id" in annot_df.columns and "valence_arousal_emotion" in annot_df.columns:
        emotion_map = {
            int(t): e
            for t, e in zip(annot_df["trial_id"], annot_df["valence_arousal_emotion"])
        }
        skeleton["window_emotion"] = skeleton["trial_id"].map(emotion_map)
    else:
        if annot_df is None:
            logger.warning("[EAV] [%s] Annotation quality not found: %s", entity_id, annot_key)
        skeleton["window_emotion"] = None

    output_cols = ["window_id"]
    for signal_type in ("video", "audio", "eeg"):
        col = flag_columns.get(signal_type)
        if col and col in skeleton.columns:
            output_cols.append(col)
    output_cols.append("window_emotion")

    return skeleton[[c for c in output_cols if c in skeleton.columns]]


def _build_kemocon_report(
    minio_client,
    silver_bucket: str,
    entity_id: str,
    qf_prefix: str,
    annotation_prefix: str,
    flag_columns: Dict[str, str],
) -> Optional[pd.DataFrame]:
    entity_prefix = f"{qf_prefix}/entity={entity_id}/"

    # signal_type → {window_start_s (int) → flag}
    sig_flags: Dict[str, Dict[int, str]] = {}

    for obj in minio_client.list_objects(silver_bucket, prefix=entity_prefix, recursive=True):
        path = obj.object_name
        filename = path.split("/")[-1]
        if not filename.endswith("_quality_flags.csv"):
            continue

        signal_type = _signal_type_from_path(path)
        if flag_columns.get(signal_type) is None:
            continue

        df = _load_csv(minio_client, silver_bucket, path)
        if df is None or "quality_flag" not in df.columns or "window_start_s" not in df.columns:
            continue

        if signal_type not in sig_flags:
            sig_flags[signal_type] = {}
        for _, row in df.iterrows():
            win = int(row["window_start_s"])
            sig_flags[signal_type][win] = row["quality_flag"]

    if not sig_flags:
        return None

    all_windows = sorted(set(w for flags in sig_flags.values() for w in flags))
    skeleton = pd.DataFrame({"window_id": all_windows})

    for signal_type, flags_dict in sig_flags.items():
        col_name = flag_columns.get(signal_type)
        if col_name is None:
            continue
        skeleton[col_name] = skeleton["window_id"].map(flags_dict)

    annot_key = f"{annotation_prefix}/{entity_id}_annotation_quality.csv"
    annot_df = _load_csv(minio_client, silver_bucket, annot_key)
    if annot_df is not None and "seconds" in annot_df.columns and "valence_arousal_emotion" in annot_df.columns:
        emotion_map = {
            int(s): e
            for s, e in zip(annot_df["seconds"], annot_df["valence_arousal_emotion"])
        }
        # annotation_quality uses 5s windows; map each 1s QF window to its 5s bucket
        skeleton["window_emotion"] = skeleton["window_id"].map(
            lambda s: emotion_map.get((s // 5) * 5)
        )
    else:
        if annot_df is None:
            logger.warning("[K-EmoCon] [%s] Annotation quality not found: %s", entity_id, annot_key)
        skeleton["window_emotion"] = None

    output_cols = ["window_id"]
    for col_name in flag_columns.values():
        if col_name in skeleton.columns:
            output_cols.append(col_name)
    output_cols.append("window_emotion")

    return skeleton[[c for c in output_cols if c in skeleton.columns]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold — Step 01: Data Quality.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config.")
    return parser.parse_args()


_ANNOT_SUBDIR: Dict[str, str] = {
    "eav": "eav",
    "k-emocon": "kemocon",
}


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    silver_bucket = cfg["bucket_silver"]
    gold_bucket = cfg["bucket_gold"]

    dq_cfg = cfg.get("data_quality", {})
    output_prefix = dq_cfg.get("output_prefix", "data_quality").rstrip("/")
    aq_output_prefix = cfg.get("annotation_quality", {}).get("output_prefix", "05_annotation_quality").rstrip("/")
    qf_output_prefix = cfg.get("quality_flags", {}).get("output_prefix", "04_quality_flags").rstrip("/")

    logger.info("Starting Gold — Step 01: Data Quality")

    for dataset_key, dataset_prop in dq_cfg.get("datasets", {}).items():
        label = dataset_prop["dataset_label"]
        dataset_folder_name = dataset_prop["dataset_folder_name"]
        flag_columns = dataset_prop["flag_columns"]

        qf_prefix = f"{qf_output_prefix}/{dataset_folder_name}/files"
        annotation_prefix = f"{aq_output_prefix}/{_ANNOT_SUBDIR.get(dataset_key, dataset_key)}"

        entity_objects = _group_objects_by_entity(minio_client, silver_bucket, qf_prefix)
        logger.info("[%s] Entities: %d", label, len(entity_objects))

        for entity_id in sorted(entity_objects):
            logger.info("[%s] Processing %s", label, entity_id)

            if dataset_key == "eav":
                report_df = _build_eav_report(
                    minio_client, silver_bucket, entity_id,
                    qf_prefix, annotation_prefix, flag_columns,
                )
            elif dataset_key == "k-emocon":
                report_df = _build_kemocon_report(
                    minio_client, silver_bucket, entity_id,
                    qf_prefix, annotation_prefix, flag_columns,
                )
            else:
                logger.warning("No builder for dataset '%s' — skipping", dataset_key)
                continue

            if report_df is None or report_df.empty:
                logger.warning("[%s] [%s] Empty report — skipping", label, entity_id)
                continue

            output_key = f"{output_prefix}/{dataset_folder_name}/files/{entity_id}_data_quality.parquet"
            upload_parquet(minio_client, gold_bucket, output_key, report_df)
            logger.info("[%s] [%s] %d rows → %s/%s", label, entity_id, len(report_df), gold_bucket, output_key)

    logger.info("Gold — Step 01: Data Quality complete.")


if __name__ == "__main__":
    main()
