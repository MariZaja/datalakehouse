"""Gold — Step 03: LDA Reduction.

For each entity and modality, fits a supervised LDA (n_components=3) using
window_emotion as the class label and saves the transformed features.
Reads feature parquets from the feature_extraction output and emotion labels
from the data_quality output.
"""
import argparse
import io
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as project_config
from minio_utils import download_object, upload_parquet

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gold_lda_reduction")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")

_N_LDA = 3

_META_COLS: Set[str] = {
    "file_id", "entity_id", "trial_id", "window_id",
    "window_start_s", "window_end_s",
}

# K-EmoCon uses "kemocon" as the feature-extraction subfolder name
_FEAT_FOLDER: Dict[str, str] = {
    "eav": "eav",
    "k-emocon": "kemocon",
}

# "hr" modality is saved as "e4_hr" in feature_extraction for K-EmoCon
_KEMOCON_FILE_SUFFIX: Dict[str, str] = {
    "hr": "e4_hr",
}

_DATASET_MODALITIES: Dict[str, Set[str]] = {
    "eav": {"audio", "video", "eeg"},
    "k-emocon": {
        "audio", "video", "eeg", "attention", "meditation",
        "bvp", "acc", "hr", "ibi", "eda", "temp", "polar_hr",
    },
}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten tuple-string column names like \"('trial_id', '')\" → \"trial_id\"."""
    import ast
    new_cols = []
    for c in df.columns:
        try:
            parsed = ast.literal_eval(str(c))
            if isinstance(parsed, tuple):
                new_cols.append("_".join(str(p) for p in parsed if p))
                continue
        except (ValueError, SyntaxError):
            pass
        new_cols.append(c)
    df.columns = new_cols
    return df


def _load_parquet(minio_client, bucket: str, key: str) -> Optional[pd.DataFrame]:
    data = download_object(minio_client, bucket, key)
    if data is None:
        return None
    try:
        return _flatten_columns(pd.read_parquet(io.BytesIO(data)))
    except Exception as e:
        logger.error("Failed to parse parquet %s: %s", key, e)
        return None


def _list_entities(minio_client, bucket: str, prefix: str) -> List[str]:
    entities: set = set()
    full_prefix = prefix.rstrip("/") + "/"
    for obj in minio_client.list_objects(bucket, prefix=full_prefix, recursive=True):
        rel = obj.object_name[len(full_prefix):]
        entity = rel.split("/")[0]
        if entity:
            entities.add(entity)
    return sorted(entities)


def _apply_lda(
    feat_df: pd.DataFrame,
    dq_df: pd.DataFrame,
    dataset: str,
    entity_id: str,
    modality: str,
) -> Optional[pd.DataFrame]:
    feat_cols = [c for c in feat_df.columns if c not in _META_COLS]
    if not feat_cols:
        logger.warning("[%s] [%s] [%s] No feature columns found", dataset, entity_id, modality)
        return None

    if dataset == "eav":
        if "trial_id" not in feat_df.columns or "window_id" not in feat_df.columns:
            logger.warning("[%s] [%s] [%s] Missing trial_id/window_id", dataset, entity_id, modality)
            return None
        feat = feat_df.copy()
        # all modalities: feature_extraction window_id is 1-indexed; DQ WW is 0-indexed → subtract 1
        # eeg trial_id is 0-indexed; DQ TTT is 1-indexed → add 1
        wid_adjust = -1
        tid_adjust = 1 if modality == "eeg" else 0
        feat["_dq_key"] = feat.apply(
            lambda r: f"{int(r.trial_id) + tid_adjust:03d}_{int(r.window_id) + wid_adjust:02d}", axis=1
        )
        labeled = feat.merge(
            dq_df[["window_id", "window_emotion"]].rename(columns={"window_id": "_dq_key"}),
            on="_dq_key",
            how="inner",
        ).drop(columns=["_dq_key"])
    else:
        if "window_start_s" not in feat_df.columns or "window_start_s" not in dq_df.columns:
            logger.warning("[%s] [%s] [%s] Missing window_start_s", dataset, entity_id, modality)
            return None
        feat_sorted = feat_df.sort_values("window_start_s").reset_index(drop=True)
        dq_sorted = (
            dq_df[["window_start_s", "window_emotion"]]
            .sort_values("window_start_s")
            .reset_index(drop=True)
        )
        labeled = pd.merge_asof(feat_sorted, dq_sorted, on="window_start_s", direction="nearest")

    labeled = labeled.dropna(subset=["window_emotion"]).reset_index(drop=True)
    if labeled.empty:
        logger.warning("[%s] [%s] [%s] No labeled windows after join", dataset, entity_id, modality)
        return None

    X = labeled[feat_cols].select_dtypes(include=[np.number]).fillna(0).values
    if X.shape[1] == 0:
        logger.warning("[%s] [%s] [%s] No numeric feature columns", dataset, entity_id, modality)
        return None

    y = labeled["window_emotion"].values
    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes < 2:
        logger.warning("[%s] [%s] [%s] Only %d class — LDA requires ≥2", dataset, entity_id, modality, n_classes)
        return None

    n_components = min(_N_LDA, n_classes - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X, y)
    transformed = lda.transform(X)

    id_cols = ["entity_id"]
    if "trial_id" in labeled.columns:
        id_cols.append("trial_id")
    id_cols += ["window_id", "window_emotion"]

    out = labeled[id_cols].copy()
    for i in range(1, _N_LDA + 1):
        out[f"lda_{i}"] = transformed[:, i - 1] if i <= n_components else np.nan

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold — Step 03: LDA Reduction.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="Path to YAML config.")
    parser.add_argument(
        "--dataset", choices=["eav", "k-emocon", "all"], default="all",
    )
    parser.add_argument(
        "--modality",
        choices=sorted(
            _DATASET_MODALITIES["eav"] | _DATASET_MODALITIES["k-emocon"] | {"all"}
        ),
        default="all",
    )
    parser.add_argument("--entity", help="Start from this entity ID (inclusive).")
    parser.add_argument("--test", action="store_true", help="Process only the first entity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    gold_bucket = cfg["bucket_gold"]

    dq_cfg = cfg.get("data_quality", {})
    lda_cfg = cfg.get("lda_reduction", {})
    feat_prefix = lda_cfg.get("feature_extraction_prefix", "feature_extraction")
    dq_prefix = lda_cfg.get("data_quality_prefix", "data_quality").rstrip("/")
    output_prefix = lda_cfg.get("output_prefix", "lda_reduction").rstrip("/")

    datasets = ["eav", "k-emocon"] if args.dataset == "all" else [args.dataset]
    req_modalities: Optional[Set[str]] = None if args.modality == "all" else {args.modality}

    logger.info("Starting Gold — Step 03: LDA Reduction (n_components=%d)", _N_LDA)

    for dataset_key in datasets:
        ds_modalities = _DATASET_MODALITIES[dataset_key].copy()
        if req_modalities is not None:
            ds_modalities &= req_modalities

        feat_folder = _FEAT_FOLDER[dataset_key]
        dq_dataset_cfg = dq_cfg.get("datasets", {}).get(dataset_key, {})
        dq_folder = dq_dataset_cfg.get("dataset_folder_name", dataset_key)

        feat_entity_prefix = f"{feat_prefix}/{feat_folder}"
        entities = _list_entities(minio_client, gold_bucket, feat_entity_prefix)

        if args.entity:
            if args.entity in entities:
                entities = entities[entities.index(args.entity):]
            else:
                logger.warning("[%s] Entity '%s' not found — skipping", dataset_key, args.entity)
                entities = []
        elif args.test:
            entities = entities[:1]

        logger.info("[%s] %d entities, modalities=%s", dataset_key, len(entities), sorted(ds_modalities))

        for entity_id in entities:
            dq_key = f"{dq_prefix}/{dq_folder}/files/{entity_id}_data_quality.parquet"
            dq_df = _load_parquet(minio_client, gold_bucket, dq_key)
            if dq_df is None or "window_emotion" not in dq_df.columns:
                logger.warning("[%s] [%s] Data quality file missing or no window_emotion — skipping", dataset_key, entity_id)
                continue

            for modality in sorted(ds_modalities):
                file_suffix = _KEMOCON_FILE_SUFFIX.get(modality, modality) if dataset_key == "k-emocon" else modality
                feat_key = f"{feat_entity_prefix}/{entity_id}/{entity_id}_{file_suffix}.parquet"
                feat_df = _load_parquet(minio_client, gold_bucket, feat_key)
                if feat_df is None:
                    logger.debug("[%s] [%s] No features for modality=%s", dataset_key, entity_id, modality)
                    continue

                out_df = _apply_lda(feat_df, dq_df, dataset_key, entity_id, modality)
                if out_df is None or out_df.empty:
                    continue

                out_key = f"{output_prefix}/{feat_folder}/{entity_id}/{entity_id}_{modality}_lda.parquet"
                upload_parquet(minio_client, gold_bucket, out_key, out_df)
                logger.info(
                    "[%s] [%s] [%s] %d windows → %s/%s",
                    dataset_key, entity_id, modality, len(out_df), gold_bucket, out_key,
                )

    logger.info("Gold — Step 03: LDA Reduction complete.")


if __name__ == "__main__":
    main()
