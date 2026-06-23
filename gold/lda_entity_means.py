"""Gold — Step 03b: LDA Entity Mean Report.

Reads per-window LDA parquets from gold/lda_reduction/eav and computes,
for each entity, the mean of lda_1/lda_2/lda_3 per modality × emotion.
Outputs a single CSV with one row per entity (36 feature columns):

  entity_id,
  audio_angry_lda_1, ..., audio_calm_lda_3,   (4 emotions × 3 = 12)
  eeg_angry_lda_1,   ..., eeg_calm_lda_3,
  video_angry_lda_1, ..., video_calm_lda_3
"""
import argparse
import io
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as project_config
from minio_utils import download_object, upload_csv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gold_lda_entity_means")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")
_MODALITIES = ("audio", "eeg", "video")
# Must match window_emotion values produced by silver/05_annotation_quality.py
_EMOTIONS = ("Anger", "Sadness", "Happiness", "Calm")
_LDA_COLS = ("lda_1", "lda_2", "lda_3")


def _load_parquet(minio_client, bucket: str, key: str) -> Optional[pd.DataFrame]:
    data = download_object(minio_client, bucket, key)
    if data is None:
        return None
    try:
        return pd.read_parquet(io.BytesIO(data))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold — Step 03b: LDA Entity Mean Report.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument("--test", action="store_true", help="Process only the first entity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    gold_bucket = cfg["bucket_gold"]

    lda_prefix = cfg.get("lda_reduction", {}).get("output_prefix", "lda_reduction")
    eav_prefix = f"{lda_prefix}/eav"
    output_key = f"{eav_prefix}/entity_lda_means.csv"

    entities = _list_entities(minio_client, gold_bucket, eav_prefix)
    if args.test:
        entities = entities[:1]

    logger.info("Computing LDA means for %d entities", len(entities))

    rows = []
    for entity_id in entities:
        row = {"entity_id": entity_id}
        for modality in _MODALITIES:
            key = f"{eav_prefix}/{entity_id}/{entity_id}_{modality}_lda.parquet"
            df = _load_parquet(minio_client, gold_bucket, key)
            emotion_means = (
                df.groupby("window_emotion")[list(_LDA_COLS)].mean()
                if df is not None and "window_emotion" in df.columns
                else None
            )
            for emotion in _EMOTIONS:
                for col in _LDA_COLS:
                    out_col = f"{modality}_{emotion.lower()}_{col}"
                    if (
                        emotion_means is not None
                        and emotion in emotion_means.index
                        and col in emotion_means.columns
                    ):
                        val = emotion_means.at[emotion, col]
                        row[out_col] = 0 if pd.isna(val) else val
                    else:
                        row[out_col] = 0
        rows.append(row)
        logger.info("[%s] done", entity_id)

    if not rows:
        logger.warning("No entities found under %s/%s", gold_bucket, eav_prefix)
        return

    col_order = ["entity_id"] + [
        f"{m}_{e.lower()}_{c}" for m in _MODALITIES for e in _EMOTIONS for c in _LDA_COLS
    ]
    report = pd.DataFrame(rows)[col_order]

    upload_csv(minio_client, gold_bucket, output_key, report)
    logger.info("Report (%d rows) → %s/%s", len(report), gold_bucket, output_key)


if __name__ == "__main__":
    main()
