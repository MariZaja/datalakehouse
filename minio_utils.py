import io
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger("silver.minio_utils")


def download_object(minio_client, bucket: str, key: str) -> Optional[bytes]:
    try:
        response = minio_client.get_object(bucket, key)
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except Exception as e:
        logger.error("Failed to download %s/%s: %s", bucket, key, e)
        return None


def _group_objects_by_entity(minio_client, bucket: str, prefix: str) -> Dict[str, List[Any]]:
    entity_objects: Dict[str, List[Any]] = {}
    full_prefix = prefix.rstrip("/") + "/"
    for obj in minio_client.list_objects(bucket, prefix=full_prefix, recursive=True):
        for seg in obj.object_name.split("/"):
            if seg.startswith("entity="):
                eid = seg[len("entity="):]
                entity_objects.setdefault(eid, []).append(obj)
                break
    return entity_objects


def upload_csv(minio_client, bucket: str, key: str, df: pd.DataFrame) -> None:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    minio_client.put_object(
        bucket, key,
        data=io.BytesIO(csv_bytes),
        length=len(csv_bytes),
        content_type="text/csv",
    )


def load_missingness_report(
    minio_client, bucket: str, key: str
) -> Tuple[Dict[Tuple[str, str, str], float], Set[Tuple[str, str, str]]]:
    """Return (sample_rate_lookup, total_missing_set).

    sample_rate_lookup: {(participant_id, unit_id, signal_type): sample_rate_hz}
    total_missing_set:  keys where status == 'total_missing' — skip processing entirely
    """
    data = download_object(minio_client, bucket, key)
    if data is None:
        logger.warning("Missingness report not found at %s/%s — sample rates will fall back to config", bucket, key)
        return {}, set()
    df = pd.read_csv(io.BytesIO(data))
    sr_lookup: Dict[Tuple[str, str, str], float] = {}
    miss_skip: Set[Tuple[str, str, str]] = set()
    for _, row in df.iterrows():
        key_tuple = (str(row["participant_id"]), str(row["unit_id"]), str(row["signal_type"]))
        if str(row.get("status", "")) == "total_missing":
            miss_skip.add(key_tuple)
        sr = row.get("sample_rate_hz")
        if pd.notna(sr) and float(sr) > 0:
            sr_lookup[key_tuple] = float(sr)
    return sr_lookup, miss_skip
