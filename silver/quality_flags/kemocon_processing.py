import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from .minio_utils import download_object, upload_csv
from .signal_readers import _output_filename
from .biosignals import (
    process_e4_bvp, process_e4_eda, process_e4_acc, process_hr,
    process_e4_ibi, process_e4_temp, process_brainwave, process_att_med,
)
from .audio_video_qc import process_kemocon_audio, process_kemocon_video

logger = logging.getLogger("silver_quality_flags")

_KEMOCON_SIGNAL_PROCESSORS = {
    "E4_BVP":    process_e4_bvp,
    "E4_EDA":    process_e4_eda,
    "E4_ACC":    process_e4_acc,
    "E4_HR":     process_hr,
    "E4_IBI":    process_e4_ibi,
    "E4_TEMP":   process_e4_temp,
    "BrainWave": process_brainwave,
    "Attention": process_att_med,
    "Meditation": process_att_med,
    "Polar_HR":  process_hr,
    "audio":     process_kemocon_audio,
    "video":     process_kemocon_video,
}


def load_kemocon_subjects(minio_client, bucket: str, path: str) -> Dict[int, Dict[str, int]]:
    data = download_object(minio_client, bucket, path)
    if data is None:
        logger.error("Cannot load subjects.csv from %s/%s", bucket, path)
        return {}
    df = pd.read_csv(io.BytesIO(data))
    result = {}
    for _, row in df.iterrows():
        pid = int(row["pid"])
        result[pid] = {"startTime": int(row["startTime"]), "endTime": int(row["endTime"])}
    return result


def _pid_from_entity_id(entity_id: str) -> int:
    return int(entity_id[1:])


def process_kemocon_entity(
    minio_client,
    bucket: str,
    entity_id: str,
    objects: List[Any],
    subjects_map: Dict[int, Dict[str, int]],
    kemocon_md_cfg: Dict[str, Any],
    qf_cfg: Dict[str, Any],
    miss_lookup: Dict[Tuple[str, str, str], float],
    miss_skip: Set[Tuple[str, str, str]],
    output_prefix: str,
    skip_video: bool = False,
    video_only: bool = False,
    modality_filter: Optional[Set[str]] = None,
) -> None:
    """Process all signals for one K-EmoCon entity; upload one CSV per signal."""
    pid = _pid_from_entity_id(entity_id)
    subject = subjects_map.get(pid)
    if not subject:
        logger.warning("[K-EmoCon] [%s] No subject entry — skipping", entity_id)
        return

    debate_start_s = subject["startTime"] / 1000.0
    debate_end_s = subject["endTime"] / 1000.0
    debate_duration_s = debate_end_s - debate_start_s

    timestamp_col = kemocon_md_cfg.get("timestamp_col", "timestamp")
    ts_unit_ms = kemocon_md_cfg.get("timestamp_unit_ms", False)
    expected_signals: List[Dict] = kemocon_md_cfg.get("expected_signals", [])

    obj_by_fname: Dict[str, Any] = {}
    obj_by_ext: Dict[str, List[Any]] = {}
    for obj in objects:
        fname = obj.object_name.split("/")[-1]
        obj_by_fname[fname] = obj
        obj_by_ext.setdefault(Path(fname).suffix.lower(), []).append(obj)

    computed_dfs: Dict[str, pd.DataFrame] = {}

    for sig in expected_signals:
        signal_type = sig["signal_type"]
        modality = sig.get("modality", signal_type.lower())
        if modality_filter is not None:
            if signal_type.lower() not in modality_filter:
                continue
        else:
            if skip_video and signal_type == "video":
                logger.info("[K-EmoCon] [%s] video — skipped (--skip-video)", entity_id)
                continue
            if video_only and signal_type != "video":
                continue
        processor = _KEMOCON_SIGNAL_PROCESSORS.get(signal_type)
        if processor is None:
            logger.warning("[K-EmoCon] [%s] No processor for signal '%s' — skipping", entity_id, signal_type)
            continue

        if (entity_id, entity_id, signal_type) in miss_skip:
            logger.info("[K-EmoCon] [%s] %s → excluded (total_missing)", entity_id, signal_type)
            continue

        file_obj = None
        if sig.get("filename"):
            file_obj = obj_by_fname.get(sig["filename"])
        elif sig.get("ext"):
            candidates = obj_by_ext.get(sig["ext"], [])
            file_obj = candidates[0] if candidates else None

        file_data: Optional[bytes] = None
        if file_obj is not None:
            file_data = download_object(minio_client, bucket, file_obj.object_name)

        declared_hz: Optional[float] = sig.get("declared_hz")
        sample_rate_hz = miss_lookup.get((entity_id, entity_id, signal_type)) or declared_hz

        try:
            if signal_type == "audio":
                file_id = file_obj.object_name.split("/")[-1] if file_obj else "audio"
                df = processor(file_data, entity_id, debate_duration_s, qf_cfg, file_id=file_id)
            elif signal_type == "video":
                df = processor(file_data, entity_id, debate_duration_s, qf_cfg)
            elif signal_type == "E4_IBI":
                df = processor(
                    file_data, entity_id, debate_start_s, debate_duration_s,
                    timestamp_col, ts_unit_ms, qf_cfg,
                )
            elif signal_type in ("E4_HR", "Polar_HR"):
                df = processor(
                    file_data, entity_id, debate_start_s, debate_duration_s,
                    timestamp_col, ts_unit_ms, sample_rate_hz, qf_cfg,
                    bvp_df=computed_dfs.get("E4_BVP") if signal_type == "E4_HR" else None,
                )
            else:
                df = processor(
                    file_data, entity_id, debate_start_s, debate_duration_s,
                    timestamp_col, ts_unit_ms, sample_rate_hz, qf_cfg,
                )
        except Exception as e:
            logger.error("[K-EmoCon] [%s] %s: processor failed: %s", entity_id, signal_type, e)
            continue

        if df is None or df.empty:
            logger.warning("[K-EmoCon] [%s] %s: no output produced", entity_id, signal_type)
            continue

        computed_dfs[signal_type] = df

        if signal_type == "audio" and file_obj is not None:
            stem = Path(file_obj.object_name.split("/")[-1]).stem
            output_key = (
                f"{output_prefix}/k-emocon/files/entity={entity_id}"
                f"/modality={modality}/{stem}_quality_flags.csv"
            )
        else:
            output_key = (
                f"{output_prefix}/k-emocon/files/entity={entity_id}"
                f"/modality={modality}/{_output_filename(entity_id, signal_type)}"
            )
        upload_csv(minio_client, bucket, output_key, df)
        logger.info("[K-EmoCon] [%s] %-12s → uploaded %s", entity_id, signal_type, output_key)
