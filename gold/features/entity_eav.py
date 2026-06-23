import logging
import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from minio_utils import download_object, upload_parquet

from .audio import extract_audio_egemaps_windowed
from .eeg import extract_eeg_mne_features
from .video import extract_video_openface

logger = logging.getLogger(__name__)


def _trial_id_from_filename(filename: str) -> Optional[int]:
    m = re.match(r"^(\d+)", filename)
    return int(m.group(1)) if m else None


def process_eav_entity(
    minio_client,
    silver_bucket: str,
    gold_bucket: str,
    entity_id: str,
    silver_files_prefix: str,
    output_prefix: str,
    smile: Optional[Any],
    eeg_sig_cfg: Dict[str, Any],
    window_size_s: float = 0.3,
    trial_duration_s: float = 20.0,
    openface_available: bool = True,
    modalities: Optional[Set[str]] = None,
) -> None:
    if modalities is None:
        modalities = {"audio", "video", "eeg"}
    entity_prefix = f"{silver_files_prefix}/entity={entity_id}"

    # ── audio ──
    if "audio" in modalities and smile is not None:
        audio_frames: List[pd.DataFrame] = []
        audio_prefix = f"{entity_prefix}/modality=audio/"
        for obj in minio_client.list_objects(silver_bucket, prefix=audio_prefix, recursive=True):
            fname = obj.object_name.split("/")[-1]
            if not fname.lower().endswith(".wav"):
                continue
            trial_id = _trial_id_from_filename(fname)
            audio_bytes = download_object(minio_client, silver_bucket, obj.object_name)
            if audio_bytes is None:
                continue
            df = extract_audio_egemaps_windowed(audio_bytes, fname, smile, window_size_s, max_duration_s=trial_duration_s)
            if df is not None:
                df.insert(1, "entity_id", entity_id)
                if trial_id is not None:
                    df.insert(2, "trial_id", trial_id)
                audio_frames.append(df)

        if audio_frames:
            combined = pd.concat(audio_frames, ignore_index=True)
            key = f"{output_prefix}/eav/{entity_id}/{entity_id}_audio.parquet"
            upload_parquet(minio_client, gold_bucket, key, combined)
            logger.info("[EAV] [%s] audio → %d trials → %s/%s", entity_id, len(audio_frames), gold_bucket, key)
        else:
            logger.warning("[EAV] [%s] No audio features extracted", entity_id)

    # ── video ──
    if "video" in modalities and openface_available:
        video_frames: List[pd.DataFrame] = []
        video_prefix = f"{entity_prefix}/modality=video/"
        for obj in minio_client.list_objects(silver_bucket, prefix=video_prefix, recursive=True):
            fname = obj.object_name.split("/")[-1]
            if not fname.lower().endswith(".mp4"):
                continue
            trial_id = _trial_id_from_filename(fname)
            video_bytes = download_object(minio_client, silver_bucket, obj.object_name)
            if video_bytes is None:
                continue
            df = extract_video_openface(video_bytes, fname, window_size_s=window_size_s, max_duration_s=trial_duration_s)
            if df is not None:
                df.insert(1, "entity_id", entity_id)
                if trial_id is not None:
                    df.insert(2, "trial_id", trial_id)
                video_frames.append(df)

        if video_frames:
            combined = pd.concat(video_frames, ignore_index=True)
            key = f"{output_prefix}/eav/{entity_id}/{entity_id}_video.parquet"
            upload_parquet(minio_client, gold_bucket, key, combined)
            logger.info("[EAV] [%s] video → %d trials → %s/%s", entity_id, len(video_frames), gold_bucket, key)
        else:
            logger.warning("[EAV] [%s] No video features extracted", entity_id)

    # ── eeg ──
    if "eeg" not in modalities:
        return
    eeg_prefix = f"{entity_prefix}/modality=eeg/"
    mat_obj = None
    for obj in minio_client.list_objects(silver_bucket, prefix=eeg_prefix, recursive=True):
        if obj.object_name.lower().endswith(".mat"):
            mat_obj = obj
            break

    if mat_obj is not None:
        from silver.quality_flags.signal_readers import load_mat_eeg

        mat_bytes = download_object(minio_client, silver_bucket, mat_obj.object_name)
        if mat_bytes is not None:
            eeg_arr = load_mat_eeg(mat_bytes)
            if eeg_arr is not None:
                df = extract_eeg_mne_features(
                    eeg_arr,
                    entity_id,
                    sfreq=float(eeg_sig_cfg.get("declared_hz", 500.0)),
                    n_instances=int(eeg_sig_cfg.get("expected_instances", 200)),
                    n_timepoints=int(eeg_sig_cfg["expected_timepoints"]) if eeg_sig_cfg.get("expected_timepoints") else None,
                    window_size_s=window_size_s,
                    tp_axis=int(eeg_sig_cfg.get("timepoints_axis", 0)),
                    inst_axis=int(eeg_sig_cfg.get("instances_axis", 2)),
                )
                if df is not None:
                    key = f"{output_prefix}/eav/{entity_id}/{entity_id}_eeg.parquet"
                    upload_parquet(minio_client, gold_bucket, key, df)
                    logger.info("[EAV] [%s] eeg → %d trials → %s/%s", entity_id, len(df), gold_bucket, key)
            else:
                logger.warning("[EAV] [%s] EEG .mat could not be loaded", entity_id)
    else:
        logger.warning("[EAV] [%s] No .mat file found for EEG", entity_id)
