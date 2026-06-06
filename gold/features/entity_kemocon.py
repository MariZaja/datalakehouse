import logging
from typing import Any, Dict, Optional, Set

import pandas as pd

from minio_utils import download_object, upload_parquet

from .audio import extract_audio_egemaps_windowed
from .e4_physio import (
    extract_acc_features,
    extract_bvp_features,
    extract_e4_hr_features,
    extract_eda_features,
    extract_ibi_features,
    extract_polar_hr_features,
    extract_temp_features,
)
from .eeg import extract_kemocon_brainwave, extract_kemocon_scalar_windowed
from .video import extract_video_openface

logger = logging.getLogger(__name__)


def process_kemocon_entity(
    minio_client,
    silver_bucket: str,
    gold_bucket: str,
    entity_id: str,
    silver_files_prefix: str,
    output_prefix: str,
    smile: Optional[Any],
    subjects_map: Dict[int, Dict[str, int]],
    window_size_s: float = 0.3,
    hr_ibi_window_size_s: float = 10.0,
    hr_ibi_step_s: float = 0.3,
    eda_temp_window_size_s: float = 30.0,
    eda_temp_step_s: float = 0.3,
    openface_available: bool = True,
    modalities: Optional[Set[str]] = None,
) -> None:
    if modalities is None:
        modalities = {"audio", "video", "eeg", "bvp", "acc", "hr", "ibi", "eda", "temp"}
    entity_prefix = f"{silver_files_prefix}/entity={entity_id}"

    pid = int(entity_id[1:])
    subject = subjects_map.get(pid)
    if subject is not None:
        debate_start_ms = float(subject["startTime"])
        debate_end_ms = float(subject["endTime"])
        debate_duration_s = (debate_end_ms - debate_start_ms) / 1000.0
    else:
        logger.warning("[K-EmoCon] [%s] No subject entry — debate window unavailable", entity_id)
        debate_start_ms = None
        debate_end_ms = None
        debate_duration_s = None

    # ── audio ──
    if "audio" in modalities and smile is not None:
        audio_prefix = f"{entity_prefix}/modality=audio/"
        for obj in minio_client.list_objects(silver_bucket, prefix=audio_prefix, recursive=True):
            fname = obj.object_name.split("/")[-1]
            if not fname.lower().endswith(".wav"):
                continue
            audio_bytes = download_object(minio_client, silver_bucket, obj.object_name)
            if audio_bytes is None:
                continue
            df = extract_audio_egemaps_windowed(
                audio_bytes, fname, smile, window_size_s, max_duration_s=debate_duration_s,
            )
            if df is not None:
                df.insert(1, "entity_id", entity_id)
                key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_audio.parquet"
                upload_parquet(minio_client, gold_bucket, key, df)
                logger.info("[K-EmoCon] [%s] audio → %s/%s", entity_id, gold_bucket, key)
            break  # one audio file per entity

    # ── video ──
    if "video" in modalities and openface_available:
        video_prefix = f"{entity_prefix}/modality=video/"
        for obj in minio_client.list_objects(silver_bucket, prefix=video_prefix, recursive=True):
            fname = obj.object_name.split("/")[-1]
            if not fname.lower().endswith(".mp4"):
                continue
            video_bytes = download_object(minio_client, silver_bucket, obj.object_name)
            if video_bytes is None:
                continue
            df = extract_video_openface(
                video_bytes, fname, max_duration_s=debate_duration_s, window_size_s=window_size_s,
            )
            if df is not None:
                df.insert(1, "entity_id", entity_id)
                key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_video.parquet"
                upload_parquet(minio_client, gold_bucket, key, df)
                logger.info("[K-EmoCon] [%s] video → %s/%s", entity_id, gold_bucket, key)
            break  # one video file per entity

    # ── eeg: BrainWave band powers + Attention + Meditation ──
    if "eeg" in modalities:
        eeg_prefix = f"{entity_prefix}/modality=eeg/"
        eeg_files: Dict[str, bytes] = {}
        for obj in minio_client.list_objects(silver_bucket, prefix=eeg_prefix, recursive=True):
            fname = obj.object_name.split("/")[-1]
            if fname in ("BrainWave.csv", "Attention.csv", "Meditation.csv"):
                data = download_object(minio_client, silver_bucket, obj.object_name)
                if data is not None:
                    eeg_files[fname] = data

        if "BrainWave.csv" not in eeg_files:
            logger.warning("[K-EmoCon] [%s] BrainWave.csv not found — skipping EEG", entity_id)
        else:
            df = extract_kemocon_brainwave(
                eeg_files["BrainWave.csv"],
                entity_id,
                window_size_s,
                debate_start_ms,
                debate_end_ms,
            )
            if df is not None:
                key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_eeg.parquet"
                upload_parquet(minio_client, gold_bucket, key, df)
                logger.info("[K-EmoCon] [%s] eeg → %s/%s", entity_id, gold_bucket, key)

        if "Attention.csv" in eeg_files:
            df_att = extract_kemocon_scalar_windowed(
                eeg_files["Attention.csv"], entity_id, "attention",
                debate_start_ms, debate_end_ms,
                window_size_s=hr_ibi_window_size_s, step_s=hr_ibi_step_s,
            )
            if df_att is not None:
                key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_attention.parquet"
                upload_parquet(minio_client, gold_bucket, key, df_att)
                logger.info("[K-EmoCon] [%s] attention → %s/%s", entity_id, gold_bucket, key)
            else:
                logger.warning("[K-EmoCon] [%s] No attention features extracted", entity_id)
        else:
            logger.warning("[K-EmoCon] [%s] Attention.csv not found", entity_id)

        if "Meditation.csv" in eeg_files:
            df_med = extract_kemocon_scalar_windowed(
                eeg_files["Meditation.csv"], entity_id, "meditation",
                debate_start_ms, debate_end_ms,
                window_size_s=hr_ibi_window_size_s, step_s=hr_ibi_step_s,
            )
            if df_med is not None:
                key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_meditation.parquet"
                upload_parquet(minio_client, gold_bucket, key, df_med)
                logger.info("[K-EmoCon] [%s] meditation → %s/%s", entity_id, gold_bucket, key)
            else:
                logger.warning("[K-EmoCon] [%s] No meditation features extracted", entity_id)
        else:
            logger.warning("[K-EmoCon] [%s] Meditation.csv not found", entity_id)

    # ── physio: BVP, ACC, HR, IBI, EDA, TEMP (Empatica E4) ──
    physio_modalities = modalities & {"bvp", "acc", "hr", "ibi", "eda", "temp"}
    if physio_modalities:
        physio_prefix = f"{entity_prefix}/modality=biosignal/"
        wanted: Set[str] = set()
        if "bvp" in physio_modalities:
            wanted.add("E4_BVP.csv")
        if "acc" in physio_modalities:
            wanted.add("E4_ACC.csv")
        if "hr" in physio_modalities:
            wanted.add("E4_HR.csv")
        if "ibi" in physio_modalities:
            wanted.add("E4_IBI.csv")
        if "eda" in physio_modalities:
            wanted.add("E4_EDA.csv")
        if "temp" in physio_modalities:
            wanted.add("E4_TEMP.csv")

        physio_files: Dict[str, bytes] = {}
        for obj in minio_client.list_objects(silver_bucket, prefix=physio_prefix, recursive=True):
            fname = obj.object_name.split("/")[-1]
            if fname in wanted:
                data = download_object(minio_client, silver_bucket, obj.object_name)
                if data is not None:
                    physio_files[fname] = data

        if "bvp" in physio_modalities:
            if "E4_BVP.csv" in physio_files:
                df = extract_bvp_features(
                    physio_files["E4_BVP.csv"], entity_id, window_size_s,
                    debate_start_ms=debate_start_ms, debate_end_ms=debate_end_ms,
                )
                if df is not None:
                    key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_bvp.parquet"
                    upload_parquet(minio_client, gold_bucket, key, df)
                    logger.info("[K-EmoCon] [%s] bvp → %s/%s", entity_id, gold_bucket, key)
                else:
                    logger.warning("[K-EmoCon] [%s] No BVP features extracted", entity_id)
            else:
                logger.warning("[K-EmoCon] [%s] E4_BVP.csv not found — skipping BVP", entity_id)

        if "acc" in physio_modalities:
            if "E4_ACC.csv" in physio_files:
                df = extract_acc_features(
                    physio_files["E4_ACC.csv"], entity_id, window_size_s,
                    debate_start_ms=debate_start_ms, debate_end_ms=debate_end_ms,
                )
                if df is not None:
                    key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_acc.parquet"
                    upload_parquet(minio_client, gold_bucket, key, df)
                    logger.info("[K-EmoCon] [%s] acc → %s/%s", entity_id, gold_bucket, key)
                else:
                    logger.warning("[K-EmoCon] [%s] No ACC features extracted", entity_id)
            else:
                logger.warning("[K-EmoCon] [%s] E4_ACC.csv not found — skipping ACC", entity_id)

        if "hr" in physio_modalities:
            if "E4_HR.csv" in physio_files:
                df = extract_e4_hr_features(
                    physio_files["E4_HR.csv"], entity_id,
                    window_size_s=hr_ibi_window_size_s, step_s=hr_ibi_step_s,
                    debate_start_ms=debate_start_ms, debate_end_ms=debate_end_ms,
                )
                if df is not None:
                    key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_e4_hr.parquet"
                    upload_parquet(minio_client, gold_bucket, key, df)
                    logger.info("[K-EmoCon] [%s] e4_hr → %s/%s", entity_id, gold_bucket, key)
                else:
                    logger.warning("[K-EmoCon] [%s] No E4 HR features extracted", entity_id)
            else:
                logger.warning("[K-EmoCon] [%s] E4_HR.csv not found — skipping E4 HR", entity_id)

        if "ibi" in physio_modalities:
            if "E4_IBI.csv" in physio_files:
                df = extract_ibi_features(
                    physio_files["E4_IBI.csv"], entity_id,
                    window_size_s=hr_ibi_window_size_s, step_s=hr_ibi_step_s,
                    debate_start_ms=debate_start_ms, debate_end_ms=debate_end_ms,
                )
                if df is not None:
                    key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_ibi.parquet"
                    upload_parquet(minio_client, gold_bucket, key, df)
                    logger.info("[K-EmoCon] [%s] ibi → %s/%s", entity_id, gold_bucket, key)
                else:
                    logger.warning("[K-EmoCon] [%s] No IBI features extracted", entity_id)
            else:
                logger.warning("[K-EmoCon] [%s] E4_IBI.csv not found — skipping IBI", entity_id)

        if "eda" in physio_modalities:
            if "E4_EDA.csv" in physio_files:
                df = extract_eda_features(
                    physio_files["E4_EDA.csv"], entity_id,
                    window_size_s=eda_temp_window_size_s, step_s=eda_temp_step_s,
                    debate_start_ms=debate_start_ms, debate_end_ms=debate_end_ms,
                )
                if df is not None:
                    key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_eda.parquet"
                    upload_parquet(minio_client, gold_bucket, key, df)
                    logger.info("[K-EmoCon] [%s] eda → %s/%s", entity_id, gold_bucket, key)
                else:
                    logger.warning("[K-EmoCon] [%s] No EDA features extracted", entity_id)
            else:
                logger.warning("[K-EmoCon] [%s] E4_EDA.csv not found — skipping EDA", entity_id)

        if "temp" in physio_modalities:
            if "E4_TEMP.csv" in physio_files:
                df = extract_temp_features(
                    physio_files["E4_TEMP.csv"], entity_id,
                    window_size_s=eda_temp_window_size_s, step_s=eda_temp_step_s,
                    debate_start_ms=debate_start_ms, debate_end_ms=debate_end_ms,
                )
                if df is not None:
                    key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_temp.parquet"
                    upload_parquet(minio_client, gold_bucket, key, df)
                    logger.info("[K-EmoCon] [%s] temp → %s/%s", entity_id, gold_bucket, key)
                else:
                    logger.warning("[K-EmoCon] [%s] No TEMP features extracted", entity_id)
            else:
                logger.warning("[K-EmoCon] [%s] E4_TEMP.csv not found — skipping TEMP", entity_id)

    # ── Polar HR (stored with NeuroSky data in modality=eeg) ──
    if "hr" in modalities:
        polar_prefix = f"{entity_prefix}/modality=eeg/"
        for obj in minio_client.list_objects(silver_bucket, prefix=polar_prefix, recursive=True):
            if obj.object_name.split("/")[-1] == "Polar_HR.csv":
                polar_data = download_object(minio_client, silver_bucket, obj.object_name)
                if polar_data is not None:
                    df = extract_polar_hr_features(
                        polar_data, entity_id,
                        window_size_s=hr_ibi_window_size_s, step_s=hr_ibi_step_s,
                        debate_start_ms=debate_start_ms, debate_end_ms=debate_end_ms,
                    )
                    if df is not None:
                        key = f"{output_prefix}/kemocon/{entity_id}/{entity_id}_polar_hr.parquet"
                        upload_parquet(minio_client, gold_bucket, key, df)
                        logger.info("[K-EmoCon] [%s] polar_hr → %s/%s", entity_id, gold_bucket, key)
                    else:
                        logger.warning("[K-EmoCon] [%s] No Polar HR features extracted", entity_id)
                break
