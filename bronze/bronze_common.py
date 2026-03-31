import os
from typing import List, NamedTuple

from minio import Minio


class FileEntry(NamedTuple):
    local_path: str
    s3_path: str
    dataset: str
    entity_id: str
    modality: str
    file_name: str
    table: str


def env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise ValueError(f"Required environment variable {key!r} is not set.")
    return val


def get_minio_client() -> Minio:
    return Minio(
        env("MINIO_ENDPOINT"),
        access_key=env("MINIO_ACCESS_KEY"),
        secret_key=env("MINIO_SECRET_KEY"),
        secure=False,
    )


def build_eav_manifest(eav_root: str, base_path: str) -> List[FileEntry]:
    entries: List[FileEntry] = []
    modality_map = {"Audio": "audio", "Video": "video", "EEG": "eeg"}

    for item in sorted(os.listdir(eav_root)):
        subject_path = os.path.join(eav_root, item)
        if not item.startswith("subject") or not os.path.isdir(subject_path):
            continue
        for src_mod, dst_mod in modality_map.items():
            src_dir = os.path.join(subject_path, src_mod)
            if not os.path.isdir(src_dir):
                continue
            for file_name in sorted(os.listdir(src_dir)):
                if file_name.startswith("."):
                    continue
                local_file = os.path.join(src_dir, file_name)
                if not os.path.isfile(local_file):
                    continue
                s3_path = f"{base_path}/files/entity={item}/modality={dst_mod}/{file_name}"
                entries.append(FileEntry(
                    local_path=local_file, s3_path=s3_path,
                    dataset="EAV", entity_id=item,
                    modality=dst_mod, file_name=file_name, table="files",
                ))

    aux_root = os.path.dirname(eav_root.rstrip("/\\"))
    aux_files = {
        "meta_data.csv": "metadata",
        "subjects.csv": "metadata",
        "questionnaire.xlsx": "annotations",
    }
    for file_name, category in aux_files.items():
        local_file = os.path.join(aux_root, file_name)
        if not os.path.exists(local_file):
            continue
        s3_path = f"{base_path}/auxiliary/{category}/{file_name}"
        entries.append(FileEntry(
            local_path=local_file, s3_path=s3_path,
            dataset="EAV", entity_id="global",
            modality=category, file_name=file_name, table="aux",
        ))

    return entries


def build_kemocon_manifest(kemocon_root: str, base_path: str) -> List[FileEntry]:
    entries: List[FileEntry] = []

    audio_dir = os.path.join(kemocon_root, "debate_audios")
    if os.path.isdir(audio_dir):
        for file_name in sorted(os.listdir(audio_dir)):
            if not file_name.endswith(".wav") or file_name.startswith("."):
                continue
            entity_id = file_name.replace(".wav", "").replace(".", "_")
            local_file = os.path.join(audio_dir, file_name)
            s3_path = f"{base_path}/files/entity={entity_id}/modality=audio/{file_name}"
            entries.append(FileEntry(
                local_path=local_file, s3_path=s3_path,
                dataset="K-EmoCon", entity_id=entity_id,
                modality="audio", file_name=file_name, table="files",
            ))

    video_dir = os.path.join(kemocon_root, "debate_recordings")
    if os.path.isdir(video_dir):
        for file_name in sorted(os.listdir(video_dir)):
            if not file_name.endswith(".mp4") or file_name.startswith("."):
                continue
            entity_id = file_name.split("_")[0]
            local_file = os.path.join(video_dir, file_name)
            s3_path = f"{base_path}/files/entity={entity_id}/modality=video/{file_name}"
            entries.append(FileEntry(
                local_path=local_file, s3_path=s3_path,
                dataset="K-EmoCon", entity_id=entity_id,
                modality="video", file_name=file_name, table="files",
            ))

    e4_dir = os.path.join(kemocon_root, "e4_data")
    if os.path.isdir(e4_dir):
        for participant_id in sorted(os.listdir(e4_dir)):
            part_dir = os.path.join(e4_dir, participant_id)
            if not os.path.isdir(part_dir) or participant_id.startswith("."):
                continue
            for file_name in sorted(os.listdir(part_dir)):
                if file_name.startswith("."):
                    continue
                local_file = os.path.join(part_dir, file_name)
                if not os.path.isfile(local_file):
                    continue
                s3_path = f"{base_path}/files/entity={participant_id}/modality=biosignal/{file_name}"
                entries.append(FileEntry(
                    local_path=local_file, s3_path=s3_path,
                    dataset="K-EmoCon", entity_id=participant_id,
                    modality="biosignal", file_name=file_name, table="files",
                ))

    eeg_dir = os.path.join(kemocon_root, "neurosky_polar_data")
    if os.path.isdir(eeg_dir):
        for participant_id in sorted(os.listdir(eeg_dir)):
            part_dir = os.path.join(eeg_dir, participant_id)
            if not os.path.isdir(part_dir) or participant_id.startswith("."):
                continue
            for file_name in sorted(os.listdir(part_dir)):
                if file_name.startswith("."):
                    continue
                local_file = os.path.join(part_dir, file_name)
                if not os.path.isfile(local_file):
                    continue
                s3_path = f"{base_path}/files/entity={participant_id}/modality=eeg/{file_name}"
                entries.append(FileEntry(
                    local_path=local_file, s3_path=s3_path,
                    dataset="K-EmoCon", entity_id=participant_id,
                    modality="eeg", file_name=file_name, table="files",
                ))

    aux_dirs = {
        "emotion_annotations": "annotations",
        "data_quality_tables": "quality",
        "metadata": "metadata",
    }
    for src_dir_name, category in aux_dirs.items():
        full_dir = os.path.join(kemocon_root, src_dir_name)
        if not os.path.isdir(full_dir):
            continue
        for root, _, files in os.walk(full_dir):
            for file_name in sorted(files):
                if file_name.startswith("."):
                    continue
                local_file = os.path.join(root, file_name)
                relative_path = os.path.relpath(local_file, kemocon_root)
                s3_path = f"{base_path}/auxiliary/{category}/{relative_path}"
                entries.append(FileEntry(
                    local_path=local_file, s3_path=s3_path,
                    dataset="K-EmoCon", entity_id="global",
                    modality=category, file_name=file_name, table="aux",
                ))

    return entries
