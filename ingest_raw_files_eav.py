from minio import Minio
import os

LOCAL_EAV_SUBJECTS_PATH = "./data/EAV/EAV"
LOCAL_EAV_PATH = "./data/EAV"
BUCKET = "raw"
BASE_PREFIX = "eav"

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def upload_file(local_path, s3_path):
    client.fput_object(
        BUCKET,
        s3_path,
        local_path
    )
    print(f"Uploaded {local_path} -> s3://{BUCKET}/{s3_path}")

# INGEST SUBJECT DATA
for item in os.listdir(LOCAL_EAV_SUBJECTS_PATH):
    item_path = os.path.join(LOCAL_EAV_SUBJECTS_PATH, item)

    if item.startswith("subject") and os.path.isdir(item_path):
        subject_id = item

        for modality in ["Audio", "Video", "EEG"]:
            src_dir = os.path.join(item_path, modality)
            if not os.path.exists(src_dir):
                continue

            for file in os.listdir(src_dir):
                local_file = os.path.join(src_dir, file)

                s3_path = (
                    f"{BASE_PREFIX}/subjects/"
                    f"{subject_id}/"
                    f"{modality.lower()}/"
                    f"{file}"
                )

                upload_file(local_file, s3_path)

# INGEST GLOBAL ANNOTATIONS
for file in os.listdir(LOCAL_EAV_PATH):
    if file.endswith((".csv", ".xlsx")):
        local_file = os.path.join(LOCAL_EAV_PATH, file)
        s3_path = f"{BASE_PREFIX}/annotations/{file}"
        upload_file(local_file, s3_path)
