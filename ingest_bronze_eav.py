import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from pyspark.sql.functions import current_timestamp
import config

load_dotenv()
# Lokalna ścieżka do danych źródłowych
LOCAL_EAV_PATH = os.getenv("LOCAL_EAV_PATH")
BUCKET = os.getenv("BUCKET_BRONZE")
BASE_PATH = os.getenv("BASE_PATH_EAV")

minio_client, spark = config.config()

records = []

modality_map = {
    "Audio": "audio",
    "Video": "video",
    "EEG": "eeg"
}

for item in os.listdir(LOCAL_EAV_PATH):
    subject_path = os.path.join(LOCAL_EAV_PATH, item)

    if item.startswith("subject") and os.path.isdir(subject_path):
        subject_id = item

        for src_modality, dst_modality in modality_map.items():
            src_dir = os.path.join(subject_path, src_modality)
            if not os.path.exists(src_dir):
                continue

            for file_name in os.listdir(src_dir):
                local_file = os.path.join(src_dir, file_name)

                if not os.path.isfile(local_file):
                    continue

                s3_path = (
                    f"{BASE_PATH}/files/"
                    f"entity={subject_id}/"
                    f"modality={dst_modality}/"
                    f"{file_name}"
                )

                minio_client.fput_object(
                    BUCKET,
                    s3_path,
                    local_file
                )

                records.append((
                    str(uuid.uuid4()),
                    "EAV",
                    subject_id,
                    dst_modality,
                    f"s3://{BUCKET}/{s3_path}",
                    file_name,
                    datetime.utcnow()
                ))

df = spark.createDataFrame(
    records,
    [
        "file_id",
        "dataset",
        "entity_id",
        "modality",
        "file_path",
        "file_name",
        "ingestion_time"
    ]
)

(
    df
    .withColumn("ingest_ts", current_timestamp())
    .write
    .format("delta")
    .mode("append")
    .partitionBy("entity_id", "modality")
    .save(f"s3a://{BUCKET}/{BASE_PATH}/delta/eav_files_metadata")
)

print("BRONZE ingest for EAV completed successfully.")