import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from pyspark.sql.functions import current_timestamp
import config

load_dotenv()

minio_client, spark = config.config()

LOCAL_KEMOCON_PATH = os.getenv("LOCAL_KEMOCON_PATH")

BUCKET = os.getenv("BUCKET_BRONZE")
BASE_PATH = os.getenv("BASE_PATH_KEMOCON")
DATASET = "K-EmoCon"

records = []

aux_dirs = {
    "emotion_annotations": "annotations",
    "data_quality_tables": "quality",
    "metadata": "metadata"
}

for src_dir, category in aux_dirs.items():
    full_dir = os.path.join(LOCAL_KEMOCON_PATH, src_dir)
    if not os.path.exists(full_dir):
        continue

    for root, _, files in os.walk(full_dir):
        for file_name in files:
            local_file = os.path.join(root, file_name)

            relative_path = os.path.relpath(local_file, LOCAL_KEMOCON_PATH)
            s3_path = f"{BASE_PATH}/auxiliary/{category}/{relative_path}"

            minio_client.fput_object(
                BUCKET,
                s3_path,
                local_file
            )

            records.append((
                str(uuid.uuid4()),
                DATASET,
                category,
                f"s3://{BUCKET}/{s3_path}",
                file_name,
                "global",
                datetime.utcnow()
            ))

df = spark.createDataFrame(
    records,
    [
        "record_id",
        "dataset",
        "category",
        "source_path",
        "file_name",
        "related_entity",
        "ingestion_time"
    ]
)

(
    df
    .withColumn("ingest_ts", current_timestamp())
    .write
    .format("delta")
    .mode("append")
    .partitionBy("dataset", "category")
    .save(f"s3a://{BUCKET}/{BASE_PATH}/delta/auxiliary_metadata")
)

print("✅ K-EmoCon auxiliary metadata ingested to BRONZE.")