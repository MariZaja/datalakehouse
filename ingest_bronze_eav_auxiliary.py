import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from pyspark.sql.functions import current_timestamp
import config

load_dotenv()

LOCAL_EAV_PATH = os.getenv("LOCAL_EAV_PATH")
BUCKET = os.getenv("BUCKET_BRONZE")
BASE_PATH = os.getenv("BASE_PATH_EAV")
DATASET = "EAV"

minio_client, spark = config.config()

records = []

aux_files = {
    "meta_data.csv": "metadata",
    "subjects.csv": "metadata",
    "questionnaire.xlsx": "annotations"
}

for file_name, category in aux_files.items():
    local_file = os.path.join(LOCAL_EAV_PATH, file_name)
    if not os.path.exists(local_file):
        continue

    s3_path = f"{BASE_PATH}/auxiliary/{category}/{file_name}"

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

print("✅ EAV auxiliary metadata ingested to BRONZE.")