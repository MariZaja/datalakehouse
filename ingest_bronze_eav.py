import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from minio import Minio
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

load_dotenv()
# Lokalna ścieżka do danych źródłowych
LOCAL_EAV_PATH = os.getenv("LOCAL_EAV_PATH")

# MinIO / S3
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET = os.getenv("BUCKET_BRONZE")
BASE_PATH = os.getenv("BASE_PATH_EAV")

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

spark = (
    SparkSession.builder
    .appName("bronze-eav-ingest")
    .master("local[*]")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config(
        "spark.jars.packages",
        "io.delta:delta-spark_2.12:3.2.0,"
        "org.apache.hadoop:hadoop-aws:3.3.4,"
        "com.amazonaws:aws-java-sdk-bundle:1.12.316")

    .config("spark.hadoop.fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
    .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
    .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .config("spark.hadoop.io.native.lib.available", "false")

    .getOrCreate()
)

records = []

for item in os.listdir(LOCAL_EAV_PATH):
    subject_path = os.path.join(LOCAL_EAV_PATH, item)

    if item.startswith("subject") and os.path.isdir(subject_path):
        subject_id = item

        modality_map = {
            "Audio": "audio",
            "Video": "video",
            "EEG": "eeg"
        }

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
                    f"subject={subject_id}/"
                    f"modality={dst_modality}/"
                    f"{file_name}"
                )

                # Upload binary file to BRONZE
                minio_client.fput_object(
                    BUCKET,
                    s3_path,
                    local_file
                )

                # Collect metadata record
                records.append((
                    str(uuid.uuid4()),
                    "EAV",
                    subject_id,
                    dst_modality,
                    f"s3://{BUCKET}/{s3_path}",
                    file_name,
                    datetime.now()
                ))

df = spark.createDataFrame(
    records,
    [
        "file_id",
        "dataset",
        "subject_id",
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
    .partitionBy("subject_id", "modality")
    .save(f"s3a://{BUCKET}/{BASE_PATH}/delta/eav_files_metadata")
)

print("✅ BRONZE ingest for EAV completed successfully.")