import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from minio import Minio
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

load_dotenv()

LOCAL_EAV_PATH = os.getenv("LOCAL_EAV_PATH")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET = os.getenv("BUCKET_BRONZE")
BASE_PATH = os.getenv("BASE_PATH_EAV")
DATASET = "EAV"

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