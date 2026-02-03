import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from minio import Minio
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp
load_dotenv()

LOCAL_KEMOCON_PATH = os.getenv("LOCAL_KEMOCON_PATH")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

BUCKET = os.getenv("BUCKET_BRONZE")
BASE_PATH = os.getenv("BASE_PATH_KEMOCON")
DATASET = "K-EmoCon"

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

modality_dirs = {
    "debate_audios": "audio",
    "debate_recordings": "video",
    "neurosky_polar_data": "eeg",
    "e4_data": "biosignal"
}

# ====================AUDIO (debate_audios)=================================

audio_dir = os.path.join(LOCAL_KEMOCON_PATH, "debate_audios")
if os.path.exists(audio_dir):
    for file_name in os.listdir(audio_dir):
        if not file_name.endswith(".wav"):
            continue

        # p1.p2.wav -> p1_p2
        entity_id = file_name.replace(".wav", "").replace(".", "_")
        local_file = os.path.join(audio_dir, file_name)

        s3_path = f"{BASE_PATH}/files/entity={entity_id}/modality=audio/{file_name}"
        minio_client.fput_object(BUCKET, s3_path, local_file)

        records.append((
            str(uuid.uuid4()), DATASET, entity_id, "audio",
            f"s3://{BUCKET}/{s3_path}", file_name, datetime.utcnow()
        ))

# ====================VIDEO (debate_recordings)=================================

video_dir = os.path.join(LOCAL_KEMOCON_PATH, "debate_recordings")
if os.path.exists(video_dir):
    for file_name in os.listdir(video_dir):
        if not file_name.endswith(".mp4"):
            continue

        # p2_854.mp4 -> p2
        entity_id = file_name.split("_")[0]
        local_file = os.path.join(video_dir, file_name)

        s3_path = f"{BASE_PATH}/files/entity={entity_id}/modality=video/{file_name}"
        minio_client.fput_object(BUCKET, s3_path, local_file)

        records.append((
            str(uuid.uuid4()), DATASET, entity_id, "video",
            f"s3://{BUCKET}/{s3_path}", file_name, datetime.utcnow()
        ))

# ====================BIOSIGNALS=================================

e4_dir = os.path.join(LOCAL_KEMOCON_PATH, "e4_data")
if os.path.exists(e4_dir):
    for participant_id in os.listdir(e4_dir):
        part_dir = os.path.join(e4_dir, participant_id)
        if not os.path.isdir(part_dir):
            continue

        for file_name in os.listdir(part_dir):
            local_file = os.path.join(part_dir, file_name)

            s3_path = f"{BASE_PATH}/files/entity={participant_id}/modality=biosignal/{file_name}"
            minio_client.fput_object(BUCKET, s3_path, local_file)

            records.append((
                str(uuid.uuid4()), DATASET, participant_id, "biosignal",
                f"s3://{BUCKET}/{s3_path}", file_name, datetime.utcnow()
            ))

# ====================EEG (neurosky_polar_data)=================================           

eeg_dir = os.path.join(LOCAL_KEMOCON_PATH, "neurosky_polar_data")
if os.path.exists(eeg_dir):
    for participant_id in os.listdir(eeg_dir):
        part_dir = os.path.join(eeg_dir, participant_id)
        if not os.path.isdir(part_dir):
            continue

        for file_name in os.listdir(part_dir):
            local_file = os.path.join(part_dir, file_name)

            s3_path = f"{BASE_PATH}/files/entity={participant_id}/modality=eeg/{file_name}"
            minio_client.fput_object(BUCKET, s3_path, local_file)

            records.append((
                str(uuid.uuid4()), DATASET, participant_id, "eeg",
                f"s3://{BUCKET}/{s3_path}", file_name, datetime.utcnow()
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
    .save(f"s3a://{BUCKET}/{BASE_PATH}/delta/k_emocon_files_metadata")
)

print("✅ BRONZE ingest for K-EmoCon completed successfully.")