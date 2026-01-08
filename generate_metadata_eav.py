from minio import Minio
import uuid
import psycopg2
from datetime import datetime

BUCKET = "raw"
PREFIX = "eav/"

MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

PG_CONN = {
    "host": "localhost",
    "port": 5432,
    "dbname": "datalakehouse",
    "user": "postgres",
    "password": "abc123"
}

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

pg = psycopg2.connect(**PG_CONN)
cursor = pg.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS raw_files_metadata (
        file_id UUID PRIMARY KEY,
        dataset TEXT NOT NULL,
        subject_id TEXT,
        modality TEXT,
        file_path TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_extension TEXT,
        file_size_bytes BIGINT,
        checksum_sha256 TEXT,
        ingestion_time TIMESTAMP,
        source_system TEXT
    );
""")


objects = minio_client.list_objects(
    BUCKET,
    prefix=PREFIX,
    recursive=True
)

for obj in objects:
    path_parts = obj.object_name.split("/")

    subject_id = None
    modality = None

    if "subjects" in path_parts:
        subject_id = path_parts[path_parts.index("subjects") + 1]
        modality = path_parts[path_parts.index(subject_id) + 1]

    cursor.execute("""
        INSERT INTO raw_files_metadata (
            file_id, dataset, subject_id, modality,
            file_path, file_name, file_extension,
            file_size_bytes, ingestion_time, source_system
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        str(uuid.uuid4()),
        "EAV",
        subject_id,
        modality,
        f"s3a://{BUCKET}/{obj.object_name}",
        obj.object_name.split("/")[-1],
        obj.object_name.split(".")[-1],
        obj.size,
        datetime.utcnow(),
        "EAV_original"
    ))

pg.commit()
cursor.close()
pg.close()