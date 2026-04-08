# Data Lakehouse for Multimodal Affective Datasets

A data lakehouse for two multimodal emotion recognition datasets: **EAV** and **K-EmoCon**. Built on a medallion architecture (Bronze → Silver → Gold) backed by MinIO object storage, Apache Spark, and Delta Lake.

Currently implemented: Bronze (raw ingestion and validation) and Silver (partially — entity resolution, time audit, missingness detection). Further Silver steps and the Gold layer — curated, analysis-ready feature tables — are in progress.

---

## Datasets

| Dataset | Modalities | Participants | Reference |
|---------|-----------|-------------|-----------|
| **EAV** | EEG (.mat), Audio (.wav), Video (.mp4) | 30 subjects | Lee et al. 2024 (Scientific Data) |
| **K-EmoCon** | Biosignals (Empatica E4), EEG (NeuroSky), Polar HR, Audio (.wav), Video (.mp4) | 32 dyads | Park et al. 2020 (Scientific Data) |

---

## Architecture

```
Local raw data
      │
      ▼
[Bronze layer]  ── MinIO bucket: bronze
      │
      ▼
[Silver layer]  ── MinIO bucket: silver
```

### Bronze layer (`bronze/`)

| Script | Purpose |
|--------|---------|
| `validate_raw.py` | Validates local raw files before upload (format, duration, sample rate, size) |
| `upload_bronze.py` | Uploads validated files to MinIO; builds Delta Lake metadata tables |
| `validate_bronze_integrity.py` | Post-upload integrity check (checksums, manifest completeness) |
| `bronze_common.py` | Shared utilities: MinIO client, manifest builders for EAV and K-EmoCon |

### Silver layer (`silver/`)

Steps run in order; each step reads from the previous step's output.

| Step | Script | Output prefix | Description |
|------|--------|--------------|-------------|
| 01 | *(entity resolution)* | `01_entity_resolution/` | Normalizes participant IDs; rewrites Delta tables |
| 02 | *(time audit)* | `02_time_audit/metadata/time_audit.csv` | Verifies signal properties (Hz, n_samples, duration) against declared specs |
| 03 | `silver_missingness_detection.py` | `03_missingness/` | Detects and catalogs missing or gapped signals |

---

## Configuration

All dataset-specific parameters live in [`pipeline_config.yaml`](pipeline_config.yaml) — no dataset logic is hardcoded in scripts. Adding a new dataset means adding a new section to the config.

Key top-level keys:

| Key | Purpose |
|-----|---------|
| `bucket` / `bucket_silver` | MinIO bucket names |
| `datasets.eav` / `datasets.kemocon` | Bronze paths, modalities, entity ID patterns |
| `time_audit.datasets.*` | Signal declarations for Step 02 (Hz, device, modality) |
| `bronze_validation.datasets.*` | Local validation rules (extensions, duration, size thresholds) |
| `missingness_detection.datasets.*` | Expected inventory, gap-detection flags, known-missing entries |

---

## Infrastructure

| Component | Role |
|-----------|------|
| **MinIO** | S3-compatible object store (bronze + silver buckets) |
| **Apache Spark + Delta Lake** | ACID metadata tables during bronze upload |
| **pandas / scipy / soundfile / cv2** | Signal inspection in silver steps (no Spark needed) |

Connection credentials are loaded from a `.env` file:

```
MINIO_ENDPOINT=...
MINIO_ACCESS_KEY=...
MINIO_SECRET_KEY=...
LOCAL_EAV_PATH=...
LOCAL_KEMOCON_PATH=...
```

---

## Running the pipeline

### Bronze — via orchestration script (recommended)

[`run_bronze_pipeline.sh`](run_bronze_pipeline.sh) runs all three bronze stages in sequence (fail-fast) and saves per-run reports to a timestamped `reports_<timestamp>/` directory.

```bash
# Single dataset (optional: override raw data path as second argument)
./run_bronze_pipeline.sh eav [/path/to/raw/EAV]
./run_bronze_pipeline.sh kemocon [/path/to/raw/K-EmoCon]

# Both datasets (uses LOCAL_EAV_PATH and LOCAL_KEMOCON_PATH from .env)
./run_bronze_pipeline.sh all
```

Stages:

| Stage | Description |
|-------|-------------|
| `validate_raw` | Checks local files before upload; aborts pipeline on failure |
| `upload_bronze` | Idempotent upload to MinIO; exit 2 = partial (continues to integrity check) |
| `validate_bronze_integrity` | Verifies MinIO contents match source; skipped if upload was a total failure |

Exit codes: `0` = all PASS, `1` = at least one stage FAILED.

### Bronze — manual (individual scripts)

```bash
cd bronze
python validate_raw.py --dataset eav
python upload_bronze.py --dataset eav
python validate_bronze_integrity.py --dataset eav
```

### Silver steps

```bash
# from project root
python silver/silver_entity_resolution.py
python silver/silver_time_audit.py
python silver/silver_missingness_detection.py
```

---

## Silver Step 03 — Missingness Detection outputs

| File | Content |
|------|---------|
| `03_missingness/missingness_report.csv` | One row per expected *(participant × signal × unit_id)*; columns: `status`, `missing_pct`, `reason`, … |
| `03_missingness/missingness_gaps.json` | Gap intervals (start/end seconds) for signals with `status = partial_missing` |

Known missing data (documented in the original papers) is declared in `pipeline_config.yaml` under `missingness_detection.datasets.<name>.known_missing` — not hardcoded in script logic.

---

## Project structure

```
project/
├── bronze/
│   ├── bronze_common.py
│   ├── upload_bronze.py
│   ├── validate_bronze_integrity.py
│   └── validate_raw.py
├── silver/
│   └── silver_missingness_detection.py
├── config.py                 # MinIO + Spark session factory
├── pipeline_config.yaml      # All dataset/signal/path configuration
└── data/                     # Local raw data (not committed)
    ├── EAV/
    └── K-EmoCon/
```
