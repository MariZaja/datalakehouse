#!/usr/bin/env bash
# run_bronze_pipeline.sh - Bronze ingestion pipeline orchestrator.
#
# Usage:
#   ./run_bronze_pipeline.sh <dataset> [raw_data_path]
#
# Arguments:
#   dataset        eav | kemocon | all
#   raw_data_path  (optional) path to the raw dataset root on this VM.
#                  For 'eav'    → sets LOCAL_EAV_PATH to this value.
#                  For 'kemocon'→ sets LOCAL_KEMOCON_PATH to this value.
#                  For 'all'    → ignored; use LOCAL_EAV_PATH and
#                                 LOCAL_KEMOCON_PATH env vars (or .env file).
#
# Pipeline stages (fail-fast):
#   1. validate_raw          - check local files before upload
#   2. upload_bronze         - idempotent upload to MinIO
#   3. validate_bronze_integrity - verify MinIO contents match source
#
# Exit codes:
#   0 - all stages PASS
#   1 - at least one stage FAILED
#
set -euo pipefail

# Argument parsing
DATASET="${1:-}"
RAW_PATH="${2:-}"

if [[ -z "$DATASET" ]]; then
    echo "Usage: $0 <eav|kemocon|all> [raw_data_path]"
    exit 1
fi

if [[ "$DATASET" != "eav" && "$DATASET" != "kemocon" && "$DATASET" != "all" ]]; then
    echo "Error: dataset must be one of: eav, kemocon, all (got: '$DATASET')"
    exit 1
fi

# Optionally override local path env vars from positional argument
if [[ -n "$RAW_PATH" ]]; then
    case "$DATASET" in
        eav)     export LOCAL_EAV_PATH="$RAW_PATH" ;;
        kemocon) export LOCAL_KEMOCON_PATH="$RAW_PATH" ;;
        all)
            echo "Note: raw_data_path is ignored for dataset=all." \
                 "Set LOCAL_EAV_PATH and LOCAL_KEMOCON_PATH explicitly or via .env."
            ;;
    esac
fi

# Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORTS_DIR="$SCRIPT_DIR/reports_${TIMESTAMP}"
mkdir -p "$REPORTS_DIR"

PYTHON="${PYTHON:-python3}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
log_sep() { echo "------------------------------------------------------------"; }

# Stage runner — returns exit code without killing the script
STAGE_RESULTS=()   # "STAGE_NAME:STATUS" accumulator

run_stage() {
    local stage_name="$1"
    local description="$2"
    shift 2
    local cmd=("$@")

    log_sep
    log "STAGE: $description"
    log "CMD:   ${cmd[*]}"
    log_sep

    set +e
    "${cmd[@]}"
    local exit_code=$?
    set -e

    if [[ $exit_code -eq 0 ]]; then
        log "STAGE $stage_name: PASS (exit $exit_code)"
        STAGE_RESULTS+=("$stage_name:PASS")
        return 0
    else
        log "STAGE $stage_name: FAIL (exit $exit_code)"
        STAGE_RESULTS+=("$stage_name:FAIL")
        return $exit_code
    fi
}

# Stage 1 — validate_raw
log "========================================================"
log "  Bronze Pipeline  |  dataset=$DATASET  |  $TIMESTAMP"
log "========================================================"

run_stage "validate_raw" "Validate raw local files (pre-upload)" \
    "$PYTHON" "$SCRIPT_DIR/validate_raw.py" \
        --dataset "$DATASET" \
        --config  "$SCRIPT_DIR/pipeline_config.yaml" \
        --output  "$REPORTS_DIR"

VALIDATE_RAW_STATUS="${STAGE_RESULTS[-1]##*:}"

if [[ "$VALIDATE_RAW_STATUS" == "FAIL" ]]; then
    log "FAIL-FAST: validate_raw failed — aborting pipeline."
    STAGE_RESULTS+=("upload_bronze:SKIPPED")
    STAGE_RESULTS+=("validate_bronze_integrity:SKIPPED")
else
    # Stage 2 — upload_bronze
    set +e
    "$PYTHON" "$SCRIPT_DIR/upload_bronze.py" \
        --dataset "$DATASET" \
        --output  "$REPORTS_DIR"
    UPLOAD_EXIT=$?
    set -e

    if [[ $UPLOAD_EXIT -eq 0 ]]; then
        log "STAGE upload_bronze: PASS"
        STAGE_RESULTS+=("upload_bronze:PASS")
    elif [[ $UPLOAD_EXIT -eq 1 ]]; then
        # FATAL: zero files reached MinIO
        log "STAGE upload_bronze: FAIL (FATAL — zero files in bronze)"
        STAGE_RESULTS+=("upload_bronze:FAIL")
        STAGE_RESULTS+=("validate_bronze_integrity:SKIPPED")
    elif [[ $UPLOAD_EXIT -eq 2 ]]; then
        # Partial success — still run integrity check on what made it
        log "STAGE upload_bronze: PARTIAL (some files failed)"
        STAGE_RESULTS+=("upload_bronze:PARTIAL")
    else
        log "STAGE upload_bronze: FAIL (exit $UPLOAD_EXIT)"
        STAGE_RESULTS+=("upload_bronze:FAIL")
        STAGE_RESULTS+=("validate_bronze_integrity:SKIPPED")
    fi

    UPLOAD_STATUS="${STAGE_RESULTS[-1]##*:}"

    # Run integrity check unless upload was a total failure
    if [[ "$UPLOAD_STATUS" != "FAIL" && "$UPLOAD_STATUS" != "SKIPPED" ]] || \
       [[ "${STAGE_RESULTS[-1]}" == "upload_bronze:PARTIAL" ]]; then

        # Stage 3 — validate_bronze_integrity
        run_stage "validate_bronze_integrity" "Validate bronze integrity in MinIO (post-upload)" \
            "$PYTHON" "$SCRIPT_DIR/validate_bronze_integrity.py" \
                --dataset "$DATASET" \
                --config  "$SCRIPT_DIR/pipeline_config.yaml" \
                --output  "$REPORTS_DIR"
    fi
fi

# Final summary
log_sep
log "PIPELINE SUMMARY"
log_sep

OVERALL=0
printf "  %-35s  %s\n" "Stage" "Status"
printf "  %-35s  %s\n" "-----" "------"
for entry in "${STAGE_RESULTS[@]}"; do
    stage="${entry%%:*}"
    status="${entry##*:}"
    printf "  %-35s  %s\n" "$stage" "$status"
    if [[ "$status" == "FAIL" ]]; then
        OVERALL=1
    fi
done

log_sep
log "Reports directory: $REPORTS_DIR"

if [[ $OVERALL -eq 0 ]]; then
    log "PIPELINE RESULT: PASS"
else
    log "PIPELINE RESULT: FAIL"
fi

exit $OVERALL
