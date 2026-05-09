#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${GDELT_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${GDELT_DATA_ROOT}"
elif [[ -d "${REPO_ROOT}/../modeling" || -d "${REPO_ROOT}/../features" ]]; then
  DATA_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
else
  DATA_ROOT="${REPO_ROOT}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_TABLE="${MODEL_TABLE:-${DATA_ROOT}/modeling/model_table_ddfe_root_country_clean_full.parquet}"
RESULTS_DIR="${RESULTS_DIR:-${DATA_ROOT}/results}"
EVENT_FEATURES_FILE="${EVENT_FEATURES_FILE:-${DATA_ROOT}/data/online/daily_event_features.parquet}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${DATA_ROOT}/artifacts/online}"
LATEST_FEATURE_DIR="${LATEST_FEATURE_DIR:-${DATA_ROOT}/data/online}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dashboard/data}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ ! -f "${MODEL_TABLE}" ]]; then
  echo "Missing real model table: ${MODEL_TABLE}"
  echo "Set GDELT_DATA_ROOT or MODEL_TABLE to a directory/file produced by src/data_pipeline."
  exit 1
fi

echo "Using data root: ${DATA_ROOT}"
echo "Using model table: ${MODEL_TABLE}"

echo "Step 1/6: Export real daily event features"
"${PYTHON_BIN}" src/online/export_daily_event_features.py \
  --model_table "${MODEL_TABLE}" \
  --output_file "${EVENT_FEATURES_FILE}"

echo "Step 2/6: Train online models on real historical rows"
"${PYTHON_BIN}" src/online/train_online_models.py \
  --model_table "${MODEL_TABLE}" \
  --output_dir "${ARTIFACT_DIR}"

echo "Step 3/6: Build dashboard evaluation JSON from real data"
"${PYTHON_BIN}" src/dashboard/build_dashboard_data.py \
  --model_table "${MODEL_TABLE}" \
  --results_dir "${RESULTS_DIR}" \
  --output_dir "${OUTPUT_DIR}"

echo "Step 4/6: Build latest online feature rows"
"${PYTHON_BIN}" src/online/update_online_data.py \
  --output_dir "${LATEST_FEATURE_DIR}" \
  --feature_schema "${ARTIFACT_DIR}/feature_schema.json" \
  --event_features_file "${EVENT_FEATURES_FILE}"

echo "Step 5/6: Run online inference"
"${PYTHON_BIN}" src/online/run_daily_inference.py \
  --features "${LATEST_FEATURE_DIR}/latest_features.parquet" \
  --artifact_dir "${ARTIFACT_DIR}" \
  --output_dir "${OUTPUT_DIR}"

echo "Step 6/6: Validate online dashboard JSON"
"${PYTHON_BIN}" src/dashboard/validate_dashboard_data.py \
  --data_dir "${OUTPUT_DIR}" \
  --mode online

echo "Dashboard data is ready in ${OUTPUT_DIR}"
