#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="${PROJECT_ROOT}"
elif [[ -d "${REPO_ROOT}/../features" || -d "${REPO_ROOT}/../modeling" || -d "${REPO_ROOT}/../asset_prices" ]]; then
  PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
else
  PROJECT_ROOT="${REPO_ROOT}"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

START_DATE="${START_DATE:-2016-03-01}"
END_DATE="${END_DATE:-$(date -u -d '+1 day' +%F)}"
TICKERS="${TICKERS:-QQQ=QQQ,Gold=GLD,WTI_Oil=USO}"

ASSET_PRICE_FILE="${ASSET_PRICE_FILE:-${PROJECT_ROOT}/asset_prices/daily_asset_prices.parquet}"
EVENT_LONG_DIR="${EVENT_LONG_DIR:-${PROJECT_ROOT}/features/event_long_clean_full}"
BASE_MODEL_TABLE="${BASE_MODEL_TABLE:-${PROJECT_ROOT}/modeling/model_table_clean_full.parquet}"
ONLINE_MODEL_TABLE="${ONLINE_MODEL_TABLE:-${PROJECT_ROOT}/modeling/model_table_ddfe_root_country_clean_full.parquet}"
DDFE_OUTPUT_DIR="${DDFE_OUTPUT_DIR:-${PROJECT_ROOT}/results/ddfe_root_country_clean_full_v1}"
ONLINE_EVENT_FEATURES_FILE="${ONLINE_EVENT_FEATURES_FILE:-${PROJECT_ROOT}/data/online/daily_event_features.parquet}"

TOP_ROOT_K="${TOP_ROOT_K:-20}"
TOP_COUNTRY_K="${TOP_COUNTRY_K:-20}"
MIN_NONZERO_DAYS="${MIN_NONZERO_DAYS:-30}"
TOP_PAIRS="${TOP_PAIRS:-400}"
WINDOWS="${WINDOWS:-20 60}"
Z_THRESHOLD="${Z_THRESHOLD:-2.0}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ ! -d "${EVENT_LONG_DIR}" ]]; then
  echo "Missing EVENT_LONG_DIR: ${EVENT_LONG_DIR}"
  echo "Provide aggregated event_long parquet files before building the online model table."
  exit 1
fi

mkdir -p "$(dirname "${ASSET_PRICE_FILE}")" "$(dirname "${BASE_MODEL_TABLE}")" "$(dirname "${ONLINE_MODEL_TABLE}")" "${DDFE_OUTPUT_DIR}" "$(dirname "${ONLINE_EVENT_FEATURES_FILE}")"

echo "Step 1/4: Download asset prices through ${END_DATE}"
"${PYTHON_BIN}" src/data_pipeline/download_asset_prices.py \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --output_file "${ASSET_PRICE_FILE}" \
  --tickers "${TICKERS}"

echo "Step 2/4: Build base model table"
"${PYTHON_BIN}" src/data_pipeline/build_model_table.py \
  --price_file "${ASSET_PRICE_FILE}" \
  --event_long_dir "${EVENT_LONG_DIR}" \
  --output_file "${BASE_MODEL_TABLE}" \
  --top_root_k "${TOP_ROOT_K}" \
  --top_country_k "${TOP_COUNTRY_K}"

echo "Step 3/4: Build data-driven DDFe model table"
"${PYTHON_BIN}" src/data_pipeline/build_data_driven_root_country_features.py \
  --event_long_dir "${EVENT_LONG_DIR}" \
  --model_table "${BASE_MODEL_TABLE}" \
  --output_file "${ONLINE_MODEL_TABLE}" \
  --output_dir "${DDFE_OUTPUT_DIR}" \
  --min_nonzero_days "${MIN_NONZERO_DAYS}" \
  --top_pairs "${TOP_PAIRS}" \
  --windows ${WINDOWS} \
  --z_threshold "${Z_THRESHOLD}"

echo "Step 4/4: Export daily event features for online inference"
"${PYTHON_BIN}" src/online/export_daily_event_features.py \
  --model_table "${ONLINE_MODEL_TABLE}" \
  --output_file "${ONLINE_EVENT_FEATURES_FILE}"

echo "Online model table ready: ${ONLINE_MODEL_TABLE}"
echo "Online event features ready: ${ONLINE_EVENT_FEATURES_FILE}"
