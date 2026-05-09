#!/bin/bash
#SBATCH --job-name=MODEL_TABLE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs/%x_%j.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/hrao/GDELT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

mkdir -p "${PROJECT_ROOT}/modeling" "${LOG_DIR:-${PROJECT_ROOT}/logs}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_model_table.py" \
  --price_file "${PROJECT_ROOT}/asset_prices/daily_asset_prices.parquet" \
  --event_long_dir "${PROJECT_ROOT}/features/event_long_clean_full" \
  --output_file "${PROJECT_ROOT}/modeling/model_table_clean_full.parquet" \
  --top_root_k 20 \
  --top_country_k 20

# sbatch src/data_pipeline/build_model_table.sh
