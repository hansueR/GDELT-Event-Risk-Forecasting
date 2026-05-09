#!/bin/bash
#SBATCH --job-name=DDFE_BUILD
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs/%x_%j.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/hrao/GDELT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"
mkdir -p "${PROJECT_ROOT}/modeling" "${PROJECT_ROOT}/results/ddfe_root_country_clean_full_v1" "${LOG_DIR:-${PROJECT_ROOT}/logs}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_data_driven_root_country_features.py" \
  --event_long_dir "${PROJECT_ROOT}/features/event_long_clean_full" \
  --model_table "${PROJECT_ROOT}/modeling/model_table_clean_full.parquet" \
  --output_file "${PROJECT_ROOT}/modeling/model_table_ddfe_root_country_clean_full.parquet" \
  --output_dir "${PROJECT_ROOT}/results/ddfe_root_country_clean_full_v1" \
  --min_nonzero_days 30 \
  --top_pairs 400 \
  --windows 20 60 \
  --z_threshold 2.0
