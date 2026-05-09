#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
PROJECT_ROOT="${PROJECT_ROOT:-/project/hrao/GDELT}"
RESULTS="${RESULTS:-${PROJECT_ROOT}/results}"
INPUT="${INPUT:-${PROJECT_ROOT}/modeling/model_table_ddfe_root_country_clean_full.parquet}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

"${PYTHON_BIN}" "${SRC_DIR}/experiments/03_grouped_svd_events.py" \
  --input "${INPUT}" \
  --output_dir "${RESULTS}/smoke_grouped_svd" \
  --assets Gold \
  --targets target_rv_return_3d \
  --test_year_start 2020 \
  --test_year_end 2020 \
  --model_type logistic \
  --event_prefix ddfe_

"${PYTHON_BIN}" "${SRC_DIR}/experiments/05_residual_selected_grouped_pls.py" \
  --input "${INPUT}" \
  --output_dir "${RESULTS}/smoke_residual_pls" \
  --assets Gold \
  --targets target_rv_return_3d \
  --test_year_start 2020 \
  --test_year_end 2020 \
  --model_type logistic \
  --event_prefix ddfe_
