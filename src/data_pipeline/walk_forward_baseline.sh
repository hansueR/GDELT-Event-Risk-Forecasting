#!/bin/bash
#SBATCH --job-name=WF_BASELINE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs/%x_%j.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/hrao/GDELT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

mkdir -p "${PROJECT_ROOT}/results/baseline_v1" "${LOG_DIR:-${PROJECT_ROOT}/logs}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/walk_forward_baseline.py" \
  --model_table "${PROJECT_ROOT}/modeling/model_table.parquet" \
  --output_dir "${PROJECT_ROOT}/results/baseline_v1" \
  --model_type ridge

"${PYTHON_BIN}" "${SCRIPT_DIR}/walk_forward_baseline.py" \
  --model_table "${PROJECT_ROOT}/modeling/model_table.parquet" \
  --output_dir "${PROJECT_ROOT}/results/baseline_v1" \
  --model_type random_forest
