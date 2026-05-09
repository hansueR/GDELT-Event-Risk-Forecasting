#!/bin/bash
#SBATCH --job-name=ASSET_PRICE
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs/%x_%j.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/hrao/GDELT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$REPO_ROOT"

mkdir -p "${PROJECT_ROOT}/asset_prices" "${LOG_DIR:-${PROJECT_ROOT}/logs}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/download_asset_prices.py" \
  --start 2016-03-01 \
  --end 2026-03-31 \
  --output_file "${PROJECT_ROOT}/asset_prices/daily_asset_prices.parquet" \
  --tickers "QQQ=QQQ,Gold=GLD,WTI_Oil=USO"


# sbatch src/data_pipeline/download_asset_prices.sh
