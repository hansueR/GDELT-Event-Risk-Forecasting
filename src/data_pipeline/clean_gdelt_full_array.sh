#!/bin/bash
#SBATCH --job-name=GDELT_CLEAN_FULL_ARRAY
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --array=1-121%6
#SBATCH --output=/project/hrao/GDELT/logs/clean_full_array/%x_%A_%a.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/hrao/GDELT}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MONTH_LIST="${MONTH_LIST:-${PROJECT_ROOT}/manifests/months_clean_full.txt}"

MONTH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MONTH_LIST")

if [[ -z "$MONTH" ]]; then
    echo "No month found for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi

RAW_DIR="${RAW_DIR:-${PROJECT_ROOT}/raw_zip/$MONTH}"
CLEAN_DIR="${CLEAN_DIR:-${PROJECT_ROOT}/clean_full}"
OUT_DIR="$CLEAN_DIR/events_$MONTH"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/clean_full_array}"
SCRIPT="${SCRIPT_DIR}/clean_gdelt_month_full.py"

COLUMN_MODE="${COLUMN_MODE:-full}"

mkdir -p "$OUT_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "MONTH=$MONTH"
echo "RAW_DIR=$RAW_DIR"
echo "OUT_DIR=$OUT_DIR"
echo "COLUMN_MODE=$COLUMN_MODE"

if [[ ! -d "$RAW_DIR" ]]; then
    echo "Missing RAW_DIR: $RAW_DIR"
    exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

"${PYTHON_BIN}" "$SCRIPT" \
  --input_dir "$RAW_DIR" \
  --output_dir "$OUT_DIR" \
  --column_mode "$COLUMN_MODE"

echo "done month=$MONTH"
