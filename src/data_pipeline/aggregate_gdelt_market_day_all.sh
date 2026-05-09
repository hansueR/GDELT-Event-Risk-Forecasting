#!/bin/bash
#SBATCH --job-name=GDELT_AGG
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --partition=bigTiger
#SBATCH --nodelist=itiger04
#SBATCH --output=/project/hrao/GDELT/logs/%x_%j.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project/hrao/GDELT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CLEAN_DIR="${CLEAN_DIR:-${PROJECT_ROOT}/clean_full}"
FEATURE_DIR="${FEATURE_DIR:-${PROJECT_ROOT}/features/event_long_clean_full}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
MISSING_FILE="$LOG_DIR/missing_event_clean_full_months.txt"

mkdir -p "$FEATURE_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

: > "$MISSING_FILE"

shopt -s nullglob

for monthdir in "$CLEAN_DIR"/events_*; do
    [[ -d "$monthdir" ]] || continue

    month=$(basename "$monthdir" | sed 's/events_//')
    outfile="$FEATURE_DIR/events_market_day_long_${month}.parquet"
    doneflag="$LOG_DIR/agg_clean_full_${month}.done"

    if [[ -f "$doneflag" && -f "$outfile" ]]; then
        echo "[$(date)] skip $month"
        continue
    fi

    parquet_count=$(find "$monthdir" -maxdepth 1 -type f -name "*.parquet" | wc -l)

    if [[ "$parquet_count" -eq 0 ]]; then
        echo "[$(date)] WARNING: skip $month because no parquet files found in $monthdir"
        echo "$month $monthdir" >> "$MISSING_FILE"
        continue
    fi

    echo "[$(date)] aggregate $month"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/aggregate_market_day.py" \
      --input_dir "$monthdir" \
      --output_file "$outfile" \
      --close_hour 16

    touch "$doneflag"
    echo "[$(date)] done $month"
done

echo "[$(date)] all monthly event aggregation done."
