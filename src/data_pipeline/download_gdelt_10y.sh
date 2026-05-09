#!/bin/bash
#SBATCH --job-name=GDELT
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
MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/manifests/by_month_201603_202603}"
RAW_DIR="${RAW_DIR:-${PROJECT_ROOT}/raw_zip}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
PARALLEL="${PARALLEL:-16}"

mkdir -p "$RAW_DIR" "$LOG_DIR"
cd "$REPO_ROOT"

shopt -s nullglob

for mf in "$MANIFEST_DIR"/*.txt; do
    month=$(basename "$mf" .txt)
    outdir="$RAW_DIR/$month"
    doneflag="$LOG_DIR/download_${month}.done"
    logfile="$LOG_DIR/download_${month}.log"

    mkdir -p "$outdir"

    if [[ -f "$doneflag" ]]; then
        echo "[$(date)] skip $month (already done)"
        continue
    fi

    echo "[$(date)] start $month" | tee -a "$logfile"
    echo "manifest: $mf" | tee -a "$logfile"
    echo "outdir:   $outdir" | tee -a "$logfile"

    # cat "$mf" | xargs -n 1 -P "$PARALLEL" wget -c -nv -P "$outdir" >> "$logfile" 2>&1
    set +e
    xargs -a "$mf" -n 1 -P "$PARALLEL" wget -c -nv -P "$outdir" >> "$logfile" 2>&1
    dl_status=$?
    set -e
    
    expected=$(wc -l < "$mf")
    got=$(find "$outdir" -maxdepth 1 -type f -name '*.export.CSV.zip' | wc -l)

    echo "expected=$expected got=$got" | tee -a "$logfile"

    if [[ "$expected" -eq "$got" ]]; then
        touch "$doneflag"
        echo "[$(date)] done $month" | tee -a "$logfile"
    else
        echo "[$(date)] WARNING: file count mismatch for $month" | tee -a "$logfile"
    fi
done

echo "[$(date)] all available months processed."

# sbatch src/data_pipeline/download_gdelt_10y.sh
