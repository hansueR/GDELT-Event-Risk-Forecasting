#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${GDELT_DATA_ROOT:-}" ]]; then
  PROJECT_ROOT="${GDELT_DATA_ROOT}"
elif [[ -d "${REPO_ROOT}/../features" || -d "${REPO_ROOT}/../raw_zip" ]]; then
  PROJECT_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
else
  PROJECT_ROOT="${REPO_ROOT}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
START_DATE="${START_DATE:-$(date -u -d '7 days ago' +%F)}"
END_DATE="${END_DATE:-$(date -u -d '+1 day' +%F)}"
PARALLEL="${PARALLEL:-8}"
COLUMN_MODE="${COLUMN_MODE:-full}"

MANIFEST_DIR="${MANIFEST_DIR:-${PROJECT_ROOT}/manifests/online_update/by_month}"
RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/raw_zip}"
CLEAN_ROOT="${CLEAN_ROOT:-${PROJECT_ROOT}/clean_full}"
EVENT_LONG_DIR="${EVENT_LONG_DIR:-${PROJECT_ROOT}/features/event_long_clean_full}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

mkdir -p "${MANIFEST_DIR}" "${RAW_ROOT}" "${CLEAN_ROOT}" "${EVENT_LONG_DIR}"

echo "Step 1/4: Generate GDELT manifest from ${START_DATE} to ${END_DATE}"
"${PYTHON_BIN}" src/data_pipeline/make_gdelt_manifest.py \
  --start "${START_DATE}" \
  --end "${END_DATE}" \
  --output_dir "${MANIFEST_DIR}"

echo "Step 2/4: Download GDELT export zip files"
shopt -s nullglob
for manifest in "${MANIFEST_DIR}"/*.txt; do
  month="$(basename "${manifest}" .txt)"
  outdir="${RAW_ROOT}/${month}"
  mkdir -p "${outdir}"
  echo "Downloading ${month} -> ${outdir}"
  xargs -a "${manifest}" -n 1 -P "${PARALLEL}" wget -c -nv -P "${outdir}"
done

echo "Step 3/4: Clean monthly GDELT zip files"
for manifest in "${MANIFEST_DIR}"/*.txt; do
  month="$(basename "${manifest}" .txt)"
  raw_dir="${RAW_ROOT}/${month}"
  clean_dir="${CLEAN_ROOT}/events_${month}"
  echo "Cleaning ${month}"
  rm -rf "${clean_dir}"
  mkdir -p "${clean_dir}"
  "${PYTHON_BIN}" src/data_pipeline/clean_gdelt_month_full.py \
    --input_dir "${raw_dir}" \
    --output_dir "${clean_dir}" \
    --column_mode "${COLUMN_MODE}"
done

echo "Step 4/4: Aggregate market-day event features"
for manifest in "${MANIFEST_DIR}"/*.txt; do
  month="$(basename "${manifest}" .txt)"
  clean_dir="${CLEAN_ROOT}/events_${month}"
  outfile="${EVENT_LONG_DIR}/events_market_day_long_${month}.parquet"
  echo "Aggregating ${month}"
  "${PYTHON_BIN}" src/data_pipeline/aggregate_market_day.py \
    --input_dir "${clean_dir}" \
    --output_file "${outfile}" \
    --close_hour 16
done

echo "Updated event_long features in ${EVENT_LONG_DIR}"
