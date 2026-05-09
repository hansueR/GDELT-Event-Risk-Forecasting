#!/bin/bash
#SBATCH --job-name=DDFe_Fusion
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/hrao/GDELT/logs/DDFe_Fusion_%j.out
#SBATCH --error=/project/hrao/GDELT/logs/DDFe_Fusion_%j.err

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-NA}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
PROJECT_ROOT="${PROJECT_ROOT:-/project/hrao/GDELT}"
RESULTS="${RESULTS:-${PROJECT_ROOT}/results}"
INPUT="${INPUT:-${PROJECT_ROOT}/modeling/model_table_ddfe_root_country_clean_full.parquet}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Using Python: ${PYTHON_BIN}"
echo "Input file: ${INPUT}"
echo "Results root: ${RESULTS}"

echo "Step 1/8: price-only baseline"
"${PYTHON_BIN}" "${SRC_DIR}/experiments/01_price_only_baseline.py" \
  --input "${INPUT}" \
  --output_dir "${RESULTS}/ddfe_price_only_v1"

echo "Step 2/8: raw DDFe concat"
"${PYTHON_BIN}" "${SRC_DIR}/experiments/02_raw_event_concat.py" \
  --input "${INPUT}" \
  --output_dir "${RESULTS}/ddfe_raw_concat_v1"

echo "Step 3/8: grouped SVD DDFe"
"${PYTHON_BIN}" "${SRC_DIR}/experiments/03_grouped_svd_events.py" \
  --input "${INPUT}" \
  --output_dir "${RESULTS}/ddfe_grouped_svd_v1"

echo "Step 4/8: residual-selected grouped SVD"
"${PYTHON_BIN}" "${SRC_DIR}/experiments/04_residual_selected_grouped_svd.py" \
  --input "${INPUT}" \
  --output_dir "${RESULTS}/ddfe_residual_grouped_svd_v1"

echo "Step 5/8: residual-selected grouped PLS"
"${PYTHON_BIN}" "${SRC_DIR}/experiments/05_residual_selected_grouped_pls.py" \
  --input "${INPUT}" \
  --output_dir "${RESULTS}/ddfe_residual_grouped_pls_v1"

echo "Step 6/8: residual fusion with regime interactions"
"${PYTHON_BIN}" "${SRC_DIR}/experiments/06_residual_fusion_interactions.py" \
  --input "${INPUT}" \
  --output_dir "${RESULTS}/ddfe_residual_fusion_interactions_v1"

echo "Step 7/8: summarize results"
"${PYTHON_BIN}" "${SRC_DIR}/analysis/summarize_event_fusion_results.py" \
  --results_root "${RESULTS}" \
  --output_dir "${RESULTS}/ddfe_event_fusion_summary_v1"

echo "Step 8/8: make figures"
"${PYTHON_BIN}" "${SRC_DIR}/analysis/make_event_fusion_figures.py" \
  --summary_dir "${RESULTS}/ddfe_event_fusion_summary_v1" \
  --output_dir "${RESULTS}/ddfe_event_fusion_summary_v1/figures"

echo "Job finished at: $(date)"
echo "Summary output:"
ls -lh "${RESULTS}/ddfe_event_fusion_summary_v1" || true
ls -lh "${RESULTS}/ddfe_event_fusion_summary_v1/figures" || true


# sbatch src/pipelines/run_event_fusion_pipeline.sh
