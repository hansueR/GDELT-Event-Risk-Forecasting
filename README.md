# GDELT Event-Driven Volatility Risk Forecasting

## Overview
This project studies whether structured GDELT event data can improve short-horizon volatility-risk forecasting for macro assets. The main assets are Gold, QQQ, and WTI Oil. The forecasting task focuses on 3-day and 5-day future realized volatility and unexpected-volatility alerting.

## Data
- GDELT 2.0 Events are used to construct daily event features.
- Price data are collected from Yahoo Finance and include OHLCV fields.
- Raw data and generated model tables are not included in the repository because they are large.

## Methods
The compared methods are:
- PO: price-only baseline
- RAW: raw event concatenation
- SVD: grouped SVD event compression
- R-SVD: residual-selected grouped SVD
- R-PLS: residual-selected grouped PLS
- RFI: residual fusion with interaction features

The main evaluation uses yearly walk-forward testing.

## Repository Structure
```text
src/
  gdelt_risk/       Core reusable Python modules
  experiments/      Experiment entry scripts
  analysis/         Result summarization and plotting scripts
  data_pipeline/    Data download, cleaning, aggregation, and feature construction scripts
  pipelines/        End-to-end shell pipelines
```

## How to Run
Run one experiment:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
python src/experiments/01_price_only_baseline.py \
  --input /path/to/model_table_ddfe_root_country_clean_full.parquet \
  --output_dir /path/to/results/ddfe_price_only_v1
```

Run the full event-fusion pipeline:

```bash
bash src/pipelines/run_event_fusion_pipeline.sh
```

Run with Slurm:

```bash
sbatch src/pipelines/run_event_fusion_pipeline.sh
```

`INPUT`, `RESULTS`, `PROJECT_ROOT`, and `PYTHON_BIN` can be overridden as environment variables.

## Outputs
Each experiment writes:
- `fold_predictions.csv`
- `fold_metrics.csv`
- `summary_metrics.csv`
- optional selected event feature files

The analysis pipeline writes:
- `all_methods_summary.csv`
- `delta_vs_price_only.csv`
- `best_by_asset_target.csv`
- `summary_report.txt`
- figures

## Notes
Large data files, intermediate parquet files, logs, and results are intentionally excluded from GitHub.
