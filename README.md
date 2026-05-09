# GDELT Event-Driven Volatility Risk Forecasting

**Public Dashboard:** https://hansuer.github.io/GDELT-Event-Risk-Forecasting/

The URL is served by GitHub Pages after the `Deploy dashboard` or `Daily online forecast` GitHub Actions workflow completes successfully.

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

## Dashboard
This repository currently publishes a static historical dashboard under `dashboard/`. It is a static GitHub Pages site: the browser loads only HTML, CSS, JavaScript, and committed JSON files from `dashboard/data/`. No Python backend is required at runtime.

The dashboard visualizes:
- close price and event intensity over time
- future realized volatility and a selected model forecast or alert score
- alert metrics by method
- summary metrics by method

Raw GDELT files, large parquet model tables, and large intermediate experiment outputs are not committed. To regenerate the small dashboard JSON files locally from available artifacts, run:

```bash
python src/dashboard/build_dashboard_data.py \
  --model_table /path/to/model_table_ddfe_root_country_clean_full.parquet \
  --results_dir /path/to/results \
  --output_dir dashboard/data
```

The builder does not generate simulated data and does not truncate the historical series by default. The committed static dashboard JSON is generated from the full available real model table and real experiment metrics. If the source files are unavailable, the builder writes empty JSON payloads with warnings instead of placeholder metrics or fake time series.

Preview locally for development:

```bash
python -m http.server 8000 -d dashboard
```

Then open `http://localhost:8000`.

To publish through GitHub Pages, enable Pages in the repository settings and choose GitHub Actions as the source. The manual `.github/workflows/deploy_dashboard.yml` workflow deploys the committed `dashboard/` directory after validation. Run it with `mode=static` to publish the historical dashboard JSON currently committed under `dashboard/data/`.

## TODO: Daily Online Dashboard
The daily online live dashboard is planned but not the current deployed mode. The target design is still static at serving time: a scheduled workflow will train models, run daily inference, write JSON files into `dashboard/data/`, validate them, and deploy GitHub Pages without a Python backend.

Online model training is separate from walk-forward evaluation. Walk-forward runs are used to estimate historical performance. Final online models are trained on all available historical rows before deployment.

To rebuild the latest online model table from aggregated event history and prices:

```bash
PROJECT_ROOT=$PWD \
EVENT_LONG_DIR=features/event_long_clean_full \
ONLINE_MODEL_TABLE=modeling/model_table_ddfe_root_country_clean_full.parquet \
ONLINE_EVENT_FEATURES_FILE=data/online/daily_event_features.parquet \
bash src/pipelines/build_online_model_table.sh
```

The script downloads current asset prices, builds the base model table, builds DDFe root-country features, and exports date-level daily event features for online inference.

On this project layout, local real data can also live outside the repository under `GDELT_DATA_ROOT` such as `/project/hrao/GDELT`. The online scripts auto-detect that parent data root when it contains `modeling/`, `features/`, or `asset_prices/`.

To update local GDELT event data from the public GDELT 2.0 Events files:

```bash
GDELT_DATA_ROOT=/project/hrao/GDELT \
START_DATE=2026-03-31 \
END_DATE=2026-05-10 \
PYTHON_BIN=/path/to/python \
bash src/pipelines/update_gdelt_events_local.sh
```

The update script generates URLs like `http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip`, downloads the 15-minute event zip files, cleans them, and rebuilds `features/event_long_clean_full/events_market_day_long_YYYYMM.parquet`.

To run the full local dashboard pipeline from real data after event history is current:

```bash
GDELT_DATA_ROOT=/project/hrao/GDELT \
PYTHON_BIN=/path/to/python \
bash src/pipelines/run_online_dashboard_local.sh
```

Train the final online models:

```bash
python src/online/train_online_models.py \
  --model_table modeling/model_table_ddfe_root_country_clean_full.parquet \
  --output_dir artifacts/online
```

The online forecasts are generated from the full trained feature schema saved in `artifacts/online/feature_schema.json`. Daily inference fails if required training price or event columns are missing, extra unseen event columns are ignored, and column order is matched to the saved schema. The aggregate event-intensity curve shown in the dashboard is only a visualization signal; it is not used as a replacement model input.
If the trained schema contains event features, daily inference requires a real `--event_features_file`; it will not replace missing event data with simulated aggregate values.

Daily inference can then be run with:

```bash
python src/online/update_online_data.py \
  --output_dir data/online \
  --feature_schema artifacts/online/feature_schema.json \
  --event_features_file /path/to/daily_event_features.parquet

python src/online/run_daily_inference.py \
  --features data/online/latest_features.parquet \
  --artifact_dir artifacts/online \
  --output_dir dashboard/data

python src/online/backfill_actuals.py \
  --history dashboard/data/prediction_history.json \
  --features data/online/latest_features.parquet
```

Latest 3d and 5d predictions are marked with `actual_log_rv: null` and `status: "pending"` until enough future trading days are available to compute the realized volatility target. `backfill_actuals.py` resolves older predictions when the future window is available.

Before deployment, validate the generated JSON:

```bash
python src/dashboard/validate_dashboard_data.py \
  --data_dir dashboard/data \
  --mode online
```

`.github/workflows/daily_forecast.yml` runs daily online retraining from refreshed historical data, daily feature update, inference, actual backfill, validation, and GitHub Pages deployment. It can consume real inputs in two ways:
- `ONLINE_DATA_BUNDLE_URL`: a zip/tar bundle containing paths such as `features/event_long_clean_full/`, `modeling/`, or `results/`
- `ONLINE_MODEL_TABLE_URL`: a direct URL to a prebuilt `model_table_ddfe_root_country_clean_full.parquet`

If `features/event_long_clean_full/` is available, the workflow rebuilds the model table daily before training. Otherwise, it uses the prebuilt model table URL and exports daily event features from that table. Set `PREDICTION_HISTORY_URL` to the deployed `dashboard/data/prediction_history.json` URL if you want the workflow to merge today's predictions with previously deployed history. If required real inputs are missing, the workflow fails instead of publishing placeholder forecasts. `.github/workflows/train_online_models.yml` is still available for manual artifact builds and debugging.

Raw GDELT files, large historical model tables, and large model artifacts are not committed to GitHub by default. Store or provide those artifacts through your deployment process before expecting live non-placeholder forecasts.

## Notes
Large data files, intermediate parquet files, logs, and results are intentionally excluded from GitHub.
