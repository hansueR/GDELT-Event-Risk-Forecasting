# Active Pipeline Scripts

The active event table for the next project direction is:

`/project/hrao/GDELT/modeling/model_table_ddfe_root_country_clean_full.parquet`

Active event features use `ddfe_` column prefixes. The old asset-theme shock table and `asfe_` columns are deprecated and retained only as optional reference under `deprecated/old_asset_theme_fdr_pipeline/`.

## Data Construction Path

1. Download GDELT zip files with `src/data_pipeline/download_gdelt_10y.sh`.
2. Clean monthly raw files with `src/data_pipeline/clean_gdelt_full_array.sh`.
3. Build `features/event_long_clean_full` with `src/data_pipeline/aggregate_gdelt_market_day_all.sh`.
4. Download prices with `src/data_pipeline/download_asset_prices.py` or `src/data_pipeline/download_asset_prices.sh`.
5. Build the base price/event model table with `src/data_pipeline/build_model_table.py` or `src/data_pipeline/build_model_table.sh`.
6. Build the active ddfe root-country table with `src/data_pipeline/build_data_driven_root_country_features.py` or `src/data_pipeline/build_data_driven_root_country_features.sh`.
7. Profile the active table with `src/data_pipeline/profile_event_data.py --event_prefixes ddfe_`.

`src/data_pipeline/walk_forward_baseline.py` is kept as price-only / walk-forward reference code. It is not the final event-modeling experiment path.
