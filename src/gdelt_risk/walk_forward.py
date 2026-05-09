import numpy as np
import pandas as pd


def prepare_asset_target_frame(df, asset, target, feature_cols=None):
    data = df[df["asset"] == asset].copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)
    data["year"] = data["date"].dt.year
    cols = ["date", "asset", "year", target]
    if feature_cols:
        cols += [c for c in feature_cols if c not in cols]
    data = data[cols].replace([np.inf, -np.inf], np.nan)
    return data.dropna(subset=[target]).reset_index(drop=True)


def iter_year_folds(data, test_year_start=None, test_year_end=None, min_train_rows=500, min_test_rows=30):
    years = sorted(int(y) for y in data["year"].dropna().unique())
    for year in years:
        if test_year_start is not None and year < test_year_start:
            continue
        if test_year_end is not None and year > test_year_end:
            continue
        train_idx = data.index[data["year"] < year].to_numpy()
        test_idx = data.index[data["year"] == year].to_numpy()
        if len(train_idx) < min_train_rows or len(test_idx) < min_test_rows:
            continue
        yield year, train_idx, test_idx


def make_high_vol_labels(train_target, test_target, quantile=0.80):
    threshold = float(np.nanquantile(train_target, quantile))
    return (train_target >= threshold).astype(int), (test_target >= threshold).astype(int), threshold
