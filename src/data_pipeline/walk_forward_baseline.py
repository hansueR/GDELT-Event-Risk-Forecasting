from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PRICE_FEATURES = [
    "log_return_1d",
    "abs_log_return_1d",
    "squared_log_return_1d",
    "parkinson_vol_1d",
    "hist_rv_return_3d",
    "hist_abs_return_mean_3d",
    "hist_parkinson_vol_3d",
    "hist_rv_return_5d",
    "hist_abs_return_mean_5d",
    "hist_parkinson_vol_5d",
    "hist_rv_return_10d",
    "hist_abs_return_mean_10d",
    "hist_parkinson_vol_10d",
    "hist_rv_return_20d",
    "hist_abs_return_mean_20d",
    "hist_parkinson_vol_20d",
]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def directional_accuracy(y_true, y_pred, baseline_level):
    """
    Directional accuracy for whether volatility is above a baseline level.
    baseline_level is usually the train-set mean target.
    """
    true_up = y_true > baseline_level
    pred_up = y_pred > baseline_level
    return np.mean(true_up == pred_up)


def get_event_features(df):
    event_cols = [c for c in df.columns if c.startswith("event_")]

    # Drop rolling sums of mean-like variables because the current build script
    # sums them over windows. Counts and sums are fine, but means should not be summed.
    bad_roll_mean_cols = [
        c for c in event_cols
        if (
            ("tone_weighted_mean_roll" in c)
            or ("goldstein_weighted_mean_roll" in c)
        )
    ]

    event_cols = [c for c in event_cols if c not in bad_roll_mean_cols]
    return event_cols


def make_model(model_type):
    if model_type == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_one_setting(df, asset, target_col, feature_cols, model_type):
    data = df[df["asset"] == asset].copy()
    data = data.sort_values("date").reset_index(drop=True)

    needed_cols = ["date", target_col] + feature_cols
    data = data[needed_cols].replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 500:
        raise ValueError(f"Too few rows for {asset} {target_col}: {len(data)}")

    # Walk-forward by year.
    # Train on all data before test year, test on that year.
    data["year"] = pd.to_datetime(data["date"]).dt.year

    results = []
    all_preds = []

    test_years = sorted(data["year"].unique())
    test_years = [y for y in test_years if y >= 2019]

    for test_year in test_years:
        train = data[data["year"] < test_year].copy()
        test = data[data["year"] == test_year].copy()

        if len(train) < 500 or len(test) < 30:
            continue

        X_train = train[feature_cols]
        y_train = train[target_col].values

        X_test = test[feature_cols]
        y_test = test[target_col].values

        model = make_model(model_type)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred = np.maximum(pred, 0.0)

        train_mean = float(np.mean(y_train))

        row = {
            "asset": asset,
            "target": target_col,
            "model_type": model_type,
            "test_year": int(test_year),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(rmse(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "directional_acc": float(directional_accuracy(y_test, pred, train_mean)),
            "train_target_mean": train_mean,
        }
        results.append(row)

        pred_df = pd.DataFrame({
            "date": test["date"].values,
            "asset": asset,
            "target": target_col,
            "model_type": model_type,
            "test_year": test_year,
            "y_true": y_test,
            "y_pred": pred,
        })
        all_preds.append(pred_df)

    result_df = pd.DataFrame(results)

    if all_preds:
        pred_df = pd.concat(all_preds, ignore_index=True)
    else:
        pred_df = pd.DataFrame()

    return result_df, pred_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_table",
        default="/project/hrao/GDELT/modeling/model_table.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/baseline_v1",
    )
    parser.add_argument(
        "--model_type",
        default="ridge",
        choices=["ridge", "random_forest"],
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.model_table)
    df["date"] = pd.to_datetime(df["date"])

    event_features = get_event_features(df)

    # Log-transform nonnegative event magnitude features.
    # This reduces the effect of very large event counts.
    for col in event_features:
        if df[col].min(skipna=True) >= 0:
            df[col] = np.log1p(df[col])

    target_cols = [
        "target_rv_return_1d",
        "target_rv_return_3d",
        "target_rv_return_5d",
    ]

    assets = sorted(df["asset"].unique())

    experiment_configs = {
        "price_only": PRICE_FEATURES,
        "price_plus_event": PRICE_FEATURES + event_features,
    }

    all_results = []
    all_preds = []

    for feature_set_name, feature_cols in experiment_configs.items():
        print(f"\n=== Feature set: {feature_set_name} ===")
        print("n_features:", len(feature_cols))

        for asset in assets:
            for target_col in target_cols:
                print(f"Running {asset} | {target_col} | {args.model_type}")

                result_df, pred_df = evaluate_one_setting(
                    df=df,
                    asset=asset,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    model_type=args.model_type,
                )

                result_df["feature_set"] = feature_set_name
                pred_df["feature_set"] = feature_set_name

                all_results.append(result_df)
                all_preds.append(pred_df)

    results = pd.concat(all_results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    results_file = out_dir / f"walk_forward_results_{args.model_type}.csv"
    preds_file = out_dir / f"walk_forward_predictions_{args.model_type}.csv"
    summary_file = out_dir / f"walk_forward_summary_{args.model_type}.csv"

    results.to_csv(results_file, index=False)
    preds.to_csv(preds_file, index=False)

    summary = (
        results
        .groupby(["feature_set", "asset", "target", "model_type"])
        .agg(
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
            directional_acc_mean=("directional_acc", "mean"),
            n_test_total=("n_test", "sum"),
        )
        .reset_index()
    )

    summary.to_csv(summary_file, index=False)

    print("\nSaved:")
    print(results_file)
    print(preds_file)
    print(summary_file)

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()