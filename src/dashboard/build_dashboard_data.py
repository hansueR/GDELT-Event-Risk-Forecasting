import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online.paths import default_model_table, default_results_dir


ASSETS = ["Gold", "QQQ", "WTI_Oil"]
TARGETS = {
    "target_rv_return_3d": "3d",
    "target_rv_return_5d": "5d",
}
METHOD_DIRS = {
    "price_only": "ddfe_price_only_v1",
    "raw_concat": "ddfe_raw_concat_v1",
    "grouped_svd": "ddfe_grouped_svd_v1",
    "residual_grouped_svd": "ddfe_residual_grouped_svd_v1",
    "residual_grouped_pls": "ddfe_residual_grouped_pls_v1",
    "residual_fusion_interactions": "ddfe_residual_fusion_interactions_v1",
}
SUMMARY_COLUMNS = ["method", "asset", "horizon", "mae", "rmse", "r2", "qlike", "auc", "ap", "p_at_10", "r_at_10"]
ALERT_COLUMNS = ["method", "asset", "horizon", "auc", "ap", "p_at_10", "r_at_10"]
SERIES_COLUMNS = ["date", "asset", "close", "event_intensity", "shock_day", "horizon", "realized_volatility", "forecasts", "alert_scores"]
HISTORY_COLUMNS = [
    "prediction_date",
    "asset",
    "horizon",
    "method",
    "predicted_log_rv",
    "actual_log_rv",
    "alert_score",
    "risk_level",
    "status",
]


def warn(message, warnings):
    warnings.append(message)
    print(f"WARNING: {message}")


def clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    return value


def records(df):
    return [{k: clean_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def read_table(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() in [".csv", ".txt"]:
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported input table format: {path}")


def horizon_from_target(target):
    if target in TARGETS:
        return TARGETS[target]
    text = str(target)
    if "3d" in text:
        return "3d"
    if "5d" in text:
        return "5d"
    return text


def pick_close_column(df):
    for col in ["adj_close", "close", "Adj Close", "Close"]:
        if col in df.columns:
            return col
    price_like = [c for c in df.columns if "close" in c.lower()]
    return price_like[0] if price_like else None


def pick_event_intensity(df):
    for col in ["event_intensity", "log_event_count", "ddfe_global_event_total"]:
        if col in df.columns:
            return col
    log_count_cols = [c for c in df.columns if "log_count" in c.lower()]
    if log_count_cols:
        return log_count_cols[:25]
    event_cols = [c for c in df.columns if c.startswith("ddfe_")]
    return event_cols[:25] if event_cols else None


def pick_shock_column(df):
    for col in ["shock_day", "is_shock_day"]:
        if col in df.columns:
            return col
    shock_cols = [c for c in df.columns if "shock" in c.lower()]
    return shock_cols[0] if shock_cols else None


def pick_forecast_columns(df):
    skip = {"date", "asset"}
    forecast_cols = []
    for col in df.columns:
        name = col.lower()
        if col in skip or col in TARGETS:
            continue
        if "forecast" in name or "prediction" in name or name.endswith("_pred"):
            forecast_cols.append(col)
    return forecast_cols[:12]


def build_base_series(model_table, warnings, max_rows_per_asset_horizon=None):
    if model_table.empty:
        warn("Model table is missing or empty; dashboard time series will be empty.", warnings)
        return pd.DataFrame(columns=SERIES_COLUMNS)

    df = model_table.copy()
    if "date" not in df.columns or "asset" not in df.columns:
        warn("Model table must contain date and asset columns; dashboard time series will be empty.", warnings)
        return pd.DataFrame(columns=SERIES_COLUMNS)

    close_col = pick_close_column(df)
    event_col = pick_event_intensity(df)
    shock_col = pick_shock_column(df)
    forecast_cols = pick_forecast_columns(df)
    target_cols = [col for col in TARGETS if col in df.columns]
    if close_col is None:
        warn("No close or adjusted close column found; close values will be null.", warnings)
    if event_col is None:
        warn("No event intensity column found; event values will be null.", warnings)
    if not target_cols:
        warn("No 3d or 5d realized volatility target columns found; realized volatility values will be null.", warnings)
        target_cols = ["target_rv_return_3d", "target_rv_return_5d"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    rows = []
    for asset in ASSETS:
        asset_df = df[df["asset"] == asset].sort_values("date")
        if asset_df.empty:
            continue
        for target in target_cols:
            horizon = horizon_from_target(target)
            cols = ["date", "asset"]
            for col in [close_col, shock_col, target]:
                if col and col in asset_df.columns and col not in cols:
                    cols.append(col)
            cols.extend([c for c in forecast_cols if c in asset_df.columns and c not in cols])
            if isinstance(event_col, str) and event_col in asset_df.columns:
                cols.append(event_col)
            elif isinstance(event_col, list):
                cols.extend([c for c in event_col if c in asset_df.columns])

            out = asset_df[cols].copy()
            if max_rows_per_asset_horizon is not None:
                out = out.tail(max_rows_per_asset_horizon)
            if isinstance(event_col, list) and event_col:
                out["event_intensity"] = out[[c for c in event_col if c in out.columns]].sum(axis=1, skipna=True)
            elif isinstance(event_col, str) and event_col in out.columns:
                out["event_intensity"] = out[event_col]
            else:
                out["event_intensity"] = np.nan
            out["close"] = out[close_col] if close_col and close_col in out.columns else np.nan
            out["shock_day"] = out[shock_col].fillna(0).astype(float).gt(0) if shock_col and shock_col in out.columns else False
            out["horizon"] = horizon
            out["realized_volatility"] = out[target] if target in out.columns else np.nan
            out["forecasts"] = [
                {col: clean_value(row[col]) for col in forecast_cols if col in out.columns and clean_value(row[col]) is not None}
                for _, row in out.iterrows()
            ]
            out["alert_scores"] = [{} for _ in range(len(out))]
            rows.append(out[["date", "asset", "close", "event_intensity", "shock_day", "horizon", "realized_volatility", "forecasts", "alert_scores"]])

    if not rows:
        warn("No matching asset rows found in model table; dashboard time series will be empty.", warnings)
        return pd.DataFrame(columns=SERIES_COLUMNS)

    return pd.concat(rows, ignore_index=True)


def method_from_dir(path):
    parent = path.parent.name
    for method, dirname in METHOD_DIRS.items():
        if parent == dirname:
            return method
    return parent


def find_result_files(results_dir, filename):
    root = Path(results_dir)
    if not root.exists():
        return []
    known = [root / dirname / filename for dirname in METHOD_DIRS.values()]
    found = [path for path in known if path.exists()]
    seen = set(found)
    for path in root.rglob(filename):
        if path not in seen:
            found.append(path)
    return found


def read_result_frames(results_dir, filename, warnings):
    frames = []
    for path in find_result_files(results_dir, filename):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            warn(f"Could not read {path}: {exc}", warnings)
            continue
        if "method" not in df.columns:
            df["method"] = method_from_dir(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def attach_scores(series_df, predictions, warnings):
    if series_df.empty or predictions.empty:
        if predictions.empty:
            warn("No fold_predictions.csv files found; alert scores will be empty.", warnings)
        return series_df

    required = {"date", "asset", "target", "method", "y_score"}
    if not required.issubset(predictions.columns):
        warn("fold_predictions.csv files are missing required date/asset/target/method/y_score columns.", warnings)
        return series_df

    out = series_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    predictions["horizon"] = predictions["target"].map(horizon_from_target)

    score_map = {}
    for row in predictions[["date", "asset", "horizon", "method", "y_score"]].dropna(subset=["date"]).itertuples(index=False):
        key = (row.date, row.asset, row.horizon)
        score_map.setdefault(key, {})[row.method] = clean_value(row.y_score)

    out["alert_scores"] = [
        score_map.get((row.date, row.asset, row.horizon), row.alert_scores if isinstance(row.alert_scores, dict) else {})
        for row in out.itertuples(index=False)
    ]
    return out


def normalize_metric_columns(df):
    out = pd.DataFrame()
    if df.empty:
        return out
    out["method"] = df["method"] if "method" in df.columns else "unknown"
    out["asset"] = df["asset"] if "asset" in df.columns else "All"
    if "horizon" in df.columns:
        out["horizon"] = df["horizon"]
    elif "target" in df.columns:
        out["horizon"] = df["target"].map(horizon_from_target)
    else:
        out["horizon"] = "All"

    aliases = {
        "mae": ["mae", "MAE"],
        "rmse": ["rmse", "RMSE"],
        "r2": ["r2", "R2", "r_squared"],
        "qlike": ["qlike", "QLIKE"],
        "auc": ["auc", "AUC"],
        "ap": ["ap", "average_precision", "AP"],
        "p_at_10": ["p_at_10", "precision_at_10", "precision_at_top10pct", "precision_at_top20pct"],
        "r_at_10": ["r_at_10", "recall_at_10", "recall_at_top10pct", "recall_at_top20pct"],
    }
    for dest, sources in aliases.items():
        source = next((col for col in sources if col in df.columns), None)
        out[dest] = df[source] if source else np.nan
    return out


def precision_recall_at_top_pct(y_true, y_score, pct=0.10):
    frame = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
    if frame.empty:
        return np.nan, np.nan
    frame["y_true"] = frame["y_true"].astype(int)
    k = max(1, int(math.ceil(len(frame) * pct)))
    top = frame.sort_values("y_score", ascending=False).head(k)
    positives = float(top["y_true"].sum())
    total_positives = float(frame["y_true"].sum())
    precision = positives / k
    recall = positives / total_positives if total_positives else np.nan
    return precision, recall


def top10_from_predictions(predictions):
    if predictions.empty or not {"method", "asset", "target", "y_true", "y_score"}.issubset(predictions.columns):
        return pd.DataFrame(columns=["method", "asset", "horizon", "p_at_10", "r_at_10"])
    work = predictions.copy()
    work["horizon"] = work["target"].map(horizon_from_target)
    rows = []
    for keys, group in work.groupby(["method", "asset", "horizon"], dropna=False):
        precision, recall = precision_recall_at_top_pct(group["y_true"], group["y_score"], pct=0.10)
        rows.append(
            {
                "method": keys[0],
                "asset": keys[1],
                "horizon": keys[2],
                "p_at_10": precision,
                "r_at_10": recall,
            }
        )
    return pd.DataFrame(rows)


def build_metrics(results_dir, warnings):
    summary = read_result_frames(results_dir, "summary_metrics.csv", warnings)
    folds = read_result_frames(results_dir, "fold_metrics.csv", warnings)
    predictions = read_result_frames(results_dir, "fold_predictions.csv", warnings)
    source = summary if not summary.empty else folds
    if source.empty:
        warn("No summary_metrics.csv or fold_metrics.csv files found; dashboard metrics will be empty.", warnings)
        return pd.DataFrame(columns=SUMMARY_COLUMNS), pd.DataFrame(columns=ALERT_COLUMNS)

    normalized = normalize_metric_columns(source)
    metric_cols = [c for c in SUMMARY_COLUMNS if c not in ["method", "asset", "horizon"]]
    grouped = (
        normalized.groupby(["method", "asset", "horizon"], dropna=False)[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    top10 = top10_from_predictions(predictions)
    if not top10.empty:
        grouped = grouped.drop(columns=["p_at_10", "r_at_10"], errors="ignore").merge(
            top10, on=["method", "asset", "horizon"], how="left"
        )
    return grouped[SUMMARY_COLUMNS], grouped[ALERT_COLUMNS]


def risk_level(score):
    if score is None or pd.isna(score):
        return "unknown"
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "elevated"
    return "normal"


def build_prediction_history(predictions, warnings):
    if predictions.empty:
        warn("No fold_predictions.csv files found; historical prediction charts will be empty.", warnings)
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    required = {"date", "asset", "target", "method"}
    if not required.issubset(predictions.columns):
        warn("fold_predictions.csv files are missing required date/asset/target/method columns.", warnings)
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    work = predictions.copy()
    work["prediction_date"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work["horizon"] = work["target"].map(horizon_from_target)
    work["predicted_log_rv"] = work["y_pred"] if "y_pred" in work.columns else np.nan

    # For alert-only experiments y_true is a binary high-volatility label, not a log-RV value.
    is_log_target = work["target"].astype(str).str.contains("log", case=False, na=False)
    work["actual_log_rv"] = np.where(is_log_target & work.get("y_true", pd.Series(np.nan, index=work.index)).notna(), work["y_true"], np.nan)

    if "y_score" in work.columns:
        work["alert_score"] = work["y_score"]
    elif "residual_score" in work.columns:
        work["alert_score"] = work["residual_score"]
    else:
        work["alert_score"] = np.nan

    work["risk_level"] = [risk_level(score) for score in work["alert_score"]]
    work["status"] = "historical"
    out = work[HISTORY_COLUMNS].dropna(subset=["prediction_date", "asset", "horizon", "method"]).copy()
    out = out.drop_duplicates(subset=["prediction_date", "asset", "horizon", "method"], keep="last")
    return out.sort_values(["prediction_date", "asset", "horizon", "method"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Build small static JSON files for the public dashboard.")
    default_table = default_model_table()
    parser.add_argument(
        "--model_table",
        default=str(default_table) if default_table.exists() else "",
        help="Path to model table parquet/csv/json. Defaults to GDELT_DATA_ROOT/modeling when available.",
    )
    parser.add_argument(
        "--results_dir",
        default=str(default_results_dir()),
        help="Directory containing experiment output subdirectories.",
    )
    parser.add_argument("--output_dir", default="dashboard/data", help="Dashboard data output directory.")
    parser.add_argument(
        "--max_rows_per_asset_horizon",
        type=int,
        default=None,
        help="Optional display row cap. Leave unset to publish the full real historical series.",
    )
    args = parser.parse_args()

    warnings = []
    output_dir = Path(args.output_dir)
    model_table = read_table(args.model_table) if args.model_table else pd.DataFrame()
    if args.model_table and model_table.empty:
        warn(f"Model table not found or empty: {args.model_table}", warnings)

    series = build_base_series(model_table, warnings, max_rows_per_asset_horizon=args.max_rows_per_asset_horizon)
    predictions = read_result_frames(args.results_dir, "fold_predictions.csv", warnings)
    series = attach_scores(series, predictions, warnings)
    summary, alert = build_metrics(args.results_dir, warnings)
    history = build_prediction_history(predictions, warnings)

    metadata = {
        "project": "GDELT Event-Driven Volatility Risk Forecasting",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "uses_simulated_data": False,
        "has_model_table": not model_table.empty,
        "has_metrics": not summary.empty,
        "series_rows": int(len(series)),
        "summary_metric_rows": int(len(summary)),
        "alert_metric_rows": int(len(alert)),
        "historical_prediction_rows": int(len(history)),
        "row_policy": "full_history" if args.max_rows_per_asset_horizon is None else f"tail_{args.max_rows_per_asset_horizon}_per_asset_horizon",
        "model_table": str(args.model_table) if args.model_table else None,
        "results_dir": str(args.results_dir),
        "warnings": warnings,
    }

    write_json(output_dir / "dashboard_series.json", {"series": records(series), "warnings": warnings})
    write_json(output_dir / "summary_metrics.json", {"metrics": records(summary), "warnings": warnings})
    write_json(output_dir / "alert_metrics.json", {"metrics": records(alert), "warnings": warnings})
    write_json(
        output_dir / "latest_predictions.json",
        {
            "generated_at": metadata["generated_at"],
            "mode": "static_historical",
            "predictions": [],
            "warnings": [],
        },
    )
    write_json(
        output_dir / "prediction_history.json",
        {
            "generated_at": metadata["generated_at"],
            "mode": "static_historical",
            "predictions": records(history),
            "warnings": warnings,
        },
    )
    write_json(
        output_dir / "model_metadata.json",
        {
            "generated_at": metadata["generated_at"],
            "mode": "static_historical",
            "training_end_date": None,
            "model_table": str(args.model_table) if args.model_table else None,
            "warnings": [],
        },
    )
    write_json(output_dir / "metadata.json", metadata)
    print(f"Wrote dashboard data to {output_dir}")


if __name__ == "__main__":
    main()
