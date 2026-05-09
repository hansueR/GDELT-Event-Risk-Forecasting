import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online.common import read_json, target_for_horizon, write_json


def main():
    parser = argparse.ArgumentParser(description="Backfill realized future volatility for pending online predictions.")
    parser.add_argument("--history", default="dashboard/data/prediction_history.json")
    parser.add_argument("--features", default="data/online/latest_features.parquet")
    parser.add_argument("--output", default=None)
    parser.add_argument("--alert_threshold", type=float, default=0.5)
    args = parser.parse_args()

    history_path = Path(args.history)
    payload = read_json(history_path, {"predictions": []})
    predictions = payload.get("predictions", [])
    if not Path(args.features).exists():
        print(f"WARNING: missing features file {args.features}; nothing to backfill.")
        return

    features = pd.read_parquet(args.features)
    features["date"] = pd.to_datetime(features["date"]).dt.date.astype(str)
    index = features.set_index(["asset", "date"])

    updated = 0
    for row in predictions:
        if row.get("status") == "resolved":
            continue
        target = row.get("target") or target_for_horizon(row["horizon"])
        key = (row["asset"], row["prediction_date"])
        if key not in index.index or target not in features.columns:
            continue
        actual = index.loc[key, target]
        if isinstance(actual, pd.Series):
            actual = actual.iloc[-1]
        if pd.isna(actual):
            continue
        actual_log = float(np.log(max(float(actual), 1e-12)))
        row["actual_log_rv"] = actual_log
        if row.get("predicted_log_rv") is not None:
            row["residual"] = actual_log - float(row["predicted_log_rv"])
        if row.get("alert_score") is not None:
            actual_alert = float(row["alert_score"]) >= args.alert_threshold
            if row.get("high_vol_log_threshold") is not None:
                high_actual = actual_log >= row["high_vol_log_threshold"]
                row["hit"] = bool(actual_alert == high_actual)
        row["status"] = "resolved"
        updated += 1

    output = Path(args.output) if args.output else history_path
    write_json(output, {**payload, "predictions": predictions})
    print(f"Resolved {updated} pending predictions.")


if __name__ == "__main__":
    main()
