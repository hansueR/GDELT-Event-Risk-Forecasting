import argparse
import json
from pathlib import Path


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing dashboard data file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rows(payload, key):
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def fail_if_placeholder(records, name):
    bad = [row for row in records if str(row.get("method", "")).lower() == "placeholder"]
    if bad:
        raise ValueError(f"{name} contains placeholder method rows.")


def main():
    parser = argparse.ArgumentParser(description="Validate dashboard JSON before deployment.")
    parser.add_argument("--data_dir", default="dashboard/data")
    parser.add_argument("--mode", choices=["static", "online"], default="online")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    series_payload = read_json(data_dir / "dashboard_series.json")
    summary_payload = read_json(data_dir / "summary_metrics.json")
    alert_payload = read_json(data_dir / "alert_metrics.json")
    metadata = read_json(data_dir / "metadata.json")

    series = rows(series_payload, "series")
    summary = rows(summary_payload, "metrics")
    alert = rows(alert_payload, "metrics")
    fail_if_placeholder(series, "dashboard_series.json")
    fail_if_placeholder(summary, "summary_metrics.json")
    fail_if_placeholder(alert, "alert_metrics.json")

    if metadata.get("uses_simulated_data") is True:
        raise ValueError("metadata.json reports uses_simulated_data=true.")

    if args.mode == "online":
        latest_payload = read_json(data_dir / "latest_predictions.json")
        history_payload = read_json(data_dir / "prediction_history.json")
        model_metadata = read_json(data_dir / "model_metadata.json")
        latest = rows(latest_payload, "predictions")
        history = rows(history_payload, "predictions")
        fail_if_placeholder(latest, "latest_predictions.json")
        fail_if_placeholder(history, "prediction_history.json")

        if not latest:
            raise ValueError("latest_predictions.json has no online predictions.")
        missing_scores = [
            row for row in latest
            if row.get("predicted_log_rv") is None or row.get("alert_score") is None
        ]
        if missing_scores:
            raise ValueError(f"{len(missing_scores)} latest prediction rows are missing model outputs.")
        if not model_metadata.get("training_end_date"):
            raise ValueError("model_metadata.json is missing training_end_date.")

    print(f"Dashboard data validation passed for {args.mode} mode.")


if __name__ == "__main__":
    main()
