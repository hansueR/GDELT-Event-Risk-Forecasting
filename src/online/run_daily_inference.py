import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdelt_risk.models import predict_score
from online.common import (
    add_rfi_interactions,
    align_to_schema,
    compact_event_intensity,
    load_joblib,
    model_key,
    model_paths,
    now_utc,
    read_json,
    target_for_horizon,
    write_json,
)


def risk_level(score):
    if score is None:
        return "unknown"
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "elevated"
    return "normal"


def load_history(path):
    payload = read_json(path, {"predictions": []})
    return payload.get("predictions", [])


def merge_history(history, latest):
    keyed = {(r["prediction_date"], r["asset"], r["horizon"], r["method"]): r for r in history}
    for row in latest:
        key = (row["prediction_date"], row["asset"], row["horizon"], row["method"])
        existing = keyed.get(key, {})
        if existing.get("status") == "resolved":
            row["actual_log_rv"] = existing.get("actual_log_rv")
            row["residual"] = existing.get("residual")
            row["hit"] = existing.get("hit")
            row["status"] = "resolved"
        keyed[key] = {**existing, **row}
    return sorted(keyed.values(), key=lambda r: (r["prediction_date"], r["asset"], r["horizon"], r["method"]))


def predict_one(features, artifact_dir, schema_entry, metadata, thresholds):
    asset = schema_entry["asset"]
    horizon = schema_entry["horizon"]
    method = schema_entry["method"]
    rows = features[features["asset"] == asset].sort_values("date")
    if rows.empty:
        return None
    row = rows.tail(1).copy()
    paths = model_paths(artifact_dir, asset, horizon, method)
    classifier = load_joblib(paths["classifier"])
    regressor = load_joblib(paths["regressor"])

    if method == "PO":
        X = align_to_schema(row, schema_entry["columns"])
    elif method == "RFI":
        price_model = load_joblib(paths["price_model"])
        transformer = load_joblib(paths["transformer"])
        price_features = metadata.get("price_features", [])
        X_price = align_to_schema(row, price_features)
        price_score = predict_score(price_model, X_price)
        factors = transformer.transform(row)
        X_raw = pd.concat([X_price, factors], axis=1)
        X_raw = add_rfi_interactions(X_raw, price_score, list(factors.columns), transformer.high_price_threshold_)
        X = align_to_schema(X_raw, schema_entry["columns"])
    else:
        return None

    alert_score = float(predict_score(classifier, X)[0])
    predicted_log_rv = float(regressor.predict(X)[0])
    prediction_date = pd.to_datetime(row["date"].iloc[0]).date().isoformat()
    close = row["adj_close"].iloc[0] if "adj_close" in row.columns else row.get("close", pd.Series([np.nan])).iloc[0]
    event_intensity = float(compact_event_intensity(row).iloc[0])
    threshold_meta = thresholds.get(model_key(asset, horizon, "ALL"), {})
    high_vol_threshold = threshold_meta.get("high_vol_threshold")
    return {
        "prediction_date": prediction_date,
        "asset": asset,
        "horizon": horizon,
        "method": method,
        "target": target_for_horizon(horizon),
        "predicted_log_rv": predicted_log_rv,
        "alert_score": alert_score,
        "risk_level": risk_level(alert_score),
        "generated_at": now_utc(),
        "training_end_date": metadata.get("training_end_date"),
        "actual_log_rv": None,
        "residual": None,
        "hit": None,
        "status": "pending",
        "high_vol_threshold": high_vol_threshold,
        "high_vol_log_threshold": None if high_vol_threshold is None else float(np.log(max(float(high_vol_threshold), 1e-12))),
        "close": None if pd.isna(close) else float(close),
        "event_intensity": event_intensity,
    }


def write_dashboard_series(features, output_dir, warnings):
    rows = []
    work = features.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.date.astype(str)
    intensity = compact_event_intensity(work)
    close = work["adj_close"] if "adj_close" in work.columns else work.get("close", pd.Series([np.nan] * len(work)))
    for horizon in ["3d", "5d"]:
        target = target_for_horizon(horizon)
        for idx, row in work.iterrows():
            rows.append(
                {
                    "date": row["date"],
                    "asset": row["asset"],
                    "close": None if pd.isna(close.iloc[idx]) else float(close.iloc[idx]),
                    "event_intensity": float(intensity.iloc[idx]),
                    "shock_day": False,
                    "horizon": horizon,
                    "realized_volatility": None if target not in work.columns or pd.isna(row.get(target)) else float(row[target]),
                    "forecasts": {},
                    "alert_scores": {},
                }
            )
    write_json(
        Path(output_dir) / "dashboard_series.json",
        {
            "series": rows[-1500:],
            "warnings": warnings,
            "note": "Aggregate event intensity is display-only; online models use saved full feature schemas.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Run daily online inference from saved artifacts and latest features.")
    parser.add_argument("--features", default="data/online/latest_features.parquet")
    parser.add_argument("--artifact_dir", default="artifacts/online")
    parser.add_argument("--output_dir", default="dashboard/data")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    schema_path = artifact_dir / "feature_schema.json"
    metadata_path = artifact_dir / "metadata.json"
    if not schema_path.exists() or not Path(args.features).exists():
        raise SystemExit("Online artifacts or latest feature rows are missing; run train_online_models.py and update_online_data.py before inference.")

    schema = read_json(schema_path, {})
    metadata = read_json(metadata_path, {})
    thresholds = read_json(artifact_dir / "residual_thresholds.json", {})
    metadata["price_features"] = schema.get("price_features", [])
    features = pd.read_parquet(args.features)
    features["date"] = pd.to_datetime(features["date"])

    latest = []
    warnings = []
    for entry in schema.get("models", {}).values():
        try:
            pred = predict_one(features, artifact_dir, entry, metadata, thresholds)
            if pred:
                latest.append(pred)
        except Exception as exc:
            warnings.append(f"{entry.get('asset')} {entry.get('horizon')} {entry.get('method')}: {exc}")
            print(f"WARNING: {warnings[-1]}")

    if not latest:
        detail = " ".join(warnings) if warnings else "No model entries were available in feature_schema.json."
        raise SystemExit(f"No online predictions were generated. {detail}")

    history_path = output_dir / "prediction_history.json"
    history = merge_history(load_history(history_path), latest)
    write_dashboard_series(features, output_dir, warnings)
    write_json(output_dir / "latest_predictions.json", {"generated_at": now_utc(), "warnings": warnings, "predictions": latest})
    write_json(output_dir / "prediction_history.json", {"generated_at": now_utc(), "warnings": warnings, "predictions": history})
    write_json(output_dir / "model_metadata.json", {**metadata, "feature_schema_generated_at": schema.get("generated_at"), "warnings": warnings})
    print(f"Wrote online predictions to {output_dir}")


if __name__ == "__main__":
    main()
