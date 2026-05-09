import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.download_asset_prices import DEFAULT_TICKERS, add_price_features, download_one_ticker
from online.common import compact_event_intensity, now_utc, parse_list, read_json, write_json


def download_prices(assets, start_date, end_date):
    parts = []
    for asset in assets:
        ticker = DEFAULT_TICKERS.get(asset, asset)
        print(f"Downloading {asset} prices from {start_date} to {end_date}")
        price = download_one_ticker(ticker, start_date, end_date)
        price["asset"] = asset
        parts.append(price)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_event_features(path, trading_calendar):
    if not path:
        return pd.DataFrame({"date": pd.to_datetime(trading_calendar)})
    event_path = Path(path)
    if not event_path.exists():
        raise FileNotFoundError(f"Event feature file not found: {event_path}")
    if event_path.suffix.lower() in [".parquet", ".pq"]:
        events = pd.read_parquet(event_path)
    else:
        events = pd.read_csv(event_path)
    if "date" not in events.columns:
        raise ValueError(f"Event feature file must contain a date column: {event_path}")
    events["date"] = pd.to_datetime(events["date"])
    calendar = pd.DataFrame({"date": pd.to_datetime(trading_calendar).sort_values().drop_duplicates()})
    return calendar.merge(events, on="date", how="left")


def main():
    parser = argparse.ArgumentParser(description="Build latest online feature rows using the saved training schema.")
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--end_date", default=None)
    parser.add_argument("--output_dir", default="data/online")
    parser.add_argument("--assets", default="Gold,QQQ,WTI_Oil")
    parser.add_argument("--feature_schema", default="artifacts/online/feature_schema.json")
    parser.add_argument("--event_features_file", default="", help="Optional prebuilt daily event features with training-compatible columns.")
    parser.add_argument("--lookback_days", type=int, default=60)
    args = parser.parse_args()

    schema = read_json(args.feature_schema, {})
    assets = parse_list(args.assets, schema.get("assets", ["Gold", "QQQ", "WTI_Oil"]))
    training_event_cols = schema.get("event_features", [])
    if training_event_cols and not args.event_features_file:
        raise SystemExit("Training schema contains event features; provide --event_features_file with real daily event feature columns.")
    end = pd.to_datetime(args.end_date).date() if args.end_date else date.today() + timedelta(days=1)
    start = pd.to_datetime(args.start_date).date() if args.start_date else end - timedelta(days=args.lookback_days)

    warnings = []
    try:
        prices = download_prices(assets, start.isoformat(), end.isoformat())
        features = add_price_features(prices, horizons=(1, 3, 5))
    except Exception as exc:
        raise SystemExit(f"Price download failed; no online feature file was written: {exc}") from exc

    features["date"] = pd.to_datetime(features["date"])
    trading_calendar = features["date"].drop_duplicates()
    event_features = load_event_features(args.event_features_file, trading_calendar)
    event_cols = [c for c in event_features.columns if c != "date"]
    if training_event_cols and event_cols:
        latest_price_date = pd.to_datetime(features["date"]).max()
        covered = event_features.dropna(subset=training_event_cols, how="all") if set(training_event_cols).issubset(event_features.columns) else pd.DataFrame()
        latest_event_date = pd.to_datetime(covered["date"]).max() if not covered.empty else pd.NaT
        if pd.isna(latest_event_date) or latest_event_date < latest_price_date:
            raise SystemExit(
                "Online event features are stale: "
                f"latest event date {latest_event_date.date() if not pd.isna(latest_event_date) else 'NA'} "
                f"is before latest price date {latest_price_date.date()}. "
                "Update GDELT event features before running online inference."
            )
    features = features.merge(event_features, on="date", how="left")
    if event_cols:
        features[event_cols] = features[event_cols].fillna(0.0)

    missing_event_cols = [c for c in training_event_cols if c not in features.columns]
    extra_event_cols = [c for c in event_cols if c not in training_event_cols]
    if missing_event_cols:
        sample = ", ".join(missing_event_cols[:10])
        raise SystemExit(f"{len(missing_event_cols)} training event columns are missing from online features: {sample}")

    price_cols = schema.get("price_features", [])
    missing_price_cols = [col for col in price_cols if col not in features.columns]
    if missing_price_cols:
        sample = ", ".join(missing_price_cols[:10])
        raise SystemExit(f"{len(missing_price_cols)} training price columns are missing from online features: {sample}")
    for col in price_cols:
        features[col] = features[col].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    features = features.sort_values(["asset", "date"]).reset_index(drop=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_file = output_dir / "latest_features.parquet"
    features.to_parquet(feature_file, index=False)

    display = features[["date", "asset"]].copy()
    display["close"] = features["adj_close"] if "adj_close" in features.columns else features.get("close")
    display["event_intensity"] = compact_event_intensity(features)
    display["display_only"] = True
    display_records = display.tail(500).copy()
    display_records["date"] = display_records["date"].dt.date.astype(str)
    write_json(
        output_dir / "display_event_intensity.json",
        {
            "generated_at": now_utc(),
            "display_only": True,
            "note": "Aggregate event intensity is for visualization only. Online models use the full saved feature schema.",
            "warnings": warnings,
            "missing_training_event_columns": missing_event_cols[:200],
            "extra_online_event_columns": extra_event_cols[:200],
            "series": display_records.to_dict(orient="records"),
        },
    )
    print(f"Wrote {feature_file}")


if __name__ == "__main__":
    main()
