import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gdelt_risk.features import DEFAULT_ASSETS, DEFAULT_TARGETS, get_event_columns, get_price_features
from gdelt_risk.models import make_classifier, predict_score
from online.common import (
    METHODS,
    add_rfi_interactions,
    align_to_schema,
    git_commit,
    high_vol_labels,
    horizon_for_target,
    make_regressor,
    model_key,
    model_paths,
    now_utc,
    parse_list,
    safe_log_rv,
    save_joblib,
    write_json,
    OnlineRFITransformer,
)


def train_po(train, price_features, y_class, y_log, random_seed):
    X = align_to_schema(train, price_features)
    classifier = make_classifier("logistic", random_seed=random_seed, C=1.0)
    classifier.fit(X, y_class)
    regressor = make_regressor(random_seed)
    regressor.fit(X, y_log)
    return classifier, regressor, list(X.columns)


def train_rfi(train, price_features, y_class, y_log, random_seed, event_prefix, max_zero_ratio, min_nonzero_count):
    X_price = align_to_schema(train, price_features)
    price_model = make_classifier("logistic", random_seed=random_seed, C=1.0)
    price_model.fit(X_price, y_class)
    price_score = predict_score(price_model, X_price)
    residual = pd.Series(y_class, index=train.index, dtype=float) - pd.Series(price_score, index=train.index, dtype=float)

    transformer = OnlineRFITransformer(
        event_prefix=event_prefix,
        max_zero_ratio=max_zero_ratio,
        min_nonzero_count=min_nonzero_count,
        random_seed=random_seed,
    ).fit(train, residual)
    factors = transformer.transform(train)
    high_price_threshold = float(pd.Series(price_score).quantile(0.70))
    X = pd.concat([X_price, factors], axis=1)
    X = add_rfi_interactions(X, price_score, list(factors.columns), high_price_threshold)

    classifier = make_classifier("logistic", random_seed=random_seed, C=1.0)
    classifier.fit(X, y_class)
    regressor = make_regressor(random_seed)
    regressor.fit(X, y_log)
    transformer.high_price_threshold_ = high_price_threshold
    return classifier, regressor, price_model, transformer, list(X.columns)


def main():
    parser = argparse.ArgumentParser(description="Train final online models on all available historical rows.")
    parser.add_argument("--model_table", required=True)
    parser.add_argument("--output_dir", default="artifacts/online")
    parser.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--event_prefix", default="ddfe_")
    parser.add_argument("--high_vol_quantile", type=float, default=0.80)
    parser.add_argument("--max_zero_ratio", type=float, default=0.99)
    parser.add_argument("--min_nonzero_count", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.model_table)
    df["date"] = pd.to_datetime(df["date"])

    assets = parse_list(args.assets, DEFAULT_ASSETS)
    targets = parse_list(args.targets, DEFAULT_TARGETS)
    price_features = get_price_features(df)
    event_features = get_event_columns(df, args.event_prefix)
    if not price_features:
        raise ValueError("No raw price feature columns found.")
    if not event_features:
        print("WARNING: no event columns found; RFI models will be skipped.")

    feature_schema = {
        "generated_at": now_utc(),
        "price_features": price_features,
        "event_features": event_features,
        "models": {},
    }
    thresholds = {}
    methods_trained = set()

    for asset in assets:
        asset_df = df[df["asset"] == asset].sort_values("date").copy()
        if asset_df.empty:
            print(f"WARNING: no rows for asset {asset}")
            continue
        for target in targets:
            if target not in asset_df.columns:
                print(f"WARNING: missing target {target}; skip {asset}")
                continue
            horizon = horizon_for_target(target)
            train = asset_df.dropna(subset=[target]).replace([np.inf, -np.inf], np.nan).copy()
            if len(train) < 50:
                print(f"WARNING: insufficient rows for {asset} {horizon}: {len(train)}")
                continue
            y_class, threshold = high_vol_labels(train[target], args.high_vol_quantile)
            if y_class.nunique() < 2:
                print(f"WARNING: one-class target for {asset} {horizon}; skip")
                continue
            y_log = safe_log_rv(train[target])
            thresholds[model_key(asset, horizon, "ALL")] = {
                "target": target,
                "high_vol_quantile": args.high_vol_quantile,
                "high_vol_threshold": threshold,
            }

            classifier, regressor, columns = train_po(train, price_features, y_class, y_log, args.random_seed)
            paths = model_paths(out_dir, asset, horizon, "PO")
            save_joblib(classifier, paths["classifier"])
            save_joblib(regressor, paths["regressor"])
            feature_schema["models"][model_key(asset, horizon, "PO")] = {
                "method": "PO",
                "asset": asset,
                "horizon": horizon,
                "target": target,
                "columns": columns,
                "artifact_dir": str(paths["dir"]),
            }
            methods_trained.add("PO")

            if event_features:
                classifier, regressor, price_model, transformer, columns = train_rfi(
                    train,
                    price_features,
                    y_class,
                    y_log,
                    args.random_seed,
                    args.event_prefix,
                    args.max_zero_ratio,
                    args.min_nonzero_count,
                )
                paths = model_paths(out_dir, asset, horizon, "RFI")
                save_joblib(classifier, paths["classifier"])
                save_joblib(regressor, paths["regressor"])
                save_joblib(price_model, paths["price_model"])
                save_joblib(transformer, paths["transformer"])
                feature_schema["models"][model_key(asset, horizon, "RFI")] = {
                    "method": "RFI",
                    "asset": asset,
                    "horizon": horizon,
                    "target": target,
                    "columns": columns,
                    "artifact_dir": str(paths["dir"]),
                    "transformer_metadata": transformer.metadata(),
                    "high_price_threshold": transformer.high_price_threshold_,
                }
                methods_trained.add("RFI")
            print(f"Trained online models for {asset} {horizon}")

    write_json(out_dir / "feature_schema.json", feature_schema)
    write_json(
        out_dir / "event_feature_config.json",
        {
            "event_prefix": args.event_prefix,
            "event_features": event_features,
            "max_zero_ratio": args.max_zero_ratio,
            "min_nonzero_count": args.min_nonzero_count,
            "note": "Online inference must build these full event columns; aggregate intensity is display-only.",
        },
    )
    write_json(out_dir / "price_feature_config.json", {"price_features": price_features})
    write_json(out_dir / "residual_thresholds.json", thresholds)
    write_json(
        out_dir / "metadata.json",
        {
            "generated_at": now_utc(),
            "training_end_date": df["date"].max().date().isoformat(),
            "assets": assets,
            "horizons": [horizon_for_target(t) for t in targets if t in DEFAULT_TARGETS],
            "methods": sorted(methods_trained),
            "target_names": targets,
            "git_commit": git_commit(),
            "model_table": args.model_table,
        },
    )
    print(f"Saved online artifacts to {out_dir}")


if __name__ == "__main__":
    main()
