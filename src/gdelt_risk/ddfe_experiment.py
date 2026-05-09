import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gdelt_risk.event_preprocess import EventColumnFilter
from gdelt_risk.event_reduction import (
    DEFAULT_MAX_FEATURES_PER_FAMILY,
    fit_transform_grouped_pls,
    fit_transform_grouped_svd,
    select_event_features_by_residual,
)
from gdelt_risk.features import DEFAULT_ASSETS, DEFAULT_TARGETS, get_event_columns, get_price_features, map_event_families
from gdelt_risk.metrics import classification_metrics, summarize_metrics
from gdelt_risk.models import make_classifier, predict_score
from gdelt_risk.residual_targets import price_oof_predictions_by_year, residual_continuous
from gdelt_risk.walk_forward import iter_year_folds, make_high_vol_labels

METHOD_OUTPUTS = {
    "price_only": "ddfe_price_only_v1",
    "raw_concat": "ddfe_raw_concat_v1",
    "grouped_svd": "ddfe_grouped_svd_v1",
    "residual_grouped_svd": "ddfe_residual_grouped_svd_v1",
    "residual_grouped_pls": "ddfe_residual_grouped_pls_v1",
    "residual_fusion_interactions": "ddfe_residual_fusion_interactions_v1",
}


def parse_list(value, default):
    if value is None:
        return default
    if isinstance(value, list):
        return value
    return [v.strip() for v in value.split(",") if v.strip()]


def common_parser(method, default_output_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/project/hrao/GDELT/modeling/model_table_ddfe_root_country_clean_full.parquet")
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--assets", default=",".join(DEFAULT_ASSETS))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--test_year_start", type=int, default=2019)
    parser.add_argument("--test_year_end", type=int, default=None)
    parser.add_argument("--event_prefix", default="ddfe_")
    parser.add_argument(
        "--model_type",
        default="logistic",
        choices=["logistic", "random_forest", "hist_gradient_boosting"],
    )
    parser.add_argument("--high_vol_quantile", type=float, default=0.80)
    parser.add_argument("--min_train_rows", type=int, default=500)
    parser.add_argument("--min_test_rows", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_zero_ratio", type=float, default=0.99)
    parser.add_argument("--min_nonzero_count", type=int, default=30)
    parser.add_argument("--residual_method", choices=["pls", "svd"], default="pls")
    parser.add_argument(
        "--allow_in_sample_price_score_fallback",
        action="store_true",
        help=(
            "If set, residual stacking rows without OOF price scores use full-train "
            "in-sample price scores. Default is strict: drop rows without OOF scores."
        ),
    )
    parser.add_argument("--min_inner_train_rows", type=int, default=250)
    parser.add_argument("--min_inner_valid_rows", type=int, default=30)
    parser.add_argument("--method", default=method)
    return parser


def _fit_price_score(train, test, price_features, y_train, args):
    if pd.Series(y_train).nunique() < 2:
        return None, None, None
    model = make_classifier(model_type=args.model_type, random_seed=args.random_seed, C=1.0)
    model.fit(train[price_features], y_train)
    return model, predict_score(model, train[price_features]), predict_score(model, test[price_features])


def _event_filter_row(asset, target, year, event_filter):
    row = {
        "asset": asset,
        "target": target,
        "test_year": int(year),
    }
    row.update({k: v for k, v in event_filter.metadata_.items() if k != "kept_columns_by_family"})
    return row


def _prediction_frame(test, asset, target, year, method, y_test, score, threshold):
    return pd.DataFrame(
        {
            "date": test["date"].values,
            "asset": asset,
            "target": target,
            "test_year": int(year),
            "method": method,
            "y_true": y_test,
            "y_score": score,
            "high_vol_threshold": threshold,
        }
    )


def _metric_row(asset, target, year, method, args, n_train, y_test, score, threshold):
    row = {
        "method": method,
        "asset": asset,
        "target": target,
        "model_type": args.model_type,
        "test_year": int(year),
        "n_train": int(n_train),
        "high_vol_threshold": threshold,
    }
    row.update(classification_metrics(y_test, score))
    return row


def _clean_price_frame(frame, price_features, target):
    needed = ["date", "asset", target] + price_features
    keep = [c for c in needed if c in frame.columns]
    return frame.dropna(subset=[target]).replace([np.inf, -np.inf], np.nan), keep


def _make_event_factors(method, train, test, residual, args, selected_only=False):
    event_filter = EventColumnFilter(
        event_prefix=args.event_prefix,
        max_zero_ratio=args.max_zero_ratio,
        min_nonzero_count=args.min_nonzero_count,
    )
    X_event_train = event_filter.fit_transform(train)
    X_event_test = event_filter.transform(test)
    families = event_filter.families_
    selected_df = pd.DataFrame()

    if selected_only:
        selected_cols, selected_df = select_event_features_by_residual(
            X_event_train,
            residual,
            families,
            max_features_per_family=DEFAULT_MAX_FEATURES_PER_FAMILY,
        )
        X_event_train = X_event_train[selected_cols]
        X_event_test = X_event_test[selected_cols]
        families = {c: families[c] for c in selected_cols}

    if method == "pls":
        train_fac, test_fac, _, metadata = fit_transform_grouped_pls(X_event_train, X_event_test, residual, families)
    else:
        train_fac, test_fac, _, metadata = fit_transform_grouped_svd(
            X_event_train,
            X_event_test,
            families,
            random_seed=args.random_seed,
        )
    return train_fac, test_fac, event_filter, selected_df, metadata


def _regime_interactions(train_X, test_X, train, test, price_train_score, price_test_score, event_factor_cols):
    train_out = train_X.copy()
    test_out = test_X.copy()

    train_out["price_score"] = price_train_score
    test_out["price_score"] = price_test_score

    train_out["price_uncertainty"] = np.abs(price_train_score - 0.5)
    test_out["price_uncertainty"] = np.abs(price_test_score - 0.5)

    q70 = float(pd.Series(price_train_score).quantile(0.70))
    train_out["high_price_risk"] = (price_train_score >= q70).astype(int)
    test_out["high_price_risk"] = (price_test_score >= q70).astype(int)

    interaction_vars = ["price_score", "price_uncertainty", "high_price_risk"]

    for fac in event_factor_cols:
        for reg in interaction_vars:
            name = f"{fac}_x_{reg}"
            train_out[name] = train_out[fac].values * train_out[reg].values
            test_out[name] = test_out[fac].values * test_out[reg].values

    return train_out, test_out


def run_experiment(method, args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.input)
    df["date"] = pd.to_datetime(df["date"])
    assets = parse_list(args.assets, DEFAULT_ASSETS)
    targets = parse_list(args.targets, DEFAULT_TARGETS)
    price_features = get_price_features(df)
    print("Raw price features used:", price_features)
    event_cols = get_event_columns(df, args.event_prefix)
    if not price_features:
        available_price_like = [
            c
            for c in df.columns
            if any(k in c.lower() for k in ["open", "high", "low", "close", "adj", "volume", "price"])
        ]
        raise ValueError(
            "No raw OHLCV price features found. Expected columns such as "
            "adj_close, close, high, low, open, volume. "
            f"Available price-like columns: {available_price_like[:80]}"
        )
    if method != "price_only" and not event_cols:
        raise ValueError(f"No event columns found with prefix {args.event_prefix!r}.")

    pred_rows = []
    metric_rows = []
    filter_rows = []
    selected_rows = []
    factor_metadata = []

    for asset in assets:
        for target in targets:
            if target not in df.columns:
                continue
            data = df[df["asset"] == asset].sort_values("date").copy()
            if data.empty:
                continue
            data["year"] = data["date"].dt.year
            data, _ = _clean_price_frame(data, price_features, target)
            for year, train_idx, test_idx in iter_year_folds(
                data,
                test_year_start=args.test_year_start,
                test_year_end=args.test_year_end,
                min_train_rows=args.min_train_rows,
                min_test_rows=args.min_test_rows,
            ):
                train = data.loc[train_idx].copy()
                test = data.loc[test_idx].copy()
                y_train, y_test, threshold = make_high_vol_labels(
                    train[target].values,
                    test[target].values,
                    quantile=args.high_vol_quantile,
                )
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 1:
                    continue

                price_model, price_train_full, price_test = _fit_price_score(train, test, price_features, y_train, args)
                if price_model is None:
                    continue

                n_final_train = len(train)
                n_oof_price_rows = 0
                used_in_sample_price_score_fallback = False

                if method == "price_only":
                    score = price_test
                    X_train_final = train[price_features]
                    X_test_final = test[price_features]
                    final_model = price_model
                elif method == "raw_concat":
                    event_filter = EventColumnFilter(args.event_prefix, args.max_zero_ratio, args.min_nonzero_count)
                    X_event_train = event_filter.fit_transform(train)
                    X_event_test = event_filter.transform(test)
                    filter_rows.append(_event_filter_row(asset, target, year, event_filter))
                    factor_metadata.append({"asset": asset, "target": target, "test_year": int(year), "factors": []})
                    X_train_final = pd.concat([train[price_features], X_event_train], axis=1)
                    X_test_final = pd.concat([test[price_features], X_event_test], axis=1)
                    final_model = make_classifier(args.model_type, args.random_seed, C=0.1)
                    final_model.fit(X_train_final, y_train)
                    score = predict_score(final_model, X_test_final)
                elif method == "grouped_svd":
                    residual_dummy = pd.Series(y_train, index=train.index, dtype=float)
                    fac_train, fac_test, event_filter, selected_df, metadata = _make_event_factors(
                        "svd", train, test, residual_dummy, args, selected_only=False
                    )
                    filter_rows.append(_event_filter_row(asset, target, year, event_filter))
                    factor_metadata.append({"asset": asset, "target": target, "test_year": int(year), "factors": metadata})
                    X_train_final = pd.concat([train[price_features], fac_train], axis=1)
                    X_test_final = pd.concat([test[price_features], fac_test], axis=1)
                    final_model = make_classifier(args.model_type, args.random_seed, C=1.0)
                    final_model.fit(X_train_final, y_train)
                    score = predict_score(final_model, X_test_final)
                else:
                    p_oof, _ = price_oof_predictions_by_year(
                        train,
                        price_features,
                        pd.Series(y_train, index=train.index),
                        min_inner_train_rows=args.min_inner_train_rows,
                        min_inner_valid_rows=args.min_inner_valid_rows,
                        model_type="logistic",
                        random_seed=args.random_seed,
                    )
                    y_train_series = pd.Series(y_train, index=train.index)
                    residual = residual_continuous(y_train_series, p_oof)
                    reducer = "pls" if method in ["residual_grouped_pls", "residual_fusion_interactions"] and args.residual_method == "pls" else "svd"
                    fac_train, fac_test, event_filter, selected_df, metadata = _make_event_factors(
                        reducer, train, test, residual, args, selected_only=True
                    )
                    filter_rows.append(_event_filter_row(asset, target, year, event_filter))
                    if not selected_df.empty:
                        selected_df = selected_df[selected_df["selected"]].copy()
                        selected_df["asset"] = asset
                        selected_df["target"] = target
                        selected_df["test_year"] = int(year)
                        selected_rows.append(selected_df)
                    factor_metadata.append({"asset": asset, "target": target, "test_year": int(year), "reducer": reducer, "factors": metadata})
                    n_oof_price_rows = int(p_oof.notna().sum())
                    used_in_sample_price_score_fallback = bool(args.allow_in_sample_price_score_fallback)
                    if args.allow_in_sample_price_score_fallback:
                        price_train_stack = pd.Series(price_train_full, index=train.index)
                        price_train_stack.loc[p_oof.notna()] = p_oof.loc[p_oof.notna()]
                        stack_mask = pd.Series(True, index=train.index)
                    else:
                        price_train_stack = p_oof.copy()
                        stack_mask = p_oof.notna()

                    n_final_train = int(stack_mask.sum())
                    if n_final_train < args.min_train_rows or y_train_series.loc[stack_mask].nunique() < 2:
                        print(f"Skip fold due to insufficient strict OOF stacking rows: {method} {asset} {target} {year}")
                        continue

                    train_for_stack = train.loc[stack_mask].copy()
                    fac_train_for_stack = fac_train.loc[stack_mask].copy()
                    price_train_stack_final = price_train_stack.loc[stack_mask].values
                    y_train_final = y_train_series.loc[stack_mask].values

                    X_train_final = pd.concat([train_for_stack[price_features], fac_train_for_stack], axis=1)
                    X_test_final = pd.concat([test[price_features], fac_test], axis=1)
                    X_train_final["price_score"] = price_train_stack_final
                    X_test_final["price_score"] = price_test
                    if method == "residual_fusion_interactions":
                        X_train_final, X_test_final = _regime_interactions(
                            X_train_final,
                            X_test_final,
                            train_for_stack,
                            test,
                            price_train_stack_final,
                            price_test,
                            list(fac_train.columns),
                        )
                    final_model = make_classifier(args.model_type, args.random_seed, C=1.0)
                    final_model.fit(X_train_final, y_train_final)
                    score = predict_score(final_model, X_test_final)

                pred_rows.append(_prediction_frame(test, asset, target, year, method, y_test, score, threshold))
                metric_row = _metric_row(asset, target, year, method, args, len(train), y_test, score, threshold)
                metric_row["n_final_train"] = int(n_final_train)
                metric_row["n_oof_price_rows"] = int(n_oof_price_rows)
                metric_row["used_in_sample_price_score_fallback"] = bool(used_in_sample_price_score_fallback)
                metric_rows.append(metric_row)
                print(f"{method}: {asset} {target} {year} n_train={len(train)} n_test={len(test)}")

    preds = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_metrics(metrics)
    preds.to_csv(out_dir / "fold_predictions.csv", index=False)
    metrics.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    if filter_rows:
        pd.DataFrame(filter_rows).to_csv(out_dir / "fold_event_column_filtering.csv", index=False)
    if selected_rows:
        pd.concat(selected_rows, ignore_index=True).to_csv(out_dir / "fold_selected_event_features.csv", index=False)
    if factor_metadata:
        with open(out_dir / "fold_event_factor_metadata.json", "w") as f:
            json.dump(factor_metadata, f, indent=2)
    config = vars(args).copy()
    config["method"] = method
    config["price_features"] = price_features
    config["n_event_columns_input"] = len(event_cols)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    return metrics, preds


def main_for(method, default_output_dir):
    parser = common_parser(method, default_output_dir)
    args = parser.parse_args()
    run_experiment(method, args)
