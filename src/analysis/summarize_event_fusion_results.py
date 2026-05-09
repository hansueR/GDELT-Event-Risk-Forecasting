import argparse
from pathlib import Path

import pandas as pd


METHOD_DIRS = {
    "price_only": "ddfe_price_only_v1",
    "raw_concat": "ddfe_raw_concat_v1",
    "grouped_svd": "ddfe_grouped_svd_v1",
    "residual_grouped_svd": "ddfe_residual_grouped_svd_v1",
    "residual_grouped_pls": "ddfe_residual_grouped_pls_v1",
    "residual_fusion_interactions": "ddfe_residual_fusion_interactions_v1",
}

METRICS = ["auc", "average_precision", "precision_at_top20pct"]


def read_fold_metrics(results_root):
    frames = []
    for method, dirname in METHOD_DIRS.items():
        path = results_root / dirname / "fold_metrics.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["method"] = method
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_selected_features(results_root):
    frames = []
    for method, dirname in METHOD_DIRS.items():
        path = results_root / dirname / "fold_selected_event_features.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["method"] = method
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="/project/hrao/GDELT/results")
    parser.add_argument("--output_dir", default="/project/hrao/GDELT/results/ddfe_event_fusion_summary_v1")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = read_fold_metrics(results_root)
    if metrics.empty:
        raise FileNotFoundError("No new DDFe fold_metrics.csv files found.")

    summary = (
        metrics.groupby(["method", "asset", "target"], dropna=False)[METRICS + ["n_test", "positive_rate"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    summary.to_csv(out_dir / "all_methods_summary.csv", index=False)

    price = metrics[metrics["method"] == "price_only"][["asset", "target", "test_year"] + METRICS].rename(
        columns={m: f"price_only_{m}" for m in METRICS}
    )
    delta = metrics.merge(price, on=["asset", "target", "test_year"], how="left")
    for m in METRICS:
        delta[f"delta_{m}"] = delta[m] - delta[f"price_only_{m}"]
    delta.to_csv(out_dir / "delta_vs_price_only.csv", index=False)

    best_rows = []
    for metric in METRICS:
        idx = summary.groupby(["asset", "target"])[metric].idxmax()
        best = summary.loc[idx, ["asset", "target", "method", metric]].copy()
        best["metric"] = metric
        best = best.rename(columns={metric: "value"})
        best_rows.append(best)
    pd.concat(best_rows, ignore_index=True).to_csv(out_dir / "best_by_asset_target.csv", index=False)

    selected = read_selected_features(results_root)
    if not selected.empty:
        freq = (
            selected.groupby(["method", "family", "event_column"], dropna=False)
            .size()
            .reset_index(name="selection_count")
            .sort_values(["selection_count", "method", "family"], ascending=[False, True, True])
        )
        freq.to_csv(out_dir / "selected_event_feature_frequency.csv", index=False)
    else:
        pd.DataFrame(columns=["method", "family", "event_column", "selection_count"]).to_csv(
            out_dir / "selected_event_feature_frequency.csv", index=False
        )

    report_lines = [
        "DDFe event fusion summary",
        "",
        f"Methods found: {', '.join(sorted(metrics['method'].unique()))}",
        f"Fold rows: {len(metrics)}",
        "",
        "Mean metrics by method:",
        summary.groupby("method")[METRICS].mean(numeric_only=True).round(4).to_string(),
        "",
        "Mean delta versus price-only:",
        delta[delta["method"] != "price_only"].groupby("method")[[f"delta_{m}" for m in METRICS]].mean(numeric_only=True).round(4).to_string(),
    ]
    (out_dir / "summary_report.txt").write_text("\n".join(report_lines))


if __name__ == "__main__":
    main()
