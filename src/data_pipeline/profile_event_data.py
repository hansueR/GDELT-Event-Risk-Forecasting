from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def classify_event_family(col: str) -> str:
    s = col.lower()

    if "shock" in s:
        return "shock_binary"
    if "_z" in s or "zscore" in s or "z_" in s:
        return "zscore_anomaly"
    if "share" in s:
        return "share"
    if "log" in s:
        return "log_intensity"
    if "count" in s or "n_events" in s or "value" in s:
        return "raw_intensity"
    if "goldstein" in s:
        return "goldstein"
    if "tone" in s:
        return "tone"
    return "other_event"


def detect_event_cols(columns, prefixes):
    out = []
    for c in columns:
        if any(c.startswith(p) for p in prefixes):
            out.append(c)
    return out


def detect_target_cols(columns):
    return [
        c for c in columns
        if c.startswith("target_")
        or c.startswith("future_")
        or c in {"label", "y"}
    ]


def detect_price_cols(columns, event_cols, target_cols):
    exclude = set(event_cols) | set(target_cols)
    exclude |= {"date", "asset", "ticker", "symbol"}

    price_keywords = [
        "return",
        "log_return",
        "abs_return",
        "squared_return",
        "rv",
        "vol",
        "parkinson",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "ma_",
        "momentum",
        "drawdown",
    ]

    out = []
    for c in columns:
        if c in exclude:
            continue
        lc = c.lower()
        if any(k in lc for k in price_keywords):
            out.append(c)
    return out


def profile_numeric_block(df, cols, zero_tol=1e-12):
    if len(cols) == 0:
        return pd.DataFrame()

    x = df[cols].apply(pd.to_numeric, errors="coerce")
    filled = x.fillna(0.0)

    zero_mask = filled.abs() <= zero_tol
    missing_ratio = x.isna().mean()
    zero_ratio = zero_mask.mean()
    nonzero_count = (~zero_mask).sum()
    nonzero_ratio = 1.0 - zero_ratio

    stats = pd.DataFrame({
        "column": cols,
        "family": [classify_event_family(c) for c in cols],
        "missing_ratio": missing_ratio.values,
        "zero_ratio": zero_ratio.values,
        "nonzero_ratio": nonzero_ratio.values,
        "nonzero_count": nonzero_count.values,
        "mean": x.mean().values,
        "std": x.std(ddof=0).values,
        "min": x.min().values,
        "p50": x.quantile(0.50).values,
        "p95": x.quantile(0.95).values,
        "p99": x.quantile(0.99).values,
        "max": x.max().values,
        "n_unique": x.nunique(dropna=True).values,
    })

    stats["is_constant"] = stats["n_unique"] <= 1
    stats["is_very_sparse_95"] = stats["zero_ratio"] >= 0.95
    stats["is_very_sparse_99"] = stats["zero_ratio"] >= 0.99

    return stats.sort_values(
        ["zero_ratio", "nonzero_count"],
        ascending=[False, True],
    ).reset_index(drop=True)


def compute_group_stats(col_stats):
    if col_stats.empty:
        return pd.DataFrame()

    g = (
        col_stats
        .groupby("family")
        .agg(
            n_cols=("column", "count"),
            mean_zero_ratio=("zero_ratio", "mean"),
            median_zero_ratio=("zero_ratio", "median"),
            mean_nonzero_count=("nonzero_count", "mean"),
            median_nonzero_count=("nonzero_count", "median"),
            n_very_sparse_95=("is_very_sparse_95", "sum"),
            n_very_sparse_99=("is_very_sparse_99", "sum"),
            n_constant=("is_constant", "sum"),
        )
        .reset_index()
        .sort_values("mean_zero_ratio", ascending=False)
    )
    return g


def compute_asset_sparsity(df, event_cols, zero_tol=1e-12):
    if "asset" not in df.columns or len(event_cols) == 0:
        return pd.DataFrame()

    rows = []
    for asset, sub in df.groupby("asset"):
        x = sub[event_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        zero_ratio = (x.abs() <= zero_tol).to_numpy().mean()
        rows.append({
            "asset": asset,
            "n_rows": len(sub),
            "n_event_cols": len(event_cols),
            "overall_zero_ratio": zero_ratio,
            "overall_density": 1.0 - zero_ratio,
        })
    return pd.DataFrame(rows).sort_values("asset")


def compute_target_correlations(df, event_cols, target_cols, output_dir):
    if len(event_cols) == 0 or len(target_cols) == 0:
        return

    x = df[event_cols].apply(pd.to_numeric, errors="coerce")
    rows = []

    for target in target_cols:
        y = pd.to_numeric(df[target], errors="coerce")
        if y.notna().sum() < 50:
            continue

        # Pearson is fast. Spearman is more robust but slower.
        corr = x.corrwith(y, method="pearson")
        tmp = pd.DataFrame({
            "target": target,
            "column": corr.index,
            "pearson_corr": corr.values,
        })
        tmp["abs_pearson_corr"] = tmp["pearson_corr"].abs()
        tmp["family"] = tmp["column"].map(classify_event_family)
        rows.append(tmp)

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out = out.sort_values(
            ["target", "abs_pearson_corr"],
            ascending=[True, False],
        )
        out.to_csv(output_dir / "event_target_correlations.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/project/hrao/GDELT/modeling/model_table_ddfe_root_country_clean_full.parquet",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/hrao/GDELT/results/data_profile_ddfe_root_country_clean_full_v1",
    )
    parser.add_argument(
        "--event_prefixes",
        default="ddfe_",
        help="Comma-separated prefixes for event columns. Use ddfe_ for the active root-country table.",
    )
    parser.add_argument(
        "--zero_tol",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--max_print_cols",
        type=int,
        default=40,
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefixes = [p.strip() for p in args.event_prefixes.split(",") if p.strip()]

    print(f"Reading: {input_path}")
    df = pd.read_parquet(input_path)

    columns = list(df.columns)
    event_cols = detect_event_cols(columns, prefixes)
    target_cols = detect_target_cols(columns)
    price_cols = detect_price_cols(columns, event_cols, target_cols)

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    non_numeric_cols = [c for c in columns if c not in numeric_cols]

    event_stats = profile_numeric_block(df, event_cols, zero_tol=args.zero_tol)
    group_stats = compute_group_stats(event_stats)
    asset_sparsity = compute_asset_sparsity(df, event_cols, zero_tol=args.zero_tol)

    event_stats.to_csv(output_dir / "event_feature_column_stats.csv", index=False)
    group_stats.to_csv(output_dir / "event_feature_family_stats.csv", index=False)
    asset_sparsity.to_csv(output_dir / "event_asset_sparsity.csv", index=False)

    compute_target_correlations(df, event_cols, target_cols, output_dir)

    summary_lines = []
    add = summary_lines.append

    add("=" * 100)
    add("DATA PROFILE SUMMARY")
    add("=" * 100)
    add(f"Input file: {input_path}")
    add(f"Rows: {df.shape[0]:,}")
    add(f"Columns: {df.shape[1]:,}")
    add(f"Memory usage MB: {df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    add("")

    add("BASIC FORMAT")
    add("-" * 100)
    add(f"Numeric columns: {len(numeric_cols):,}")
    add(f"Non-numeric columns: {len(non_numeric_cols):,}")
    add(f"Non-numeric column names: {non_numeric_cols[:args.max_print_cols]}")
    add("")

    if "date" in df.columns:
        date_s = pd.to_datetime(df["date"], errors="coerce")
        add(f"Date range: {date_s.min()}  ->  {date_s.max()}")
        add(f"Number of unique dates: {date_s.nunique():,}")

    if "asset" in df.columns:
        add(f"Assets: {sorted(df['asset'].dropna().unique().tolist())}")
        add("Rows by asset:")
        add(df["asset"].value_counts().sort_index().to_string())
    add("")

    add("COLUMN GROUPS")
    add("-" * 100)
    add(f"Detected event prefixes: {prefixes}")
    add(f"Event columns: {len(event_cols):,}")
    add(f"Price-like columns: {len(price_cols):,}")
    add(f"Target columns: {len(target_cols):,}")
    add("")

    add("First event columns:")
    add("\n".join(event_cols[:args.max_print_cols]))
    add("")
    add("First price-like columns:")
    add("\n".join(price_cols[:args.max_print_cols]))
    add("")
    add("Target columns:")
    add("\n".join(target_cols[:args.max_print_cols]))
    add("")

    if len(event_cols) > 0:
        x = df[event_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        overall_zero_ratio = (x.abs() <= args.zero_tol).to_numpy().mean()
        overall_missing_ratio = df[event_cols].isna().to_numpy().mean()
        density = 1.0 - overall_zero_ratio

        add("EVENT SPARSITY")
        add("-" * 100)
        add(f"Overall event zero ratio: {overall_zero_ratio:.6f}")
        add(f"Overall event density: {density:.6f}")
        add(f"Overall event missing ratio: {overall_missing_ratio:.6f}")
        add(f"Event dimension / rows ratio: {len(event_cols) / max(len(df), 1):.4f}")
        add(f"Very sparse event columns, zero_ratio >= 0.95: {(event_stats['zero_ratio'] >= 0.95).sum():,}")
        add(f"Very sparse event columns, zero_ratio >= 0.99: {(event_stats['zero_ratio'] >= 0.99).sum():,}")
        add(f"Constant event columns: {(event_stats['n_unique'] <= 1).sum():,}")
        add("")

        add("EVENT FAMILY STATS")
        add("-" * 100)
        add(group_stats.to_string(index=False))
        add("")

        add("ASSET-LEVEL EVENT SPARSITY")
        add("-" * 100)
        if not asset_sparsity.empty:
            add(asset_sparsity.to_string(index=False))
        else:
            add("(No asset column found.)")
        add("")

        add("TOP 30 SPARSEST EVENT COLUMNS")
        add("-" * 100)
        sparse_cols = event_stats.sort_values(
            ["zero_ratio", "nonzero_count"],
            ascending=[False, True],
        ).head(30)
        add(sparse_cols[
            ["column", "family", "zero_ratio", "nonzero_count", "n_unique", "max"]
        ].to_string(index=False))
        add("")

        add("TOP 30 DENSEST EVENT COLUMNS")
        add("-" * 100)
        dense_cols = event_stats.sort_values(
            ["zero_ratio", "nonzero_count"],
            ascending=[True, False],
        ).head(30)
        add(dense_cols[
            ["column", "family", "zero_ratio", "nonzero_count", "n_unique", "mean", "std"]
        ].to_string(index=False))
        add("")

    add("OUTPUT FILES")
    add("-" * 100)
    add(str(output_dir / "profile_summary.txt"))
    add(str(output_dir / "event_feature_column_stats.csv"))
    add(str(output_dir / "event_feature_family_stats.csv"))
    add(str(output_dir / "event_asset_sparsity.csv"))
    add(str(output_dir / "event_target_correlations.csv"))

    summary = "\n".join(summary_lines)
    (output_dir / "profile_summary.txt").write_text(summary)

    print(summary)
    print("\nSaved profile files to:", output_dir)


if __name__ == "__main__":
    main()
