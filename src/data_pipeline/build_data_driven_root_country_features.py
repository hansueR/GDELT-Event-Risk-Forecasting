from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd


def safe_name(x: object) -> str:
    s = str(x).strip()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = s.strip("_")
    return s if s else "UNK"


def map_to_next_trading_day(event_dates: pd.Series, trading_calendar: pd.Series) -> pd.Series:
    event_dates = pd.to_datetime(event_dates)
    calendar = pd.to_datetime(trading_calendar).sort_values().drop_duplicates()

    idx = np.searchsorted(calendar.values, event_dates.values, side="left")
    valid = idx < len(calendar)

    mapped = pd.Series(pd.NaT, index=event_dates.index, dtype="datetime64[ns]")
    mapped.loc[event_dates.index[valid]] = calendar.iloc[idx[valid]].values
    return mapped


def past_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    past = s.shift(1)
    mean = past.rolling(window=window, min_periods=min_periods).mean()
    std = past.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (s - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).clip(-5, 5)


def read_event_long(event_long_dir: Path) -> pd.DataFrame:
    files = sorted(event_long_dir.glob("events_market_day_long_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No event_long parquet files found in {event_long_dir}")

    use_cols = [
        "market_day_ny",
        "EventRootCode",
        "ActionGeo_CountryCode",
        "n_events",
        "goldstein_neg_abs_sum",
        "goldstein_pos_sum",
        "goldstein_sum",
        "mentions_sum",
        "articles_sum",
        "tone_mean",
        "conflict_events",
        "coop_events",
        "material_events",
        "verbal_events",
    ]

    parts = []
    for f in files:
        df = pd.read_parquet(f)
        missing = [c for c in ["market_day_ny", "EventRootCode", "ActionGeo_CountryCode", "n_events"] if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing required columns: {missing}")
        keep = [c for c in use_cols if c in df.columns]
        df = df[keep].copy()
        parts.append(df)

    out = pd.concat(parts, ignore_index=True)
    out["market_day_ny"] = pd.to_datetime(out["market_day_ny"])
    out["EventRootCode"] = out["EventRootCode"].fillna("UNK").astype(str).str.zfill(2)
    out["ActionGeo_CountryCode"] = out["ActionGeo_CountryCode"].fillna("UNK").astype(str)
    out["ActionGeo_CountryCode"] = out["ActionGeo_CountryCode"].replace({"": "UNK", "nan": "UNK", "None": "UNK"})

    numeric_cols = [c for c in out.columns if c not in ["market_day_ny", "EventRootCode", "ActionGeo_CountryCode"]]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    return out


def choose_pairs(event_long: pd.DataFrame, min_nonzero_days: int, top_pairs: int) -> pd.DataFrame:
    pair_stats = (
        event_long.groupby(["EventRootCode", "ActionGeo_CountryCode"], dropna=False)
        .agg(
            total_events=("n_events", "sum"),
            nonzero_days=("market_day_ny", "nunique"),
            total_neg_goldstein=("goldstein_neg_abs_sum", "sum"),
            total_mentions=("mentions_sum", "sum"),
            total_articles=("articles_sum", "sum"),
        )
        .reset_index()
    )
    pair_stats["pair"] = (
        pair_stats["EventRootCode"].map(safe_name)
        + "_"
        + pair_stats["ActionGeo_CountryCode"].map(safe_name)
    )

    pair_stats = pair_stats[pair_stats["nonzero_days"] >= min_nonzero_days].copy()
    pair_stats = pair_stats.sort_values(
        ["nonzero_days", "total_events", "total_mentions"], ascending=False
    ).reset_index(drop=True)

    if top_pairs > 0:
        pair_stats = pair_stats.head(top_pairs).copy()

    pair_stats["rank_by_activity"] = np.arange(1, len(pair_stats) + 1)
    return pair_stats


def build_features(
    event_long: pd.DataFrame,
    trading_calendar: pd.Series,
    pair_stats: pd.DataFrame,
    windows: list[int],
    z_threshold: float,
) -> pd.DataFrame:
    event_long = event_long.copy()
    event_long["date"] = map_to_next_trading_day(event_long["market_day_ny"], trading_calendar)
    event_long = event_long.dropna(subset=["date"]).copy()
    event_long["date"] = pd.to_datetime(event_long["date"])

    calendar = pd.to_datetime(trading_calendar).sort_values().drop_duplicates()

    event_long["pair"] = (
        event_long["EventRootCode"].map(safe_name)
        + "_"
        + event_long["ActionGeo_CountryCode"].map(safe_name)
    )
    selected_pairs = set(pair_stats["pair"].tolist())
    sub = event_long[event_long["pair"].isin(selected_pairs)].copy()

    total_events = (
        event_long.groupby("date")["n_events"]
        .sum()
        .reindex(calendar, fill_value=0.0)
        .astype(float)
    )
    feature_data = {
        "date": calendar.values,
        "ddfe_global_event_total": total_events.values,
    }

    if sub.empty:
        raise ValueError("No event rows remain after pair filtering. Lower min_nonzero_days or top_pairs.")

    count_wide = (
        sub.pivot_table(index="date", columns="pair", values="n_events", aggfunc="sum", fill_value=0.0)
        .reindex(calendar, fill_value=0.0)
        .sort_index()
    )

    neg_goldstein_wide = (
        sub.pivot_table(index="date", columns="pair", values="goldstein_neg_abs_sum", aggfunc="sum", fill_value=0.0)
        .reindex(calendar, fill_value=0.0)
        .sort_index()
    )

    for pair in count_wide.columns:
        prefix = f"ddfe_rc_{pair}"
        count = count_wide[pair].astype(float)
        neg_goldstein = neg_goldstein_wide[pair].astype(float)
        share = (count / total_events.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        feature_data[f"{prefix}_log_count"] = np.log1p(count.values)
        feature_data[f"{prefix}_share"] = share.values
        feature_data[f"{prefix}_neg_goldstein_log"] = np.log1p(neg_goldstein.values)

        for window in windows:
            min_periods = max(5, window // 2)
            z = past_zscore(share, window=window, min_periods=min_periods)
            z_col = f"{prefix}_share_z{window}d"
            shock_col = f"{prefix}_share_shock{window}d"
            feature_data[z_col] = z.fillna(0.0).values
            feature_data[shock_col] = (z > z_threshold).astype("int8").values

    feature_daily = pd.DataFrame(feature_data)
    ddfe_cols = [c for c in feature_daily.columns if c.startswith("ddfe_")]
    feature_daily[ddfe_cols] = feature_daily[ddfe_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feature_daily


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_long_dir", default="/project/hrao/GDELT/features/event_long_clean_full")
    parser.add_argument("--model_table", default="/project/hrao/GDELT/modeling/model_table_clean_full.parquet")
    parser.add_argument("--output_file", default="/project/hrao/GDELT/modeling/model_table_ddfe_root_country_clean_full.parquet")
    parser.add_argument("--output_dir", default="/project/hrao/GDELT/results/ddfe_root_country_clean_full_v1")
    parser.add_argument("--min_nonzero_days", type=int, default=30)
    parser.add_argument("--top_pairs", type=int, default=400, help="0 means keep all pairs after min_nonzero_days filtering")
    parser.add_argument("--windows", type=int, nargs="+", default=[20, 60])
    parser.add_argument("--z_threshold", type=float, default=2.0)
    args = parser.parse_args()

    event_long_dir = Path(args.event_long_dir)
    model_table_path = Path(args.model_table)
    output_file = Path(args.output_file)
    output_dir = Path(args.output_dir)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading model table:", model_table_path)
    model = pd.read_parquet(model_table_path)
    model["date"] = pd.to_datetime(model["date"])
    trading_calendar = model["date"].sort_values().drop_duplicates()

    print("Reading event_long:", event_long_dir)
    event_long = read_event_long(event_long_dir)
    print("event_long shape:", event_long.shape)

    print("Selecting root-country pairs...")
    pair_stats = choose_pairs(
        event_long=event_long,
        min_nonzero_days=args.min_nonzero_days,
        top_pairs=args.top_pairs,
    )
    if pair_stats.empty:
        raise ValueError("No candidate pairs selected. Lower --min_nonzero_days.")

    pair_file = output_dir / "root_country_pair_summary.csv"
    pair_stats.to_csv(pair_file, index=False)
    print("selected pairs:", len(pair_stats))
    print(pair_stats.head(30).to_string(index=False))

    print("Building data-driven event features...")
    feature_daily = build_features(
        event_long=event_long,
        trading_calendar=trading_calendar,
        pair_stats=pair_stats,
        windows=args.windows,
        z_threshold=args.z_threshold,
    )

    ddfe_cols = [c for c in feature_daily.columns if c.startswith("ddfe_")]
    print("feature_daily shape:", feature_daily.shape)
    print("n_ddfe_cols:", len(ddfe_cols))

    merged = model.merge(feature_daily, on="date", how="left")
    ddfe_cols_merged = [c for c in merged.columns if c.startswith("ddfe_")]
    merged[ddfe_cols_merged] = merged[ddfe_cols_merged].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged.to_parquet(output_file, index=False)

    config = {
        "event_long_dir": str(event_long_dir),
        "model_table": str(model_table_path),
        "output_file": str(output_file),
        "min_nonzero_days": args.min_nonzero_days,
        "top_pairs": args.top_pairs,
        "windows": args.windows,
        "z_threshold": args.z_threshold,
        "n_selected_pairs": int(len(pair_stats)),
        "n_ddfe_cols": int(len(ddfe_cols_merged)),
    }
    config_file = output_dir / "ddfe_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    print("\nSaved:")
    print(output_file)
    print(pair_file)
    print(config_file)
    print("merged shape:", merged.shape)


if __name__ == "__main__":
    main()
