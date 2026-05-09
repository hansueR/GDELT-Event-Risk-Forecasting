import argparse
from pathlib import Path

import pandas as pd


def write_table(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in [".parquet", ".pq"]:
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def main():
    parser = argparse.ArgumentParser(description="Export date-level DDFe event features for online inference.")
    parser.add_argument("--model_table", required=True)
    parser.add_argument("--output_file", default="data/online/daily_event_features.parquet")
    parser.add_argument("--event_prefix", default="ddfe_")
    args = parser.parse_args()

    model_table = Path(args.model_table)
    if not model_table.exists():
        raise FileNotFoundError(f"Missing model table: {model_table}")

    df = pd.read_parquet(model_table)
    if "date" not in df.columns:
        raise ValueError(f"Model table must contain date column: {model_table}")

    event_cols = [col for col in df.columns if col.startswith(args.event_prefix)]
    if not event_cols:
        raise ValueError(f"No {args.event_prefix} event feature columns found in {model_table}")

    out = df[["date", *event_cols]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.groupby("date", as_index=False)[event_cols].max()
    out = out.sort_values("date").reset_index(drop=True)
    write_table(out, args.output_file)

    print(f"Exported {len(out)} daily rows and {len(event_cols)} event columns to {args.output_file}")


if __name__ == "__main__":
    main()
