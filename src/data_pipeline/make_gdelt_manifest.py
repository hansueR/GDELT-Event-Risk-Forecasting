import argparse
from pathlib import Path

import pandas as pd


GDELT_V2_BASE_URL = "http://data.gdeltproject.org/gdeltv2"


def main():
    parser = argparse.ArgumentParser(description="Generate GDELT 2.0 Events export.CSV.zip download manifests.")
    parser.add_argument("--start", required=True, help="Inclusive start timestamp/date, e.g. 2026-05-01.")
    parser.add_argument("--end", required=True, help="Exclusive end timestamp/date, e.g. 2026-05-10.")
    parser.add_argument("--output_dir", default="manifests/by_month")
    parser.add_argument("--freq", default="15min")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamps = pd.date_range(start=start, end=end, freq=args.freq, inclusive="left")
    if stamps.empty:
        raise ValueError("No timestamps generated for the requested range.")

    by_month = {}
    for stamp in stamps:
        month = stamp.strftime("%Y%m")
        url = f"{GDELT_V2_BASE_URL}/{stamp.strftime('%Y%m%d%H%M%S')}.export.CSV.zip"
        by_month.setdefault(month, []).append(url)

    month_list = []
    for month, urls in sorted(by_month.items()):
        path = output_dir / f"{month}.txt"
        path.write_text("\n".join(urls) + "\n", encoding="utf-8")
        month_list.append(month)
        print(f"Wrote {len(urls)} URLs to {path}")

    month_file = output_dir.parent / "months_generated.txt"
    month_file.write_text("\n".join(month_list) + "\n", encoding="utf-8")
    print(f"Wrote month list to {month_file}")


if __name__ == "__main__":
    main()
