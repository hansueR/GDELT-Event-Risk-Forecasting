from pathlib import Path
import argparse
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ALL_COLS = [
    "GLOBALEVENTID","SQLDATE","MonthYear","Year","FractionDate",
    "Actor1Code","Actor1Name","Actor1CountryCode","Actor1KnownGroupCode","Actor1EthnicCode",
    "Actor1Religion1Code","Actor1Religion2Code","Actor1Type1Code","Actor1Type2Code","Actor1Type3Code",
    "Actor2Code","Actor2Name","Actor2CountryCode","Actor2KnownGroupCode","Actor2EthnicCode",
    "Actor2Religion1Code","Actor2Religion2Code","Actor2Type1Code","Actor2Type2Code","Actor2Type3Code",
    "IsRootEvent","EventCode","EventBaseCode","EventRootCode","QuadClass",
    "GoldsteinScale","NumMentions","NumSources","NumArticles","AvgTone",
    "Actor1Geo_Type","Actor1Geo_FullName","Actor1Geo_CountryCode","Actor1Geo_ADM1Code","Actor1Geo_ADM2Code",
    "Actor1Geo_Lat","Actor1Geo_Long","Actor1Geo_FeatureID",
    "Actor2Geo_Type","Actor2Geo_FullName","Actor2Geo_CountryCode","Actor2Geo_ADM1Code","Actor2Geo_ADM2Code",
    "Actor2Geo_Lat","Actor2Geo_Long","Actor2Geo_FeatureID",
    "ActionGeo_Type","ActionGeo_FullName","ActionGeo_CountryCode","ActionGeo_ADM1Code","ActionGeo_ADM2Code",
    "ActionGeo_Lat","ActionGeo_Long","ActionGeo_FeatureID",
    "DATEADDED","SOURCEURL",
]


# A smaller but still information-rich version.
# Use this if full mode is too large.
USEFUL_COLS = [
    "GLOBALEVENTID",
    "SQLDATE",
    "IsRootEvent",
    "DATEADDED",
    "SOURCEURL",

    "Actor1Code",
    "Actor1Name",
    "Actor1CountryCode",
    "Actor1KnownGroupCode",
    "Actor1Type1Code",
    "Actor1Type2Code",
    "Actor1Type3Code",

    "Actor2Code",
    "Actor2Name",
    "Actor2CountryCode",
    "Actor2KnownGroupCode",
    "Actor2Type1Code",
    "Actor2Type2Code",
    "Actor2Type3Code",

    "EventCode",
    "EventBaseCode",
    "EventRootCode",
    "QuadClass",

    "GoldsteinScale",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "AvgTone",

    "Actor1Geo_Type",
    "Actor1Geo_FullName",
    "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",

    "Actor2Geo_Type",
    "Actor2Geo_FullName",
    "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",

    "ActionGeo_Type",
    "ActionGeo_FullName",
    "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code",
    "ActionGeo_Lat",
    "ActionGeo_Long",
    "ActionGeo_FeatureID",
]


INT_COLS = [
    "Year",
    "IsRootEvent",
    "QuadClass",
    "NumMentions",
    "NumSources",
    "NumArticles",
    "Actor1Geo_Type",
    "Actor2Geo_Type",
    "ActionGeo_Type",
]

FLOAT_COLS = [
    "FractionDate",
    "GoldsteinScale",
    "AvgTone",
    "Actor1Geo_Lat",
    "Actor1Geo_Long",
    "Actor2Geo_Lat",
    "Actor2Geo_Long",
    "ActionGeo_Lat",
    "ActionGeo_Long",
]


def get_usecols(column_mode: str) -> list[str]:
    if column_mode == "full":
        return ALL_COLS
    if column_mode == "useful":
        return USEFUL_COLS
    raise ValueError(f"Unknown column_mode: {column_mode}")


def read_one_zip(path: Path, usecols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if not names:
            raise ValueError(f"Empty zip file: {path}")
        inner = names[0]

        df = pd.read_csv(
            z.open(inner),
            sep="\t",
            header=None,
            names=ALL_COLS,
            usecols=usecols,
            dtype=str,
            keep_default_na=False,
            na_values=[],
            low_memory=False,
        )

    df["source_file"] = path.name

    # Keep code fields as strings.
    # Convert only clearly numeric fields.
    for col in INT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce").astype("Int64")

    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
                .astype("float64")
            )

    if "SQLDATE" in df.columns:
        df["sql_date"] = pd.to_datetime(
            df["SQLDATE"],
            format="%Y%m%d",
            errors="coerce",
        )

    if "DATEADDED" in df.columns:
        dt_utc = pd.to_datetime(
            df["DATEADDED"],
            format="%Y%m%d%H%M%S",
            errors="coerce",
            utc=True,
        )
        df["date_added_ts_utc"] = dt_utc
        df["date_added_ts_ny"] = dt_utc.dt.tz_convert("America/New_York")
        df["event_day_ny"] = df["date_added_ts_ny"].dt.date.astype("string")

    # Force remaining object columns to string dtype.
    # This makes parquet schema more stable across chunks.
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string")

    return df


def write_table(writer_map, out_path: Path, df: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)

    key = str(out_path)

    if key not in writer_map:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer_map[key] = pq.ParquetWriter(
            out_path,
            table.schema,
            compression="snappy",
        )

    writer_map[key].write_table(table)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--column_mode",
        choices=["full", "useful"],
        default="full",
        help="full keeps all 61 official GDELT fields; useful keeps an expanded practical subset.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    usecols = get_usecols(args.column_mode)

    files = sorted(input_dir.glob("*.export.CSV.zip"))
    if not files:
        raise FileNotFoundError(f"No *.export.CSV.zip files found in {input_dir}")

    print("input_dir:", input_dir)
    print("output_dir:", output_dir)
    print("column_mode:", args.column_mode)
    print("n_usecols:", len(usecols))
    print("n_zip_files:", len(files))

    writer_map = {}
    total_rows = 0
    failed = []

    try:
        for i, path in enumerate(files, start=1):
            try:
                df = read_one_zip(path, usecols=usecols)

                # Preserve the previous clean layout:
                # one parquet file per UTC date in the raw filename.
                # Example: 20160301000000.export.CSV.zip -> 20160301.parquet
                day_key = path.name[:8]
                out_path = output_dir / f"{day_key}.parquet"

                write_table(writer_map, out_path, df)

                total_rows += len(df)

                if i % 50 == 0 or i == len(files):
                    print(f"[{i}/{len(files)}] processed, total_rows={total_rows}")

            except Exception as e:
                print(f"[WARNING] failed: {path} | {repr(e)}")
                failed.append((str(path), repr(e)))

    finally:
        for writer in writer_map.values():
            writer.close()

    print("\nFinished.")
    print("total_rows:", total_rows)
    print("n_output_days:", len(writer_map))
    print("n_failed:", len(failed))

    if failed:
        fail_file = output_dir / "_failed_files.txt"
        with open(fail_file, "w") as f:
            for path, err in failed:
                f.write(f"{path}\t{err}\n")
        print("failed file list:", fail_file)
        raise RuntimeError(f"{len(failed)} files failed during cleaning.")

    if total_rows == 0:
        raise RuntimeError("No rows were written.")


if __name__ == "__main__":
    main()
