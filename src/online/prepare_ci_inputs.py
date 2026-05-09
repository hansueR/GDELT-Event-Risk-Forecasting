import argparse
import os
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


def env_or_arg(value, env_name):
    return value if value else os.environ.get(env_name, "")


def download(url, output_path, token=""):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    print(f"Downloading {url} -> {output_path}")
    with urlopen(request, timeout=120) as response, output_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return output_path


def assert_safe_member(output_dir, member_name):
    target = (output_dir / member_name).resolve()
    try:
        target.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe archive member path: {member_name}") from exc


def extract_bundle(path, output_dir):
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes).lower()
    print(f"Extracting {path} -> {output_dir}")
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            for member in zf.namelist():
                assert_safe_member(output_dir, member)
            zf.extractall(output_dir)
    elif suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz") or suffixes.endswith(".tar"):
        with tarfile.open(path) as tf:
            for member in tf.getmembers():
                assert_safe_member(output_dir, member.name)
            tf.extractall(output_dir)
    else:
        raise ValueError(f"Unsupported bundle format: {path}")


def require_paths(paths):
    missing = [str(path) for path in paths if path and not Path(path).exists()]
    if missing:
        raise SystemExit("Missing required CI input paths:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser(description="Prepare real data inputs for online dashboard CI jobs.")
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--data_bundle_url", default="", help="Optional zip/tar bundle containing data/modeling/results inputs.")
    parser.add_argument("--model_table_url", default="", help="Optional direct URL for the historical model table parquet.")
    parser.add_argument("--event_features_url", default="", help="Optional direct URL for daily event feature parquet/csv.")
    parser.add_argument("--model_table", default="modeling/model_table_ddfe_root_country_clean_full.parquet")
    parser.add_argument("--event_features_file", default="data/online/daily_event_features.parquet")
    parser.add_argument("--require_model_table", action="store_true")
    parser.add_argument("--require_path", action="append", default=[], help="Additional path that must exist after preparation.")
    args = parser.parse_args()

    token = os.environ.get("ONLINE_DATA_TOKEN", "")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_bundle_url = env_or_arg(args.data_bundle_url, "ONLINE_DATA_BUNDLE_URL")
    model_table_url = env_or_arg(args.model_table_url, "ONLINE_MODEL_TABLE_URL")
    event_features_url = env_or_arg(args.event_features_url, "ONLINE_EVENT_FEATURES_URL")

    if data_bundle_url:
        bundle_name = Path(data_bundle_url.split("?", 1)[0]).name or "online-data-bundle.zip"
        bundle_path = download(data_bundle_url, output_dir / ".ci_inputs" / bundle_name, token)
        extract_bundle(bundle_path, output_dir)

    if model_table_url:
        download(model_table_url, args.model_table, token)

    if event_features_url:
        download(event_features_url, args.event_features_file, token)

    required = list(args.require_path)
    if args.require_model_table:
        required.append(args.model_table)
    require_paths(required)
    print("CI inputs are ready.")


if __name__ == "__main__":
    main()
