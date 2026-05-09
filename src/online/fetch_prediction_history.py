import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch_json(url, token=""):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Restore prior prediction history before appending today's forecasts.")
    parser.add_argument("--history_url", default="", help="URL to an existing prediction_history.json.")
    parser.add_argument("--output_file", default="dashboard/data/prediction_history.json")
    parser.add_argument("--required", action="store_true", help="Fail if the history URL cannot be loaded.")
    args = parser.parse_args()

    history_url = args.history_url or os.environ.get("PREDICTION_HISTORY_URL", "")
    if not history_url:
        print("No prediction history URL configured; starting a new history file.")
        return

    token = os.environ.get("ONLINE_DATA_TOKEN", "")
    try:
        payload = fetch_json(history_url, token)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        message = f"Could not fetch prediction history from {history_url}: {exc}"
        if args.required:
            raise SystemExit(message) from exc
        print(f"WARNING: {message}")
        return

    predictions = payload.get("predictions", [])
    if not isinstance(predictions, list):
        message = f"Prediction history payload at {history_url} does not contain a predictions list."
        if args.required:
            raise SystemExit(message)
        print(f"WARNING: {message}")
        return

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Restored {len(predictions)} historical prediction rows to {output_file}")


if __name__ == "__main__":
    main()
