from pathlib import Path
import os


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    env_root = os.environ.get("GDELT_DATA_ROOT")
    if env_root:
        return Path(env_root)

    repo = repo_root()
    candidates = [
        repo,
        repo.parent,
        Path("/project/hrao/GDELT"),
    ]
    for candidate in candidates:
        if (candidate / "modeling").exists() or (candidate / "features").exists() or (candidate / "asset_prices").exists():
            return candidate
    return repo


def default_model_table() -> Path:
    return data_root() / "modeling" / "model_table_ddfe_root_country_clean_full.parquet"


def default_results_dir() -> Path:
    return data_root() / "results"


def default_event_long_dir() -> Path:
    return data_root() / "features" / "event_long_clean_full"


def default_online_event_features() -> Path:
    return data_root() / "data" / "online" / "daily_event_features.parquet"
