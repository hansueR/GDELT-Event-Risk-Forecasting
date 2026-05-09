RAW_PRICE_FEATURE_CANDIDATES = [
    "adj_close",
    "close",
    "high",
    "low",
    "open",
    "volume",
]

FORBIDDEN_PRICE_INPUTS = {
    "log_return_1d",
    "abs_log_return_1d",
    "squared_log_return_1d",
    "parkinson_vol_1d",
    "log_close",
}

FORBIDDEN_PRICE_PREFIXES = (
    "hist_",
    "target_",
)

DEFAULT_TARGETS = ["target_rv_return_3d", "target_rv_return_5d"]
DEFAULT_ASSETS = ["Gold", "QQQ", "WTI_Oil"]


def validate_raw_price_features(price_features):
    bad = [
        c
        for c in price_features
        if c in FORBIDDEN_PRICE_INPUTS or any(c.startswith(p) for p in FORBIDDEN_PRICE_PREFIXES)
    ]
    if bad:
        raise ValueError(f"Non-raw price features were selected: {bad}")


def get_price_features(df):
    """Return raw OHLCV price-side input columns only.

    These are model input features. They should not include engineered
    returns, realized volatility, rolling statistics, or target columns.
    """
    price_features = [c for c in RAW_PRICE_FEATURE_CANDIDATES if c in df.columns]
    validate_raw_price_features(price_features)
    return price_features


def get_event_columns(df, event_prefix="ddfe_"):
    """Return active DDFe event columns only."""
    return [c for c in df.columns if c.startswith(event_prefix)]


def event_family(col):
    name = col.lower()
    if name == "ddfe_global_event_total" or name.endswith("_global_event_total"):
        return "global"
    if "shock" in name:
        return "shock_binary"
    if "_z20d" in name or "_z60d" in name or "_zscore" in name or "_z_" in name:
        return "zscore_anomaly"
    if "neg_goldstein_log" in name:
        return "neg_goldstein_log"
    if "log_count" in name:
        return "log_count"
    if "_share" in name and "shock" not in name and "_z" not in name:
        return "share"
    return "other_event"


def map_event_families(cols):
    return {c: event_family(c) for c in cols}


def parse_ddfe_root_country(col):
    """Best-effort parser for names like ddfe_rc_01_US_log_count."""
    parts = col.split("_")
    if len(parts) >= 5 and parts[0] == "ddfe" and parts[1] == "rc":
        return parts[2], parts[3]
    return None, None
