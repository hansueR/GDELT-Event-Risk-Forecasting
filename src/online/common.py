import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

from gdelt_risk.event_preprocess import EventColumnFilter
from gdelt_risk.event_reduction import (
    DEFAULT_MAX_FEATURES_PER_FAMILY,
    DEFAULT_PLS_COMPONENTS,
    select_event_features_by_residual,
)
from gdelt_risk.features import DEFAULT_ASSETS, DEFAULT_TARGETS, get_event_columns, get_price_features, map_event_families
from gdelt_risk.models import make_classifier, predict_score


METHODS = ["PO", "RFI"]
TARGET_TO_HORIZON = {
    "target_rv_return_3d": "3d",
    "target_rv_return_5d": "5d",
}
HORIZON_TO_TARGET = {v: k for k, v in TARGET_TO_HORIZON.items()}
DEFAULT_ONLINE_DIR = Path("artifacts/online")


def parse_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def read_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if value is not None and not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def save_joblib(obj, path):
    if joblib is None:
        raise ImportError("joblib is required to save model artifacts.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path):
    if joblib is None:
        raise ImportError("joblib is required to load model artifacts.")
    return joblib.load(path)


def make_regressor(random_seed=42):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.05,
                    l2_regularization=1.0,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def high_vol_labels(values, quantile=0.80):
    values = pd.Series(values, dtype=float)
    threshold = float(values.quantile(quantile))
    return values.ge(threshold).astype(int), threshold


def safe_log_rv(values):
    values = pd.Series(values, dtype=float)
    return np.log(values.clip(lower=1e-12))


def align_to_schema(frame, columns):
    out = pd.DataFrame(index=frame.index)
    for col in columns:
        out[col] = frame[col] if col in frame.columns else 0.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass
class OnlineRFITransformer:
    event_prefix: str = "ddfe_"
    max_zero_ratio: float = 0.99
    min_nonzero_count: int = 30
    random_seed: int = 42
    max_features_per_family: dict = field(default_factory=lambda: dict(DEFAULT_MAX_FEATURES_PER_FAMILY))
    components_by_family: dict = field(default_factory=lambda: dict(DEFAULT_PLS_COMPONENTS))

    def fit(self, train_df, residual):
        self.event_filter_ = EventColumnFilter(
            event_prefix=self.event_prefix,
            max_zero_ratio=self.max_zero_ratio,
            min_nonzero_count=self.min_nonzero_count,
        )
        X_event = self.event_filter_.fit_transform(train_df)
        families = self.event_filter_.families_
        selected, selected_df = select_event_features_by_residual(
            X_event,
            residual,
            families,
            max_features_per_family=self.max_features_per_family,
        )
        self.selected_event_columns_ = selected
        self.selected_feature_table_ = selected_df
        self.selected_families_ = {col: families[col] for col in selected}
        X_selected = X_event[selected] if selected else pd.DataFrame(index=train_df.index)
        self.family_pipelines_ = {}
        self.factor_columns_ = []
        self.factor_metadata_ = []
        residual = pd.Series(residual, index=train_df.index)
        valid = residual.notna()

        for family in sorted(set(self.selected_families_.values())):
            cols = [c for c, fam in self.selected_families_.items() if fam == family and c in X_selected.columns]
            if not cols:
                continue
            if family == "global" or len(cols) == 1:
                pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
                pipe.fit(X_selected[cols])
                n_factors = len(cols)
                method = "scaled_raw"
            else:
                requested = int(self.components_by_family.get(family, 2))
                n_components = min(requested, len(cols), max(1, int(valid.sum()) - 1))
                if n_components < 1 or valid.sum() < 3 or float(residual.loc[valid].std(ddof=0)) == 0:
                    continue
                prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
                Xv = prep.fit_transform(X_selected.loc[valid, cols])
                pls = PLSRegression(n_components=n_components, scale=False)
                pls.fit(Xv, residual.loc[valid].values)
                pipe = Pipeline([("prep", prep), ("pls", pls)])
                n_factors = n_components
                method = "pls_residual"
            names = [f"pls_{family}_{idx + 1}" for idx in range(n_factors)]
            self.family_pipelines_[family] = {"columns": cols, "pipeline": pipe, "names": names, "method": method}
            self.factor_columns_.extend(names)
            self.factor_metadata_.append({"family": family, "method": method, "n_input_features": len(cols), "n_factors": len(names)})
        return self

    def transform(self, frame):
        if not getattr(self, "selected_event_columns_", None):
            return pd.DataFrame(index=frame.index)
        X_event = self.event_filter_.transform(frame)
        parts = []
        for family, spec in self.family_pipelines_.items():
            cols = spec["columns"]
            X = X_event.reindex(columns=cols, fill_value=0.0)
            transformed = spec["pipeline"].transform(X)
            parts.append(pd.DataFrame(transformed, index=frame.index, columns=spec["names"]))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=frame.index)

    def metadata(self):
        return {
            "event_prefix": self.event_prefix,
            "max_zero_ratio": self.max_zero_ratio,
            "min_nonzero_count": self.min_nonzero_count,
            "selected_event_columns": getattr(self, "selected_event_columns_", []),
            "factor_columns": getattr(self, "factor_columns_", []),
            "event_filter": getattr(self.event_filter_, "metadata_", {}),
            "factors": getattr(self, "factor_metadata_", []),
        }


def add_rfi_interactions(X, price_score, factor_columns, high_price_threshold):
    out = X.copy()
    out["price_score"] = price_score
    out["price_uncertainty"] = np.abs(np.asarray(price_score, dtype=float) - 0.5)
    out["high_price_risk"] = (np.asarray(price_score, dtype=float) >= high_price_threshold).astype(int)
    for factor in factor_columns:
        for reg in ["price_score", "price_uncertainty", "high_price_risk"]:
            out[f"{factor}_x_{reg}"] = out[factor].values * out[reg].values
    return out


def model_key(asset, horizon, method):
    return f"{asset}__{horizon}__{method}"


def model_paths(artifact_dir, asset, horizon, method):
    base = Path(artifact_dir) / "models" / asset / horizon / method
    return {
        "dir": base,
        "classifier": base / "alert_model.joblib",
        "regressor": base / "log_rv_model.joblib",
        "transformer": base / "transformer.joblib",
        "price_model": base / "price_model.joblib",
    }


def target_for_horizon(horizon):
    return HORIZON_TO_TARGET[horizon]


def horizon_for_target(target):
    return TARGET_TO_HORIZON[target]


def compact_event_intensity(df):
    event_cols = [c for c in df.columns if c.startswith("ddfe_")]
    if "ddfe_global_event_total" in df.columns:
        return df["ddfe_global_event_total"].fillna(0.0)
    log_cols = [c for c in event_cols if "log_count" in c.lower()]
    cols = log_cols[:50] if log_cols else event_cols[:50]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1)
