import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SVD_COMPONENTS = {
    "log_count": 8,
    "share": 8,
    "neg_goldstein_log": 8,
    "zscore_anomaly": 8,
    "shock_binary": 4,
    "other_event": 4,
}

DEFAULT_PLS_COMPONENTS = {
    "log_count": 4,
    "share": 4,
    "neg_goldstein_log": 4,
    "zscore_anomaly": 4,
    "shock_binary": 2,
    "other_event": 2,
}

DEFAULT_MAX_FEATURES_PER_FAMILY = {
    "log_count": 100,
    "share": 100,
    "neg_goldstein_log": 100,
    "zscore_anomaly": 100,
    "shock_binary": 50,
    "other_event": 20,
    "global": 5,
}


def select_event_features_by_residual(
    X_event_train,
    residual_cont,
    event_families,
    max_features_per_family=None,
):
    max_features_per_family = max_features_per_family or DEFAULT_MAX_FEATURES_PER_FAMILY
    residual = pd.Series(residual_cont, index=X_event_train.index)
    valid = residual.notna()
    rows = []
    selected = []
    for family in sorted(set(event_families.values())):
        cols = [c for c, fam in event_families.items() if fam == family and c in X_event_train.columns]
        if not cols:
            continue
        scores = []
        y = residual.loc[valid].astype(float)
        y_std = float(y.std(ddof=0))
        for col in cols:
            x = X_event_train.loc[valid, col].astype(float)
            if len(x) < 3 or y_std == 0 or float(x.std(ddof=0)) == 0:
                score = 0.0
            else:
                score = abs(float(np.corrcoef(x.values, y.values)[0, 1]))
                if not np.isfinite(score):
                    score = 0.0
            scores.append((col, score))
        scores.sort(key=lambda t: (-t[1], t[0]))
        keep_n = int(max_features_per_family.get(family, 20))
        for rank, (col, score) in enumerate(scores, start=1):
            keep = rank <= keep_n
            rows.append({"event_column": col, "family": family, "residual_score": score, "rank": rank, "selected": keep})
            if keep:
                selected.append(col)
    return selected, pd.DataFrame(rows)


def _family_columns(columns, families):
    out = {}
    for col in columns:
        out.setdefault(families.get(col, "other_event"), []).append(col)
    return out


def fit_transform_grouped_svd(X_train, X_test, families, components_by_family=None, random_seed=42, prefix="svd"):
    components_by_family = components_by_family or DEFAULT_SVD_COMPONENTS
    train_parts = []
    test_parts = []
    names = []
    metadata = []
    for family, cols in sorted(_family_columns(X_train.columns, families).items()):
        if not cols:
            continue
        if family == "global" or len(cols) == 1:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            tr = pipe.fit_transform(X_train[cols])
            te = pipe.transform(X_test[cols])
            fam_names = [f"{prefix}_{family}_{i+1}" for i in range(tr.shape[1])]
            method = "scaled_raw"
        else:
            requested = int(components_by_family.get(family, 4))
            n_components = min(requested, max(1, len(cols) - 1), X_train.shape[0] - 1)
            if n_components < 1:
                continue
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("svd", TruncatedSVD(n_components=n_components, random_state=random_seed)),
                ]
            )
            tr = pipe.fit_transform(X_train[cols])
            te = pipe.transform(X_test[cols])
            fam_names = [f"{prefix}_{family}_{i+1}" for i in range(tr.shape[1])]
            method = "truncated_svd"
        train_parts.append(pd.DataFrame(tr, index=X_train.index, columns=fam_names))
        test_parts.append(pd.DataFrame(te, index=X_test.index, columns=fam_names))
        names.extend(fam_names)
        metadata.append({"family": family, "method": method, "n_input_features": len(cols), "n_factors": len(fam_names)})
    train = pd.concat(train_parts, axis=1) if train_parts else pd.DataFrame(index=X_train.index)
    test = pd.concat(test_parts, axis=1) if test_parts else pd.DataFrame(index=X_test.index)
    return train, test, names, metadata


def fit_transform_grouped_pls(X_train, X_test, residual_cont, families, components_by_family=None, prefix="pls"):
    components_by_family = components_by_family or DEFAULT_PLS_COMPONENTS
    train_parts = []
    test_parts = []
    names = []
    metadata = []
    residual = pd.Series(residual_cont, index=X_train.index)
    valid = residual.notna()
    for family, cols in sorted(_family_columns(X_train.columns, families).items()):
        if family == "global" or len(cols) == 1:
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            tr = pipe.fit_transform(X_train[cols])
            te = pipe.transform(X_test[cols])
            fam_names = [f"{prefix}_{family}_{i+1}" for i in range(tr.shape[1])]
            method = "scaled_raw"
        else:
            requested = int(components_by_family.get(family, 2))
            n_components = min(requested, len(cols), max(1, int(valid.sum()) - 1))
            if n_components < 1 or valid.sum() < 3 or float(residual.loc[valid].std(ddof=0)) == 0:
                continue
            prep = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
            Xv = prep.fit_transform(X_train.loc[valid, cols])
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(Xv, residual.loc[valid].values)
            tr = pls.transform(prep.transform(X_train[cols]))
            te = pls.transform(prep.transform(X_test[cols]))
            fam_names = [f"{prefix}_{family}_{i+1}" for i in range(tr.shape[1])]
            method = "pls_residual"
        train_parts.append(pd.DataFrame(tr, index=X_train.index, columns=fam_names))
        test_parts.append(pd.DataFrame(te, index=X_test.index, columns=fam_names))
        names.extend(fam_names)
        metadata.append({"family": family, "method": method, "n_input_features": len(cols), "n_factors": len(fam_names)})
    train = pd.concat(train_parts, axis=1) if train_parts else pd.DataFrame(index=X_train.index)
    test = pd.concat(test_parts, axis=1) if test_parts else pd.DataFrame(index=X_test.index)
    return train, test, names, metadata
