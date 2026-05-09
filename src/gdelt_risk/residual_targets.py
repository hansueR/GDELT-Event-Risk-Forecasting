import numpy as np
import pandas as pd

from gdelt_risk.models import make_classifier, predict_score


def price_oof_predictions_by_year(
    train_df,
    price_features,
    y_high,
    min_inner_train_rows=250,
    min_inner_valid_rows=30,
    model_type="logistic",
    random_seed=42,
):
    """Generate leakage-safe price-only OOF scores inside an outer train fold."""
    y_high = pd.Series(y_high, index=train_df.index)
    years = sorted(int(y) for y in train_df["year"].dropna().unique())
    p_oof = pd.Series(np.nan, index=train_df.index, dtype=float)
    fold_rows = []
    for year in years:
        inner_train_idx = train_df.index[train_df["year"] < year]
        inner_valid_idx = train_df.index[train_df["year"] == year]
        if len(inner_train_idx) < min_inner_train_rows or len(inner_valid_idx) < min_inner_valid_rows:
            continue
        if y_high.loc[inner_train_idx].nunique() < 2:
            continue
        model = make_classifier(model_type=model_type, random_seed=random_seed, C=1.0)
        model.fit(train_df.loc[inner_train_idx, price_features], y_high.loc[inner_train_idx].values)
        p_oof.loc[inner_valid_idx] = predict_score(model, train_df.loc[inner_valid_idx, price_features])
        fold_rows.append(
            {
                "inner_valid_year": year,
                "inner_train_rows": int(len(inner_train_idx)),
                "inner_valid_rows": int(len(inner_valid_idx)),
            }
        )
    return p_oof, pd.DataFrame(fold_rows)


def residual_continuous(y_high, p_price_oof):
    y = pd.Series(y_high, index=p_price_oof.index, dtype=float)
    return y - p_price_oof


def unexpected_high_label(y_high, p_price_oof, cutoff_quantile=0.60):
    valid = p_price_oof.notna()
    cutoff = float(p_price_oof.loc[valid].quantile(cutoff_quantile)) if valid.any() else np.nan
    label = ((pd.Series(y_high, index=p_price_oof.index) == 1) & (p_price_oof < cutoff)).astype(int)
    label.loc[~valid] = np.nan
    return label, cutoff
