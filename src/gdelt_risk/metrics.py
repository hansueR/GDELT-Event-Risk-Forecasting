import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_metric(func, y_true, y_score_or_pred):
    try:
        return float(func(y_true, y_score_or_pred))
    except ValueError:
        return np.nan


def precision_recall_at_top_pct(y_true, y_score, pct=0.20):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    if len(y_true) == 0:
        return np.nan, np.nan
    k = max(1, int(np.ceil(len(y_true) * pct)))
    idx = np.argsort(-y_score)[:k]
    positives_in_top = float(y_true[idx].sum())
    precision = positives_in_top / k
    total_pos = float(y_true.sum())
    recall = positives_in_top / total_pos if total_pos > 0 else np.nan
    return float(precision), float(recall)


def classification_metrics(y_true, y_score, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    p20, r20 = precision_recall_at_top_pct(y_true, y_score, pct=0.20)
    return {
        "auc": _safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": _safe_metric(average_precision_score, y_true, y_score),
        "f1": _safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_true, y_pred),
        "precision": _safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_true, y_pred),
        "recall": _safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_true, y_pred),
        "precision_at_top20pct": p20,
        "recall_at_top20pct": r20,
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "n_test": int(len(y_true)),
    }


def summarize_metrics(fold_metrics):
    if fold_metrics.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["method", "asset", "target", "model_type"] if c in fold_metrics.columns]
    metric_cols = [
        "auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
        "precision_at_top20pct",
        "recall_at_top20pct",
        "positive_rate",
        "n_test",
    ]
    existing = [c for c in metric_cols if c in fold_metrics.columns]
    return fold_metrics.groupby(group_cols, dropna=False)[existing].mean(numeric_only=True).reset_index()


def add_delta_vs_price_only(metrics, price_metrics):
    keys = ["asset", "target", "test_year"]
    base_cols = keys + ["auc", "average_precision", "precision_at_top20pct"]
    base = price_metrics[base_cols].rename(
        columns={
            "auc": "price_only_auc",
            "average_precision": "price_only_average_precision",
            "precision_at_top20pct": "price_only_precision_at_top20pct",
        }
    )
    out = metrics.merge(base, on=keys, how="left")
    out["delta_auc"] = out["auc"] - out["price_only_auc"]
    out["delta_average_precision"] = out["average_precision"] - out["price_only_average_precision"]
    out["delta_precision_at_top20pct"] = (
        out["precision_at_top20pct"] - out["price_only_precision_at_top20pct"]
    )
    return out
