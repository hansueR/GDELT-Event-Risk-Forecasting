from dataclasses import dataclass

import numpy as np

from gdelt_risk.features import map_event_families


@dataclass
class EventColumnFilter:
    event_prefix: str = "ddfe_"
    max_zero_ratio: float = 0.99
    min_nonzero_count: int = 30

    def fit(self, train_df):
        event_cols = [c for c in train_df.columns if c.startswith(self.event_prefix)]
        kept = []
        drop_constant = []
        drop_zero = []
        drop_nonzero = []
        for col in event_cols:
            s = train_df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if s.nunique(dropna=False) <= 1:
                drop_constant.append(col)
                continue
            zero_ratio = float((s == 0).mean())
            nonzero_count = int((s != 0).sum())
            if zero_ratio >= self.max_zero_ratio:
                drop_zero.append(col)
                continue
            if nonzero_count < self.min_nonzero_count:
                drop_nonzero.append(col)
                continue
            kept.append(col)
        families = map_event_families(kept)
        kept_by_family = {}
        for col, fam in families.items():
            kept_by_family.setdefault(fam, []).append(col)
        self.event_cols_ = event_cols
        self.kept_columns_ = kept
        self.families_ = families
        self.metadata_ = {
            "original_event_columns": len(event_cols),
            "kept_event_columns": len(kept),
            "dropped_constant": len(drop_constant),
            "dropped_zero_ratio": len(drop_zero),
            "dropped_nonzero_count": len(drop_nonzero),
            "kept_by_family_counts": {k: len(v) for k, v in kept_by_family.items()},
            "kept_columns_by_family": kept_by_family,
        }
        return self

    def transform(self, df):
        return df[self.kept_columns_].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit_transform(self, train_df):
        self.fit(train_df)
        return self.transform(train_df)
