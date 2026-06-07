"""Ridge trainer (legacy thin wrapper — see models.boosters for purged CV)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.linear_model import Ridge

from dual8h.models.common import score_panel as score_panel  # re-export
from dual8h.models.common import xs_zscore_panel as _xs_zscore


def fit_ridge(train_panel: pd.DataFrame, features: List[str], target: str,
              alpha: float = 1000.0) -> Tuple[Ridge, Dict[str, Any]]:
    z = _xs_zscore(train_panel, features)
    mask = z[features].notna().all(axis=1) & z[target].notna()
    z = z[mask].reset_index(drop=True)
    X = z[features].values
    y = z[target].values
    m = Ridge(alpha=alpha, random_state=0)
    m.fit(X, y)
    info = {
        "alpha": alpha, "n_train": len(X), "features": list(features),
        "target": target,
        "coefficients": dict(zip(features, m.coef_.tolist())),
        "intercept": float(m.intercept_),
    }
    return m, info
