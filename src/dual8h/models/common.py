"""Shared ML helpers: xs-zscore, purged-CV rank-IC, panel scoring."""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from dual8h.models.cv import purged_kfold


def xs_zscore_panel(panel: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Cross-sectional z-score each feature at each timestamp.
    Vectorised with numpy bincount (~10x faster than pandas groupby.transform).
    """
    out = panel.copy()
    ts_codes, _ = pd.factorize(panel["timestamp"].values, sort=False)
    n_groups = int(ts_codes.max()) + 1
    feat_arr = panel[features].values.astype(float)
    counts = np.bincount(ts_codes, minlength=n_groups).astype(float)
    result = np.empty_like(feat_arr)
    for fi in range(feat_arr.shape[1]):
        col = feat_arr[:, fi]
        grp_sum  = np.bincount(ts_codes, weights=col,       minlength=n_groups)
        grp_sum2 = np.bincount(ts_codes, weights=col * col, minlength=n_groups)
        grp_mean = grp_sum / counts
        grp_var  = np.maximum(grp_sum2 / counts - grp_mean ** 2, 0.0)
        grp_std  = np.sqrt(grp_var)
        row_mean = grp_mean[ts_codes]
        row_std  = grp_std[ts_codes]
        result[:, fi] = np.where(row_std > 1e-12, (col - row_mean) / row_std, 0.0)
    out[features] = result
    return out


def prepare_xy(
    panel: pd.DataFrame, features: List[str], target: str
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray, pd.Series]:
    z = xs_zscore_panel(panel, features)
    mask = z[features].notna().all(axis=1) & z[target].notna()
    z = z.loc[mask].reset_index(drop=True)
    return z, z[target], z[features].values, z["timestamp"]


def mean_rank_ic(timestamps: pd.Series, preds: np.ndarray, y: np.ndarray) -> float:
    tmp = pd.DataFrame({"ts": timestamps.values, "pred": preds, "y": y})
    ics = []
    for _, g in tmp.groupby("ts"):
        if len(g) < 3 or g["pred"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        ics.append(g["pred"].rank().corr(g["y"].rank(), method="pearson"))
    return float(np.mean(ics)) if ics else float("-inf")


def cv_rank_ic(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.Series,
    n_splits: int,
    purge_days: int,
    embargo_pct: float,
    fit_predict_fn,
) -> float:
    scores = []
    for fold in purged_kfold(timestamps, n_splits=n_splits, purge_days=purge_days, embargo_pct=embargo_pct):
        preds = fit_predict_fn(fold.train_mask, fold.test_mask)
        scores.append(mean_rank_ic(timestamps.iloc[fold.test_mask], preds, y[fold.test_mask]))
    return float(np.mean(scores)) if scores else float("-inf")


def score_panel(model: Any, panel: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    z = xs_zscore_panel(panel, features)
    mask = z[features].notna().all(axis=1)
    out = panel[["timestamp", "symbol"]].copy()
    out["score"] = np.nan
    if mask.any():
        out.loc[mask, "score"] = model.predict(z.loc[mask, features].values)
    return out
