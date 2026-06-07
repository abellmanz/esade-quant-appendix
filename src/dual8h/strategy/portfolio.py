"""Leg construction with net-zero enforcement + per-asset weight cap."""
from __future__ import annotations

import numpy as np
import pandas as pd


def leg_weights(scores: pd.Series, inv_vol: pd.Series, n_long: int, n_short: int,
                enforce_net_zero: bool = True) -> pd.Series:
    """Top-n_long long / bottom-n_short short. Inverse-vol within each leg. Net zero."""
    scores = scores.dropna()
    if scores.empty:
        return pd.Series(dtype=float)
    inv_vol = inv_vol.reindex(scores.index)
    inv_vol = inv_vol.where(inv_vol > 0, np.nan).fillna(inv_vol.median())
    longs = scores.nlargest(n_long).index
    shorts = scores.nsmallest(n_short).index
    w = pd.Series(0.0, index=scores.index)
    if len(longs):
        iv = inv_vol.reindex(longs); w.loc[longs] = 0.5 * iv / iv.sum()
    if len(shorts):
        iv = inv_vol.reindex(shorts); w.loc[shorts] = -0.5 * iv / iv.sum()
    if enforce_net_zero and abs(w.sum()) > 1e-9:
        pos = w[w > 0]; neg = w[w < 0]
        if pos.sum() > 0:
            w.loc[pos.index] = 0.5 * pos / pos.sum()
        if neg.sum() < 0:
            w.loc[neg.index] = -0.5 * (-neg) / (-neg).sum()
    return w


def vol_scale(weights: pd.Series, ann_vols: pd.Series, target: float, cap: float = 2.0):
    """Scale weights to target ann vol (zero-correlation assumption). Capped at `cap`."""
    if weights.abs().sum() == 0:
        return weights, 1.0
    proj = (weights.reindex(ann_vols.index).fillna(0.0) * ann_vols).pow(2).sum()
    port_vol = float(np.sqrt(proj))
    if port_vol <= 1e-9:
        return weights, 1.0
    s = min(target / port_vol, cap)
    return weights * s, s


def apply_per_asset_cap(weights: pd.Series, cap: float) -> pd.Series:
    """Clip |w_i| <= cap, then rescale each leg back to ±0.5*scale where possible."""
    if cap is None:
        return weights
    w = weights.clip(-cap, cap)
    # Re-balance legs after clip (best-effort; with N=5 and standard caps the re-clip is rare)
    pos = w[w > 0]; neg = w[w < 0]
    pos_target = weights[weights > 0].sum()
    neg_target = weights[weights < 0].sum()
    if pos.sum() > 0 and abs(pos.sum() - pos_target) > 1e-9:
        w.loc[pos.index] = (pos / pos.sum()) * pos_target
        w = w.clip(-cap, cap)
    if neg.sum() < 0 and abs(neg.sum() - neg_target) > 1e-9:
        w.loc[neg.index] = (neg / neg.sum()) * neg_target
        w = w.clip(-cap, cap)
    return w
