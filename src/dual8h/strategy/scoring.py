"""Scoring functions for the two legs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _xs_z(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True); sd = s.std(ddof=0, skipna=True)
    if not np.isfinite(sd) or sd <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def compute_carry_score(panel: pd.DataFrame) -> pd.DataFrame:
    """Carry leg: long low-funding, short high-funding (cross-sectional)."""
    out = panel[["timestamp", "symbol"]].copy()
    z = panel.groupby("timestamp", observed=True, sort=False)["funding_rate_7d_median"].transform(_xs_z)
    out["score"] = -z
    return out
