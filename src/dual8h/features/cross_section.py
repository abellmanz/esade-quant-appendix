"""Cross-sectional features (within-timestamp only)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_sectional(panel: pd.DataFrame) -> pd.DataFrame:
    """funding_rate_7d_median (per-asset trailing) + funding_rate_xs_rank (per-timestamp)."""
    out = panel.copy().sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
    if "funding_rate" not in out.columns:
        out["funding_rate"] = np.nan
    # funding_rate_7d_median is computed on the 8h funding event stream in build-panel (_load_raw).
    # funding_rate_xs_rank: rank bar-aligned funding_rate within each timestamp (1 = lowest).
    out["funding_rate_xs_rank"] = out.groupby("timestamp", sort=False)["funding_rate"] \
                                     .transform(lambda s: s.rank(method="average"))
    return out
