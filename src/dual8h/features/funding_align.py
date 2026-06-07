"""Funding features on the native 8h event stream (matches research blocks.py)."""
from __future__ import annotations

import pandas as pd

FUNDING_CADENCE_HOURS = 8.0
EVENTS_PER_DAY = 24.0 / FUNDING_CADENCE_HOURS
WINDOW_7D_EVENTS = max(int(7 * EVENTS_PER_DAY), 1)


def funding_with_7d_median(funding: pd.DataFrame) -> pd.DataFrame:
    """Rolling 7-day median on funding *events*, then merge-ready columns."""
    f = funding.sort_values("timestamp").copy()
    f["funding_rate"] = pd.to_numeric(f["funding_rate"], errors="coerce")
    f["funding_rate_7d_median"] = (
        f["funding_rate"].rolling(WINDOW_7D_EVENTS, min_periods=1).median()
    )
    return f[["timestamp", "funding_rate", "funding_rate_7d_median"]]
