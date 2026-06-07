"""Restrict panel to full cross-sections (all symbols present with valid features)."""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def filter_to_universe(panel: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """Keep only rows for the requested symbol list."""
    sym_set = set(symbols)
    out = panel.loc[panel["symbol"].isin(sym_set)].copy()
    return out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)


def filter_common_history(
    panel: pd.DataFrame,
    symbols: List[str],
    required_cols: List[str],
) -> pd.DataFrame:
    """Keep timestamps where every symbol has non-NaN values for required_cols."""
    n = len(symbols)
    good_ts = []
    for ts, g in panel.groupby("timestamp", sort=True):
        if g["symbol"].nunique() != n:
            continue
        if set(g["symbol"]) != set(symbols):
            continue
        sub = g[required_cols]
        if sub.notna().all().all():
            good_ts.append(ts)
    if not good_ts:
        raise ValueError("No timestamp has a complete cross-section for all symbols")
    out = panel.loc[panel["timestamp"].isin(good_ts)].copy()
    logger.info(
        "Common-history filter: %d -> %d rows, %d -> %d timestamps",
        len(panel), len(out), panel["timestamp"].nunique(), out["timestamp"].nunique(),
    )
    return out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
