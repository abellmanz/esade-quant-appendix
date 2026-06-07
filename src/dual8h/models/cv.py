"""Purged-and-embargoed k-fold CV (López de Prado AFML ch. 7)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    train_mask: np.ndarray
    test_mask: np.ndarray
    fold_index: int


def purged_kfold(timestamps: pd.Series, n_splits: int = 5, purge_days: int = 7,
                 embargo_pct: float = 0.01) -> Iterator[PurgedFold]:
    if not isinstance(timestamps, pd.Series):
        timestamps = pd.Series(timestamps)
    n = len(timestamps)
    if n < n_splits * 2:
        raise ValueError(f"Need >= {n_splits * 2} obs, got {n}")
    sorted_pos = np.argsort(timestamps.values, kind="mergesort")
    sorted_ts = timestamps.iloc[sorted_pos].reset_index(drop=True)
    edges = np.linspace(0, n, n_splits + 1, dtype=int)
    embargo_bars = max(int(embargo_pct * n), 1)
    purge = pd.Timedelta(days=purge_days)
    ts_idx = pd.DatetimeIndex(sorted_ts)

    for k in range(n_splits):
        a, b = edges[k], edges[k + 1]
        test_left = ts_idx[a]; test_right = ts_idx[b - 1]
        emb_right = ts_idx[min(b + embargo_bars, n) - 1]
        sorted_train = np.ones(n, dtype=bool)
        sorted_test = np.zeros(n, dtype=bool)
        sorted_test[a:b] = True
        sorted_train[a:b] = False
        # PURGE
        overlap = np.asarray((ts_idx + purge >= test_left) & (ts_idx <= test_right))
        sorted_train &= ~overlap
        # EMBARGO
        emb = np.asarray((ts_idx > test_right) & (ts_idx <= emb_right))
        sorted_train &= ~emb
        train_mask = np.zeros(n, dtype=bool); test_mask = np.zeros(n, dtype=bool)
        train_mask[sorted_pos] = sorted_train
        test_mask[sorted_pos] = sorted_test
        yield PurgedFold(train_mask=train_mask, test_mask=test_mask, fold_index=k)
