"""Calendar splits with 7-day purge between consecutive splits.

Purge rule: a row whose timestamp + purge_days >= NEXT_split_start is dropped
from the CURRENT split. This prevents the row's 7-day forward target from
overlapping the next split.
"""
from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


def calendar_split(
    panel: pd.DataFrame, cfg_like: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, validation, test) with 7d purge applied to train and validation."""
    ts = pd.to_datetime(panel["timestamp"], utc=True)
    train_s = pd.Timestamp(cfg_like["train_start"], tz="UTC")
    train_e = pd.Timestamp(cfg_like["train_end"], tz="UTC")
    val_s = pd.Timestamp(cfg_like["validation_start"], tz="UTC")
    val_e = pd.Timestamp(cfg_like["validation_end"], tz="UTC")
    test_s = pd.Timestamp(cfg_like["test_start"], tz="UTC")
    test_e = pd.Timestamp(cfg_like["test_end"], tz="UTC")
    purge = pd.Timedelta(days=int(cfg_like["purge_days"]))

    in_train = (ts >= train_s) & (ts < train_e)
    in_val = (ts >= val_s) & (ts < val_e)
    in_test = (ts >= test_s) & (ts < test_e)

    # Purge: drop rows from earlier split whose target window crosses the next.
    train_mask = in_train & (ts + purge <= val_s)
    val_mask = in_val & (ts + purge <= test_s)
    test_mask = in_test                              # nothing after test, no purge

    train = panel.loc[train_mask].copy()
    val = panel.loc[val_mask].copy()
    test = panel.loc[test_mask].copy()
    return train, val, test
