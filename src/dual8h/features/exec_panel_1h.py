"""Minimal 1h panel for hybrid execution (log returns + funding only)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from dual8h.config import Config
from dual8h.features.funding_align import funding_with_7d_median
from dual8h.features.panel_filter import filter_common_history, filter_to_universe
from dual8h.splits.calendar import calendar_split
from dual8h.io_utils import write_parquet

logger = logging.getLogger(__name__)

EXEC_SUBDIR = "1h_exec"


def exec_1h_dir(cfg: Config, exp_tag: str = "8h_fwd5d") -> Path:
    return cfg.data_dir("experiments", cfg.profile, exp_tag, EXEC_SUBDIR)


def build_exec_panel_1h(
    cfg: Config,
    *,
    exp_tag: str = "8h_fwd5d",
    align_to_8h_timestamps: bool = True,
    force: bool = False,
) -> Path:
    """Build hourly logret/funding panel.

    Returns immediately if the panel and all split files already exist (cached).
    Pass force=True to rebuild from raw parquets.
    """
    out_dir = exec_1h_dir(cfg, exp_tag)
    out_path = out_dir / "panel.parquet"
    splits_ok = all(
        (out_dir / "splits" / f"{s}.parquet").exists()
        for s in ("train", "validation", "test")
    )
    if not force and out_path.exists() and splits_ok:
        logger.info("1h exec panel already cached, skipping rebuild (%s)", out_path)
        return out_path

    symbols = cfg.symbols
    ohlcv_dir = cfg.data_dir("raw", "ohlcv")
    funding_dir = cfg.data_dir("raw", "funding")
    parts = []
    for sym in symbols:
        ohlcv = pd.read_parquet(ohlcv_dir / f"{sym}_1h.parquet")
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
        ohlcv = ohlcv.sort_values("timestamp")
        close = ohlcv["close"].astype(float)
        ohlcv["logret_1"] = np.log(close / close.shift(1))
        funding = pd.read_parquet(funding_dir / f"{sym}_funding.parquet")
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
        fund_feats = funding_with_7d_median(funding)
        merged = pd.merge_asof(
            ohlcv[["timestamp", "logret_1"]].sort_values("timestamp"),
            fund_feats.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )
        merged["symbol"] = sym
        parts.append(merged[["timestamp", "symbol", "logret_1", "funding_rate"]])

    raw = pd.concat(parts, ignore_index=True)
    raw = filter_to_universe(raw, symbols)
    req = ["logret_1", "funding_rate"]
    panel = filter_common_history(raw, symbols, req)

    if align_to_8h_timestamps:
        panel_8h_path = cfg.data_dir("experiments", cfg.profile, exp_tag, "panel.parquet")
        if panel_8h_path.exists():
            ts_8h = set(pd.read_parquet(panel_8h_path)["timestamp"].unique())
            t_min, t_max = min(ts_8h), max(ts_8h)
            panel = panel[(panel["timestamp"] >= t_min) & (panel["timestamp"] <= t_max)].copy()
            panel = filter_common_history(panel, symbols, req)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(panel, out_path)
    meta = {
        "exp_tag": exp_tag,
        "interval": "1h",
        "purpose": "hybrid_execution_only",
        "n_rows": int(len(panel)),
        "n_timestamps": int(panel["timestamp"].nunique()),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as h:
        json.dump(meta, h, indent=2)
    logger.info("Built 1h exec panel: %d rows, %d ts", meta["n_rows"], meta["n_timestamps"])
    return out_path


def build_exec_splits_1h(cfg: Config, *, exp_tag: str = "8h_fwd5d") -> None:
    panel = pd.read_parquet(exec_1h_dir(cfg, exp_tag) / "panel.parquet")
    train, val, test = calendar_split(panel, cfg.raw["splits"])
    out = exec_1h_dir(cfg, exp_tag) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(train, out / "train.parquet")
    write_parquet(val, out / "validation.parquet")
    write_parquet(test, out / "test.parquet")
