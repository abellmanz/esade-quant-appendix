"""Per-asset trailing features. Every column at time t uses only data with timestamp <= t.

Ten features (deduplicated by IC cluster from the research pass):

  trend cluster      three_ema_y2, ffd_log_close, mom_30d, mom_7d, mom_4d
  carry              funding_rate_7d_median   (+ funding_rate_xs_rank in cross_section.py)
  realised vol       rv_close_30d, rv_garman_klass_30d
  microstructure     signed_taker_imbalance

Also computes:
  logret_1            per-bar log return (PnL accumulator)
  funding_rate        bar-aligned funding rate (per-bar funding cost)
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS = 1e-12
LN2 = float(np.log(2.0))


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a.astype(float) / b.astype(float).replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _ffd_weights(d: float, thresh: float) -> np.ndarray:
    """Truncated fractional-differencing weights (López de Prado)."""
    w = [1.0]
    k = 1
    while True:
        next_w = -w[-1] / k * (d - k + 1)
        if abs(next_w) < thresh:
            break
        w.append(next_w)
        k += 1
        if k > 5000:
            break
    return np.array(w[::-1])


def _ffd_series_legacy(x: np.ndarray, d: float = 0.4, thresh: float = 1e-4) -> np.ndarray:
    w = _ffd_weights(d, thresh)
    width = len(w)
    out = np.full(len(x), np.nan)
    for i in range(width - 1, len(x)):
        window = x[i - width + 1: i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float((w * window).sum())
    return out


def _per_asset_features_one(
    g: pd.DataFrame,
    rv_window: int = 30,
    ffd_d: float = 0.95,
    mom_4h_bars: int = 1,
    mom_7d_bars: int = 7,
    mom_30d_bars: int = 30,
    ema_short: int = 16,
    ema_long: int = 64,
) -> pd.DataFrame:
    """Compute features for one symbol's chronological frame."""
    g = g.sort_values("timestamp").reset_index(drop=True).copy()
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    open_ = g["open"].astype(float)
    volume = g["volume"].astype(float)
    taker_buy = g["taker_buy_base_volume"].astype(float)

    keep = ["timestamp", "symbol", "open", "high", "low", "close", "volume",
            "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume",
            "quote_volume"]
    for col in ("funding_rate", "funding_rate_7d_median"):
        if col in g.columns:
            keep.append(col)
    out = g[keep].copy()

    # Per-bar log return (trailing).
    logret = np.log(close / close.shift(1))
    out["logret_1"] = logret

    out["mom_4h"] = logret.rolling(mom_4h_bars, min_periods=mom_4h_bars).sum()
    out["mom_7d"] = logret.rolling(mom_7d_bars, min_periods=mom_7d_bars).sum()
    out["mom_30d"] = logret.rolling(mom_30d_bars, min_periods=mom_30d_bars).sum()

    # Realised vol (close-to-close, 30-bar trailing).
    sq = (logret ** 2).rolling(rv_window, min_periods=rv_window).sum()
    out["rv_close_30d"] = np.sqrt(sq)

    # Garman-Klass 30-bar.
    gk_inner = 0.5 * (np.log(high / low) ** 2) - (2 * LN2 - 1) * (np.log(close / open_) ** 2)
    out["rv_garman_klass_30d"] = np.sqrt(gk_inner.rolling(rv_window, min_periods=rv_window).mean().clip(lower=0))

    # Microstructure: signed taker imbalance (per-bar; trailing by construction).
    out["signed_taker_imbalance"] = 2.0 * _safe_div(taker_buy, volume) - 1.0

    # Three-EMA y2 signal (pair = 16d short, 64d long; standardised by long-window std).
    short_bars, long_bars = ema_short, ema_long
    e_s = close.ewm(span=short_bars, adjust=False, min_periods=short_bars).mean()
    e_l = close.ewm(span=long_bars, adjust=False, min_periods=long_bars).mean()
    x = e_s - e_l
    y = _safe_div(x, x.rolling(long_bars, min_periods=long_bars).std())
    out["three_ema_y2"] = y

    # FFD on log-close (causal convolution).
    from dual8h.features.ffd import compute_ffd_weights, apply_ffd
    d_ffd = ffd_d
    out["ffd_log_close"] = apply_ffd(np.log(close.values), compute_ffd_weights(d_ffd))

    return out


def compute_per_asset_features(
    panel: pd.DataFrame,
    rv_window: int = 30,
    ffd_d_per_symbol=None,
    mom_4h_bars: int = 1,
    mom_7d_bars: int = 7,
    mom_30d_bars: int = 30,
    ema_short: int = 16,
    ema_long: int = 64,
) -> pd.DataFrame:
    """Apply per-asset feature engineering to each symbol independently."""
    parts = []
    for sym, g in panel.groupby("symbol", sort=False):
        d = (ffd_d_per_symbol or {}).get(sym, 0.95)
        parts.append(_per_asset_features_one(
            g, rv_window=rv_window, ffd_d=d,
            mom_4h_bars=mom_4h_bars, mom_7d_bars=mom_7d_bars, mom_30d_bars=mom_30d_bars,
            ema_short=ema_short, ema_long=ema_long,
        ))
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
