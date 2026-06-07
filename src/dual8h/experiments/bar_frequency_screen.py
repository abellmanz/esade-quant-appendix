"""Bar-frequency screening: justify the 8h sampling choice vs 2h/4h/12h.

Leakage discipline: every number here is computed on the TRAIN window only
(2021-03-29 .. 2024-01-01). Validation/test are never touched. The screen
asks a single question — at which sampling frequency do the model's
price-based features carry the most *cross-sectional* predictive signal for
forward returns — measured by rank Information Coefficient (Spearman IC) and
its information ratio (mean IC / std IC).

Economic prior (independent of the IC numbers): Binance perpetual funding
settles every 8h, so 8h bars align one-to-one with funding events; 2h/4h
bars split a funding period and 12h/1d straddle/merge them.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from dual8h.config import Config
from dual8h.features.per_asset import _per_asset_features_one

SCREEN_FEATURES: List[str] = [
    "three_ema_y2", "ffd_log_close", "mom_30d", "mom_7d", "mom_4h",
    "rv_close_30d", "signed_taker_imbalance",
]
_OHLCV_AGG = dict(
    open="first", high="max", low="min", close="last", volume="sum",
    quote_volume="sum", trade_count="sum",
    taker_buy_base_volume="sum", taker_buy_quote_volume="sum",
)


def _resample_1h(df_1h: pd.DataFrame, hours: int) -> pd.DataFrame:
    df = df_1h.set_index("timestamp").sort_index()
    r = df.resample(f"{hours}h", label="right", closed="right").agg(_OHLCV_AGG)
    return r.dropna(subset=["close"]).reset_index()


def _features_at_interval(
    cfg: Config, hours: int, fwd_days: int,
    train_start: pd.Timestamp, train_end: pd.Timestamp,
) -> pd.DataFrame:
    bpd = 24 // hours                       # bars per day at this interval
    parts = []
    for sym in cfg.symbols:
        o = pd.read_parquet(cfg.data_dir("raw", "ohlcv") / f"{sym}_1h.parquet")
        o["timestamp"] = pd.to_datetime(o["timestamp"], utc=True)
        rs = _resample_1h(o, hours).assign(symbol=sym)
        # Lookbacks fixed in *time* so the only thing varying is sampling freq.
        f = _per_asset_features_one(
            rs, rv_window=10 * bpd, ffd_d=0.5,
            mom_4h_bars=max(1, 4 // hours),
            mom_7d_bars=7 * bpd, mom_30d_bars=30 * bpd,
            ema_short=8 * bpd, ema_long=32 * bpd,
        )
        f["symbol"] = sym
        f["fwd"] = np.log(f["close"].shift(-fwd_days * bpd) / f["close"])
        parts.append(f)
    p = pd.concat(parts, ignore_index=True)
    return p[(p["timestamp"] >= train_start) & (p["timestamp"] < train_end)].copy()


def _xs_rank_ic_series(df: pd.DataFrame, sig: str, fwd: str = "fwd") -> pd.Series:
    """Vectorised per-timestamp Spearman IC (Pearson on within-ts ranks)."""
    d = df[["timestamp", sig, fwd]].dropna()
    if d.empty:
        return pd.Series(dtype=float)
    d = d.assign(
        rx=d.groupby("timestamp")[sig].rank(),
        ry=d.groupby("timestamp")[fwd].rank(),
    )
    d["xy"] = d.rx * d.ry
    a = d.groupby("timestamp").agg(
        n=("rx", "size"), sx=("rx", "sum"), sy=("ry", "sum"),
        sxy=("xy", "sum"), sxx=("rx", lambda s: float((s * s).sum())),
        syy=("ry", lambda s: float((s * s).sum())),
    )
    a = a[a["n"] >= 5]
    cov = a.sxy - a.sx * a.sy / a.n
    vx = a.sxx - a.sx ** 2 / a.n
    vy = a.syy - a.sy ** 2 / a.n
    ic = cov / np.sqrt(vx * vy)
    return ic.replace([np.inf, -np.inf], np.nan).dropna()


def screen_one_interval(
    cfg: Config, hours: int, *, fwd_days: int = 5,
    train_start: str = "2021-03-29", train_end: str = "2024-01-01",
    features: Sequence[str] = SCREEN_FEATURES,
) -> Dict[str, float]:
    ts0 = pd.Timestamp(train_start, tz="UTC")
    ts1 = pd.Timestamp(train_end, tz="UTC")
    p = _features_at_interval(cfg, hours, fwd_days, ts0, ts1)

    rec: Dict[str, float] = {"interval": f"{hours}h", "n_bars": int(p["timestamp"].nunique())}
    p2 = p.copy()
    zcols = []
    abs_ics = []
    for c in features:
        ic_series = _xs_rank_ic_series(p, c)
        ic = float(ic_series.mean())
        rec[f"IC_{c}"] = round(ic, 4)
        abs_ics.append(abs(ic))
        # sign-align each feature to its own (train) IC direction for the composite
        rnk = p2.groupby("timestamp")[c].rank()
        demean = rnk - rnk.groupby(p2["timestamp"]).transform("mean")
        p2[c + "_z"] = np.sign(ic) * demean
        zcols.append(c + "_z")
    p2["composite"] = p2[zcols].mean(axis=1)
    cic = _xs_rank_ic_series(p2, "composite")
    rec["IC_composite"] = round(float(cic.mean()), 4)
    rec["ICIR_composite"] = round(float(cic.mean() / cic.std()), 3)
    rec["absIC_mean"] = round(float(np.mean(abs_ics)), 4)
    return rec


def screen_bar_frequencies(
    cfg: Config, intervals: Sequence[int] = (2, 4, 8, 12), **kw
) -> pd.DataFrame:
    rows = [screen_one_interval(cfg, h, **kw) for h in intervals]
    df = pd.DataFrame(rows)
    lead = ["interval", "n_bars", "IC_composite", "ICIR_composite", "absIC_mean"]
    return df[lead + [c for c in df.columns if c not in lead]]
