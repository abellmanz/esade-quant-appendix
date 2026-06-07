"""Benchmark the strategy against a passive S&P 500 buy-and-hold over the test window.

Strategy P&L is SIMULATED (net of costs + funding). The S&P 500 series is real,
public index data (source: Yahoo Finance, ^GSPC daily closes) used only as a
passive benchmark. Each book's Sharpe/vol is annualised at its native sampling
frequency (strategy = hourly bars; S&P = 252 trading days).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

SPX_CSV_DEFAULT = "data/raw/benchmark/SPX_daily.csv"


def _metrics_logret(r: np.ndarray, ppy: float) -> tuple[float, float, float]:
    r = np.asarray(r, dtype=float)
    mu, sd = r.mean(), r.std(ddof=1)
    sharpe = float(mu / sd * np.sqrt(ppy)) if sd > 0 else float("nan")
    vol = float(sd * np.sqrt(ppy))
    cum = np.cumsum(r)
    dd = float((cum - np.maximum.accumulate(cum)).min())
    return sharpe, vol, dd


def load_spx(csv_path: str | Path = SPX_CSV_DEFAULT) -> pd.DataFrame:
    spx = pd.read_csv(csv_path, parse_dates=["date"])
    spx["date"] = pd.to_datetime(spx["date"], utc=True)
    return spx.sort_values("date").reset_index(drop=True)


def compare_to_sp500(
    strategy_bars: pd.DataFrame,
    spx: pd.DataFrame,
    *,
    bars_per_year: float = 365.25 * 24,
) -> Dict[str, Any]:
    """strategy_bars: columns [timestamp, net] (hourly net log returns, test split)."""
    t = strategy_bars.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    t0, t1 = t["timestamp"].min(), t["timestamp"].max()

    spx = spx[(spx["date"] >= t0.normalize()) & (spx["date"] <= t1)].copy()
    spx["lr"] = np.log(spx["close"] / spx["close"].shift(1))

    # strategy
    r = t["net"].to_numpy()
    sh_s, vol_s, dd_s = _metrics_logret(r, bars_per_year)
    yrs_s = len(r) / bars_per_year
    cagr_s = float(np.exp(r.sum() / yrs_s) - 1)

    # S&P 500 buy & hold
    lr = spx["lr"].dropna().to_numpy()
    sh_b, vol_b, dd_b = _metrics_logret(lr, 252)
    years = (spx["date"].iloc[-1] - spx["date"].iloc[0]).days / 365.25
    tot_b = float(spx["close"].iloc[-1] / spx["close"].iloc[0] - 1)
    cagr_b = float((1 + tot_b) ** (1 / years) - 1)

    # correlation / beta (daily)
    t["date"] = t["timestamp"].dt.normalize()
    daily = t.groupby("date")["net"].sum().rename("strat").reset_index()
    mrg = daily.merge(spx[["date", "lr"]].rename(columns={"lr": "spx"}), on="date").dropna()
    beta = float(np.polyfit(mrg["spx"], mrg["strat"], 1)[0])
    corr = float(np.corrcoef(mrg["spx"], mrg["strat"])[0, 1])

    table = pd.DataFrame([
        {"book": "Basis8 (net, SIMULATED)", "cagr_%": round(cagr_s * 100, 1),
         "vol_%": round(vol_s * 100, 1), "sharpe": round(sh_s, 2),
         "max_dd_%": round((np.exp(dd_s) - 1) * 100, 1)},
        {"book": "S&P 500 buy & hold", "cagr_%": round(cagr_b * 100, 1),
         "vol_%": round(vol_b * 100, 1), "sharpe": round(sh_b, 2),
         "max_dd_%": round((np.exp(dd_b) - 1) * 100, 1)},
    ])
    # overlay equity (growth of $100) over ALL calendar days in the test window.
    # Crypto trades 7d/week; S&P is flat on days it is closed (forward-filled),
    # so the strategy line reflects its true cumulative (incl. weekend P&L).
    days = daily.sort_values("date").copy()
    days["Basis8"] = 100 * np.exp(days["strat"].cumsum())
    spx_px = spx.set_index("date")["close"].reindex(
        pd.DatetimeIndex(days["date"]), method="ffill"
    )
    spx_px = spx_px.bfill()
    days["SP500"] = 100 * (spx_px.to_numpy() / spx_px.to_numpy()[0])
    return {"table": table, "overlay": days[["date", "Basis8", "SP500"]],
            "corr": round(corr, 2), "beta": round(beta, 2),
            "window": (t0, t1), "source": "Yahoo Finance ^GSPC daily closes"}
