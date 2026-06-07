"""Audit-clean simulator: 1-bar fill lag, funding at start of period, per-bar costs.

This is the production simulator. Every audit fix from the research pass is baked in:

  * 1-bar fill lag (C3): weights set at ts_signal apply to return ts_signal+1 -> ts_signal+2.
  * Funding rate at START of holding period (extended-pass fix): use fund_wide.loc[ts_pnl_prev],
    not fund_wide.loc[ts_pnl].
  * Net-zero leg enforcement (extended-pass fix): handled in portfolio.leg_weights.
  * Per-asset weight cap (extended-pass): applied after vol scaling.
  * Fees + slippage on BOTH sides of every change in notional.
  * Perpetual funding modelled (C1): prorated per bar as funding * bar_hours / 8.

Inputs:
  panel: pandas DataFrame with at minimum columns
    timestamp, symbol, score, logret_1, rv_close_30d, funding_rate
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from dual8h.strategy.portfolio import leg_weights, vol_scale, apply_per_asset_cap

# Sensible defaults for 1d cadence. Override via cfg_overrides.
_DEFAULTS = dict(
    interval="1d",
    rebalance_every_bars=7,
    n_long=2, n_short=2,
    fee_bps_per_side=8.0, slippage_bps_per_side=2.0,
    target_vol_ann=0.15,
    execution_lag_bars=1,
    per_asset_weight_cap=0.25,
    funding_event_hours=8.0,
    rv_lookback_bars=30,
    enforce_net_zero=True,
)

from dual8h.cadence import BAR_HOURS, BARS_PER_YEAR as _CADENCE_BPY

_BARS_PER_YEAR = dict(_CADENCE_BPY)
_BARS_PER_YEAR.update({
    "5m": (365.25 * 24 * 60) / 5,
    "15m": (365.25 * 24 * 60) / 15,
    "4h": (365.25 * 24) / 4,
})
_INTERVAL_HOURS = dict(BAR_HOURS)
_INTERVAL_HOURS.update({"5m": 5 / 60, "15m": 0.25, "4h": 4.0})


def simulate_sleeve_leg(panel: pd.DataFrame, cfg_overrides: Dict[str, Any] = None) -> pd.DataFrame:
    """Run the audit-clean simulator for one leg. Returns per-bar PnL frame."""
    p = dict(_DEFAULTS); p.update(cfg_overrides or {})

    score_wide = panel.pivot(index="timestamp", columns="symbol", values="score").sort_index()
    ret_wide = panel.pivot(index="timestamp", columns="symbol", values="logret_1").sort_index()
    vol_wide = panel.pivot(index="timestamp", columns="symbol", values="rv_close_30d").sort_index()
    fund_wide = panel.pivot(index="timestamp", columns="symbol", values="funding_rate").sort_index()

    timestamps = score_wide.index
    bpy = _BARS_PER_YEAR[p["interval"]]
    bar_hours = _INTERVAL_HOURS[p["interval"]]
    funding_share = bar_hours / p["funding_event_hours"]
    sqrt_lookback = float(np.sqrt(p["rv_lookback_bars"]))
    sqrt_bpy = float(np.sqrt(bpy))
    cost_factor = 2.0 * (p["fee_bps_per_side"] + p["slippage_bps_per_side"]) / 10_000.0
    lag = max(int(p["execution_lag_bars"]), 0)

    current_w = pd.Series(0.0, index=score_wide.columns)
    rows: List[Dict[str, Any]] = []
    for idx in range(len(timestamps) - 1 - lag):
        ts_signal = timestamps[idx]
        ts_pnl = timestamps[idx + 1 + lag]
        ts_pnl_prev = timestamps[idx + lag]   # start of the holding period

        rebalanced = False; turnover = 0.0; fee_cost = 0.0; scale = 1.0
        if idx % p["rebalance_every_bars"] == 0:
            score_now = score_wide.loc[ts_signal].dropna()
            vol_now = vol_wide.loc[ts_signal]
            per_bar_vol = vol_now / sqrt_lookback
            asset_vol_ann = per_bar_vol * sqrt_bpy
            inv_vol = 1.0 / vol_now.replace(0, np.nan)
            raw_w = leg_weights(score_now, inv_vol, p["n_long"], p["n_short"],
                                enforce_net_zero=p["enforce_net_zero"])
            target_w, scale = vol_scale(raw_w, asset_vol_ann, p["target_vol_ann"])
            target_w = apply_per_asset_cap(target_w, p["per_asset_weight_cap"])
            target_w = target_w.reindex(current_w.index).fillna(0.0)
            turnover = 0.5 * float((target_w - current_w).abs().sum())
            fee_cost = turnover * cost_factor
            current_w = target_w
            rebalanced = True

        asset_rets = ret_wide.loc[ts_pnl].fillna(0.0)
        # FUNDING FIX: rate at start of holding period.
        funding_rates = fund_wide.loc[ts_pnl_prev].fillna(0.0)
        port_gross = float((current_w * asset_rets).sum())
        funding_cost = float((current_w * funding_rates * funding_share).sum())
        port_net = port_gross - fee_cost - funding_cost
        rows.append({"timestamp": ts_pnl, "port_logret_gross": port_gross,
                     "port_logret_net": port_net, "fee_cost": fee_cost,
                     "funding_cost": funding_cost, "turnover": turnover,
                     "rebalanced": rebalanced,
                     "gross_exposure": float(current_w.abs().sum()),
                     "net_exposure": float(current_w.sum()),
                     "target_vol_scale": float(scale)})
    return pd.DataFrame(rows)
