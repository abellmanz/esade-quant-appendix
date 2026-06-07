"""Hybrid simulator: 8h rebalance decisions, 1h returns and execution lag."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from dual8h.strategy.portfolio import apply_per_asset_cap, leg_weights, vol_scale
from dual8h.strategy.simulator import _BARS_PER_YEAR, _INTERVAL_HOURS

_DEFAULTS = dict(
    n_long=5,
    n_short=5,
    fee_bps_per_side=8.0,
    slippage_bps_per_side=2.0,
    target_vol_ann=0.15,
    per_asset_weight_cap=0.25,
    funding_event_hours=8.0,
    rv_lookback_bars=90,
    enforce_net_zero=True,
    rebalance_every_8h_bars=15,
    execution_lag_1h_bars=1,
)


def _rebalance_times_8h(decision_panel: pd.DataFrame, every: int) -> List[pd.Timestamp]:
    ts = decision_panel["timestamp"].drop_duplicates().sort_values()
    return [t for i, t in enumerate(ts) if i % every == 0]


def _weights_at_rebalance(
    decision_panel: pd.DataFrame,
    ts: pd.Timestamp,
    p: Dict[str, Any],
    columns: pd.Index,
) -> tuple:
    """Portfolio weights from 8h cross-section at rebalance time."""
    g = decision_panel.loc[decision_panel["timestamp"] == ts]
    score = g.set_index("symbol")["score"].dropna()
    vol = g.set_index("symbol")["rv_close_30d"]
    sqrt_lookback = float(np.sqrt(p["rv_lookback_bars"]))
    sqrt_bpy_8h = float(np.sqrt(_BARS_PER_YEAR["8h"]))
    per_bar_vol = vol / sqrt_lookback
    asset_vol_ann = per_bar_vol * sqrt_bpy_8h
    inv_vol = 1.0 / vol.replace(0, np.nan)
    raw_w = leg_weights(
        score, inv_vol, p["n_long"], p["n_short"], enforce_net_zero=p["enforce_net_zero"]
    )
    target_w, scale = vol_scale(raw_w, asset_vol_ann, p["target_vol_ann"])
    target_w = apply_per_asset_cap(target_w, p["per_asset_weight_cap"])
    return target_w.reindex(columns).fillna(0.0), float(scale)


def simulate_sleeve_hybrid(
    exec_1h: pd.DataFrame,
    decision_panel_8h: pd.DataFrame,
    *,
    cfg_overrides: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    exec_1h: timestamp, symbol, logret_1, funding_rate (1h grid)
    decision_panel_8h: timestamp, symbol, score, rv_close_30d (8h grid)

    Vectorised: weight computation runs once per rebalance event (~N/15
    iterations) and PnL is computed with numpy matrix ops.
    """
    p = dict(_DEFAULTS)
    p.update(cfg_overrides or {})

    ret_wide  = exec_1h.pivot(index="timestamp", columns="symbol", values="logret_1").sort_index()
    fund_wide = exec_1h.pivot(index="timestamp", columns="symbol", values="funding_rate").sort_index()
    columns    = ret_wide.columns
    timestamps = ret_wide.index
    n          = len(timestamps)
    lag        = max(int(p["execution_lag_1h_bars"]), 0)
    bar_hours  = _INTERVAL_HOURS["1h"]
    funding_share = bar_hours / p["funding_event_hours"]
    cost_factor   = 2.0 * (p["fee_bps_per_side"] + p["slippage_bps_per_side"]) / 10_000.0
    valid = n - 1 - lag

    rebalance_times = _rebalance_times_8h(decision_panel_8h, int(p["rebalance_every_8h_bars"]))

    # --- Step 1: compute weights at every rebalance event (small loop) -------
    rebal_w_list: List[np.ndarray] = []
    rebal_scale_list: List[float] = []
    rebal_ts_ns: List[int] = []
    for t_r in rebalance_times:
        w, sc = _weights_at_rebalance(decision_panel_8h, t_r, p, columns)
        rebal_w_list.append(w.values)
        rebal_scale_list.append(sc)
        rebal_ts_ns.append(int(t_r.value))

    if not rebal_w_list:
        result_ts = timestamps[1 + lag: valid + 1 + lag]
        z = np.zeros(valid)
        return pd.DataFrame({
            "timestamp": result_ts, "port_logret_gross": z, "port_logret_net": z,
            "fee_cost": z, "funding_cost": z, "turnover": z,
            "rebalanced": np.zeros(valid, dtype=bool),
            "gross_exposure": z, "net_exposure": z, "target_vol_scale": z,
        })

    rebal_w      = np.array(rebal_w_list)
    rebal_scales = np.array(rebal_scale_list)
    rebal_ts_arr = np.array(rebal_ts_ns)

    # --- Step 2: for each signal bar find the latest rebalance <= ts_signal --
    bar_ts_ns    = timestamps.asi8
    signal_ts_ns = bar_ts_ns[:valid]
    rebal_idx    = np.searchsorted(rebal_ts_arr, signal_ts_ns, side="right") - 1

    # --- Step 3: build per-bar weight matrix (vectorised) --------------------
    has_rebal  = rebal_idx >= 0
    safe_idx   = np.clip(rebal_idx, 0, len(rebal_w) - 1)
    w_bars     = np.where(has_rebal[:, None], rebal_w[safe_idx], 0.0)
    scale_bars = np.where(has_rebal, rebal_scales[safe_idx], 1.0)

    # --- Step 4: vectorised PnL ----------------------------------------------
    ret_arr  = np.nan_to_num(ret_wide.values,  nan=0.0)
    fund_arr = np.nan_to_num(fund_wide.values, nan=0.0)

    ret_pnl   = ret_arr [1 + lag: valid + 1 + lag]
    fund_prev = fund_arr[lag:     valid + lag    ]

    port_gross       = (w_bars * ret_pnl  ).sum(axis=1)
    funding_cost_arr = (w_bars * fund_prev * funding_share).sum(axis=1)

    # --- Step 5: fee costs at rebalance transitions (~n_rebal iterations) ----
    fee_costs      = np.zeros(valid)
    turnover_arr   = np.zeros(valid)
    rebalanced_arr = np.zeros(valid, dtype=bool)
    prev_w         = np.zeros(len(columns))
    last_applied   = -1

    for r_i in range(len(rebal_w)):
        hits = np.where(rebal_idx == r_i)[0]
        if len(hits) == 0 or r_i <= last_applied:
            continue
        bar_i    = int(hits[0])
        target_w = rebal_w[r_i]
        to       = 0.5 * float(np.abs(target_w - prev_w).sum())
        fee_costs[bar_i]      = to * cost_factor
        turnover_arr[bar_i]   = to
        rebalanced_arr[bar_i] = True
        prev_w       = target_w.copy()
        last_applied = r_i

    port_net  = port_gross - fee_costs - funding_cost_arr
    result_ts = timestamps[1 + lag: valid + 1 + lag]

    return pd.DataFrame({
        "timestamp":         result_ts,
        "port_logret_gross": port_gross,
        "port_logret_net":   port_net,
        "fee_cost":          fee_costs,
        "funding_cost":      funding_cost_arr,
        "turnover":          turnover_arr,
        "rebalanced":        rebalanced_arr,
        "gross_exposure":    np.abs(w_bars).sum(axis=1),
        "net_exposure":      w_bars.sum(axis=1),
        "target_vol_scale":  scale_bars,
    })
