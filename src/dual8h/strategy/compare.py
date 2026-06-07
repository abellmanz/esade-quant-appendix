"""Sleeve evaluation helpers for 8h dual production backtests."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from dual8h.config import Config
from dual8h.reports.metrics import ann_sharpe, calmar_log, max_drawdown_log
from dual8h.strategy.scoring import compute_carry_score
from dual8h.strategy.simulator import simulate_sleeve_leg, _BARS_PER_YEAR


def _sim_cfg(cfg: Config) -> Dict[str, Any]:
    return {
        "interval": cfg.interval,
        "rebalance_every_bars": cfg.rebalance_every_bars,
        "n_long": cfg.n_long,
        "n_short": cfg.n_short,
        "fee_bps_per_side": cfg.raw["strategy"]["fee_bps_per_side"],
        "slippage_bps_per_side": cfg.raw["strategy"]["slippage_bps_per_side"],
        "target_vol_ann": cfg.raw["strategy"]["target_vol_ann"],
        "execution_lag_bars": cfg.raw["strategy"]["execution_lag_bars"],
        "per_asset_weight_cap": cfg.raw["strategy"]["per_asset_weight_cap"],
        "funding_event_hours": cfg.raw["strategy"]["funding_event_hours"],
    }


def _evaluate_sleeve(
    cfg: Config,
    panel: pd.DataFrame,
    ml_scores: pd.DataFrame,
    sim_overrides: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    sim = {**_sim_cfg(cfg), **(sim_overrides or {})}
    interval = sim.get("interval", cfg.interval)
    bpy = _BARS_PER_YEAR[interval]
    w_ml, w_carry = cfg.sleeve_weights

    r_panel = panel.merge(ml_scores, on=["timestamp", "symbol"], how="left")
    r_sim = simulate_sleeve_leg(r_panel, cfg_overrides=sim)

    c_scores = compute_carry_score(panel)
    c_panel = panel.merge(c_scores[["timestamp", "symbol", "score"]], on=["timestamp", "symbol"], how="left")
    c_sim = simulate_sleeve_leg(c_panel, cfg_overrides=sim)

    sleeve = r_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "ml"}).merge(
        c_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "carry"}),
        on="timestamp",
        how="inner",
    )
    sleeve["net"] = w_ml * sleeve["ml"] + w_carry * sleeve["carry"]
    net = sleeve["net"]
    out = {
        "ml_sharpe": ann_sharpe(r_sim["port_logret_net"], bpy),
        "carry_sharpe": ann_sharpe(c_sim["port_logret_net"], bpy),
        "sleeve_sharpe": ann_sharpe(net, bpy),
        "sleeve_max_dd_log": max_drawdown_log(net),
        "sleeve_total_log": float(net.sum()),
        "sleeve_hit_rate": float((net > 0).mean()),
        "sleeve_calmar": calmar_log(net),
        "n_bars": int(len(net)),
    }
    return out, sleeve
