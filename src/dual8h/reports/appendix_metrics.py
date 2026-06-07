"""ESADE appendix metrics: CAGR, vol, turnover, stress scenarios."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from dual8h.models.common import score_panel
from dual8h.reports.metrics import ann_sharpe, calmar_log, max_drawdown_log
from dual8h.strategy.scoring import compute_carry_score
from dual8h.strategy.simulator import simulate_sleeve_leg, _BARS_PER_YEAR


def ann_vol(rets: pd.Series, bars_per_year: float) -> float:
    sd = float(rets.std(ddof=1))
    return float("nan") if sd <= 1e-12 else float(sd * np.sqrt(bars_per_year))


def cagr_from_logrets(rets: pd.Series, bars_per_year: float) -> float:
    """Compound annual growth from per-bar log returns."""
    n = len(rets)
    if n < 2:
        return float("nan")
    total_log = float(rets.sum())
    years = n / bars_per_year
    if years <= 0:
        return float("nan")
    return float(np.exp(total_log / years) - 1.0)


def equity_curve(net: pd.Series) -> pd.Series:
    return net.cumsum()


def performance_table(net: pd.Series, bars_per_year: float) -> Dict[str, float]:
    return {
        "cagr": cagr_from_logrets(net, bars_per_year),
        "ann_vol": ann_vol(net, bars_per_year),
        "sharpe": ann_sharpe(net, bars_per_year),
        "max_dd_log": max_drawdown_log(net),
        "calmar": calmar_log(net),
        "hit_rate": float((net > 0).mean()) if len(net) else float("nan"),
        "total_log_return": float(net.sum()),
        "n_bars": int(len(net)),
    }


def _sleeve_legs(
    panel: pd.DataFrame,
    model,
    features: List[str],
    sim_ov: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scores = score_panel(model, panel, features)
    r_panel = panel.merge(scores, on=["timestamp", "symbol"], how="left")
    ml_sim = simulate_sleeve_leg(r_panel, cfg_overrides=sim_ov)
    c_scores = compute_carry_score(panel)
    c_panel = panel.merge(c_scores[["timestamp", "symbol", "score"]], on=["timestamp", "symbol"], how="left")
    carry_sim = simulate_sleeve_leg(c_panel, cfg_overrides=sim_ov)
    return ml_sim, carry_sim


def portfolio_turnover(
    ml_sim: pd.DataFrame,
    carry_sim: pd.DataFrame,
    w_ml: float,
    w_carry: float,
    bars_per_year: float,
    rebalance_every_bars: int,
) -> Dict[str, float]:
    """Blend leg turnover on rebalance bars; annualize by rebalance frequency."""
    merged = ml_sim[["timestamp", "turnover", "rebalanced"]].rename(
        columns={"turnover": "turnover_ml", "rebalanced": "rebal_ml"}
    ).merge(
        carry_sim[["timestamp", "turnover", "rebalanced"]].rename(
            columns={"turnover": "turnover_carry", "rebalanced": "rebal_carry"}
        ),
        on="timestamp",
        how="inner",
    )
    merged["turnover_port"] = w_ml * merged["turnover_ml"] + w_carry * merged["turnover_carry"]
    rebal = merged["rebal_ml"] | merged["rebal_carry"]
    on_rebal = merged.loc[rebal, "turnover_port"]
    mean_rebal = float(on_rebal.mean()) if len(on_rebal) else 0.0
    rebalances_per_year = bars_per_year / max(rebalance_every_bars, 1)
    return {
        "mean_turnover_on_rebalance": mean_rebal,
        "annualized_turnover_proxy": mean_rebal * rebalances_per_year,
        "n_rebalance_bars": int(rebal.sum()),
    }


def evaluate_sleeve_detail(
    panel: pd.DataFrame,
    model,
    features: List[str],
    sim_ov: Dict[str, Any],
    w_ml: float,
    w_carry: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Full sleeve metrics + per-bar net returns and blended turnover."""
    interval = sim_ov["interval"]
    bpy = _BARS_PER_YEAR[interval]
    ml_sim, carry_sim = _sleeve_legs(panel, model, features, sim_ov)
    sleeve = ml_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "ml"}).merge(
        carry_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "carry"}),
        on="timestamp",
        how="inner",
    )
    sleeve["net"] = w_ml * sleeve["ml"] + w_carry * sleeve["carry"]
    to = portfolio_turnover(
        ml_sim, carry_sim, w_ml, w_carry, bpy, int(sim_ov["rebalance_every_bars"])
    )
    net = sleeve["net"]
    out = performance_table(net, bpy)
    out.update({
        "ml_sharpe": ann_sharpe(ml_sim["port_logret_net"], bpy),
        "carry_sharpe": ann_sharpe(carry_sim["port_logret_net"], bpy),
        **to,
    })
    sleeve = sleeve.merge(
        ml_sim[["timestamp", "turnover", "rebalanced"]].rename(
            columns={"turnover": "turnover_ml", "rebalanced": "rebal_ml"}
        ),
        on="timestamp",
    ).merge(
        carry_sim[["timestamp", "turnover", "rebalanced"]].rename(
            columns={"turnover": "turnover_carry", "rebalanced": "rebal_carry"}
        ),
        on="timestamp",
    )
    sleeve["turnover_port"] = w_ml * sleeve["turnover_ml"] + w_carry * sleeve["turnover_carry"]
    return out, sleeve


def stress_double_costs(
    panel: pd.DataFrame,
    model,
    features: List[str],
    sim_ov: Dict[str, Any],
    w_ml: float,
    w_carry: float,
) -> Dict[str, float]:
    stressed = dict(sim_ov)
    stressed["fee_bps_per_side"] = 2.0 * float(sim_ov.get("fee_bps_per_side", 8.0))
    stressed["slippage_bps_per_side"] = 2.0 * float(sim_ov.get("slippage_bps_per_side", 2.0))
    metrics, _ = evaluate_sleeve_detail(panel, model, features, stressed, w_ml, w_carry)
    metrics["stress"] = "double_fees_slippage"
    return metrics


def evaluate_hybrid_sleeve_detail(
    cfg,
    exp,
    panel_8h: pd.DataFrame,
    exec_1h: pd.DataFrame,
    model,
    features: List[str],
    *,
    fee_mult: float = 1.0,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Hybrid sleeve metrics (8h decide, 1h PnL) — single evaluation path for appendix."""
    from dual8h.pipeline.production_hybrid import evaluate_hybrid_sleeve
    from dual8h.models.common import score_panel

    scores = score_panel(model, panel_8h, features)
    metrics, sleeve = evaluate_hybrid_sleeve(
        cfg, exp, panel_8h, exec_1h, scores, fee_mult=fee_mult,
    )
    bpy = _BARS_PER_YEAR["1h"]
    out = performance_table(sleeve["net"], bpy)
    out.update({
        "ml_sharpe": metrics["ml_sharpe"],
        "carry_sharpe": metrics["carry_sharpe"],
        "sharpe": metrics["sleeve_sharpe"],
        "max_dd_log": metrics["sleeve_max_dd_log"],
        "calmar": metrics.get("sleeve_calmar"),
        "hit_rate": metrics["sleeve_hit_rate"],
        "n_bars": metrics["n_bars"],
    })
    return out, sleeve


def hybrid_portfolio_turnover(
    ml_sim: pd.DataFrame,
    carry_sim: pd.DataFrame,
    w_ml: float,
    w_carry: float,
    bars_per_year: float,
) -> Dict[str, float]:
    """Turnover on 1h grid from hybrid simulator rebalance flags."""
    merged = ml_sim[["timestamp", "turnover", "rebalanced"]].rename(
        columns={"turnover": "turnover_ml", "rebalanced": "rebal_ml"}
    ).merge(
        carry_sim[["timestamp", "turnover", "rebalanced"]].rename(
            columns={"turnover": "turnover_carry", "rebalanced": "rebal_carry"}
        ),
        on="timestamp",
        how="inner",
    )
    merged["turnover_port"] = w_ml * merged["turnover_ml"] + w_carry * merged["turnover_carry"]
    rebal = merged["rebal_ml"] | merged["rebal_carry"]
    on_rebal = merged.loc[rebal, "turnover_port"]
    mean_rebal = float(on_rebal.mean()) if len(on_rebal) else 0.0
    years = len(merged) / bars_per_year if len(merged) else 0.0
    n_rebal = int(rebal.sum())
    rebalances_per_year = n_rebal / years if years > 0 else 0.0
    return {
        "mean_turnover_on_rebalance": mean_rebal,
        "annualized_turnover_proxy": mean_rebal * rebalances_per_year,
        "n_rebalance_bars": n_rebal,
    }


def hybrid_sleeve_metrics(
    cfg,
    exp,
    panel_8h: pd.DataFrame,
    exec_1h: pd.DataFrame,
    model,
    features: List[str],
    *,
    fee_mult: float = 1.0,
    split: str = "",
) -> Dict[str, float]:
    """CAGR, vol, Sharpe, max DD, turnover for hybrid sleeve (export + notebook)."""
    from dual8h.models.common import score_panel
    from dual8h.pipeline.production_hybrid import _hybrid_sim_cfg
    from dual8h.strategy.hybrid_simulator import simulate_sleeve_hybrid
    from dual8h.strategy.scoring import compute_carry_score

    w_ml, w_c = cfg.sleeve_weights
    sim_h = _hybrid_sim_cfg(cfg, exp, fee_mult=fee_mult)
    scores = score_panel(model, panel_8h, features)
    r_panel = panel_8h.merge(scores, on=["timestamp", "symbol"], how="left")
    ml_dec = r_panel[["timestamp", "symbol", "score", "rv_close_30d"]]
    ml_sim = simulate_sleeve_hybrid(exec_1h, ml_dec, cfg_overrides=sim_h)
    c_scores = compute_carry_score(panel_8h)
    c_panel = panel_8h.merge(c_scores[["timestamp", "symbol", "score"]], on=["timestamp", "symbol"], how="left")
    c_dec = c_panel[["timestamp", "symbol", "score", "rv_close_30d"]]
    carry_sim = simulate_sleeve_hybrid(exec_1h, c_dec, cfg_overrides=sim_h)
    sleeve = ml_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "ml"}).merge(
        carry_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "carry"}),
        on="timestamp",
        how="inner",
    )
    sleeve["net"] = w_ml * sleeve["ml"] + w_c * sleeve["carry"]
    bpy = _BARS_PER_YEAR["1h"]
    out = performance_table(sleeve["net"], bpy)
    out.update(hybrid_portfolio_turnover(ml_sim, carry_sim, w_ml, w_c, bpy))
    out["max_dd_pct"] = float(out["max_dd_log"]) * 100.0
    out["ml_sharpe"] = ann_sharpe(ml_sim["port_logret_net"], bpy)
    out["carry_sharpe"] = ann_sharpe(carry_sim["port_logret_net"], bpy)
    if split:
        out["split"] = split
    return out


def stress_double_costs_hybrid(
    cfg,
    exp,
    panel_8h: pd.DataFrame,
    exec_1h: pd.DataFrame,
    model,
    features: List[str],
    w_ml: float,
    w_carry: float,
) -> Dict[str, float]:
    metrics, _ = evaluate_hybrid_sleeve_detail(
        cfg, exp, panel_8h, exec_1h, model, features, fee_mult=2.0,
    )
    metrics["stress"] = "double_fees_slippage_hybrid"
    return metrics


def stress_subperiod(
    sleeve: pd.DataFrame,
    start: str,
    end: str,
    bars_per_year: float,
    *,
    net_col: str = "net",
    ts_col: str = "timestamp",
) -> Dict[str, float]:
    """Metrics on a calendar slice of bar returns (sleeve frame with timestamp + net)."""
    ts = pd.to_datetime(sleeve[ts_col], utc=True)
    mask = (ts >= pd.Timestamp(start, tz="UTC")) & (ts < pd.Timestamp(end, tz="UTC"))
    sub = sleeve.loc[mask, net_col]
    if len(sub) < 10:
        return {"stress": f"subperiod_{start}_{end}", "error": "too few bars"}
    out = performance_table(sub, bars_per_year)
    out["stress"] = f"subperiod_{start}_{end}"
    out["start"] = start
    out["end"] = end
    return out
