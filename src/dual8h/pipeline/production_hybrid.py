"""Production hybrid: 8h train/decide, 1h execution — canonical production path."""
from __future__ import annotations

import os
import io

import json
import logging
from typing import Any, Dict

import pandas as pd

from dual8h.cadence import ExperimentSpec
from dual8h.config import Config, load_config
from dual8h.experiments.granularity_sweep import _exp_dir
from dual8h.features.exec_panel_1h import build_exec_panel_1h, build_exec_splits_1h, exec_1h_dir
from dual8h.models.common import score_panel
from dual8h.pipeline.execution_modes import (
    CANONICAL_MANIFEST,
    CANONICAL_SUMMARY,
    CANONICAL_TEST_BARS,
    EXECUTION_MODE_HYBRID,
)
from dual8h.pipeline.production_8h import (
    MODEL_NAME,
    ensure_model_trained,
    experiment_spec,
)
from dual8h.reports.metrics import block_bootstrap_sharpe
from dual8h.strategy.hybrid_simulator import simulate_sleeve_hybrid
from dual8h.strategy.scoring import compute_carry_score
from dual8h.strategy.simulator import _BARS_PER_YEAR

logger = logging.getLogger(__name__)

PROFILE = "wide20_8h_dual"
MODE = EXECUTION_MODE_HYBRID
MODEL_TAG_CHOSEN = "8h_fwd5d"


def _hybrid_sim_cfg(cfg: Config, exp: ExperimentSpec, *, fee_mult: float = 1.0) -> Dict[str, Any]:
    prof = cfg.raw["profiles"][cfg.profile]
    fee = float(cfg.raw["strategy"]["fee_bps_per_side"]) * fee_mult
    slip = float(cfg.raw["strategy"]["slippage_bps_per_side"]) * fee_mult
    return {
        "n_long": int(prof["n_long"]),
        "n_short": int(prof["n_short"]),
        "fee_bps_per_side": fee,
        "slippage_bps_per_side": slip,
        "target_vol_ann": cfg.raw["strategy"]["target_vol_ann"],
        "per_asset_weight_cap": cfg.raw["strategy"]["per_asset_weight_cap"],
        "funding_event_hours": cfg.raw["strategy"]["funding_event_hours"],
        "rv_lookback_bars": exp.rv_lookback_bars(),
        "rebalance_every_8h_bars": exp.rebalance_every_bars(),
        "execution_lag_1h_bars": 1,
        "enforce_net_zero": True,
    }


def evaluate_hybrid_sleeve(
    cfg: Config,
    exp: ExperimentSpec,
    panel_8h: pd.DataFrame,
    exec_1h: pd.DataFrame,
    ml_scores: pd.DataFrame,
    *,
    fee_mult: float = 1.0,
) -> tuple[Dict[str, float], pd.DataFrame]:
    w_ml, w_c = cfg.sleeve_weights
    sim_h = _hybrid_sim_cfg(cfg, exp, fee_mult=fee_mult)

    r_panel = panel_8h.merge(ml_scores, on=["timestamp", "symbol"], how="left")
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
    net = sleeve["net"]
    bpy = _BARS_PER_YEAR["1h"]
    metrics = {
        "ml_sharpe": float(ann_sharpe(ml_sim["port_logret_net"], bpy)),
        "carry_sharpe": float(ann_sharpe(carry_sim["port_logret_net"], bpy)),
        "sleeve_sharpe": float(ann_sharpe(net, bpy)),
        "sleeve_max_dd_log": max_drawdown_log(net),
        "sleeve_total_log": float(net.sum()),
        "sleeve_hit_rate": float((net > 0).mean()),
        "sleeve_calmar": calmar_log(net),
        "n_bars": int(len(net)),
        "n_symbols": len(cfg.symbols),
        "sleeve_net_sharpe": float(ann_sharpe(net, bpy)),
        "sleeve_net_max_dd_log": max_drawdown_log(net),
    }
    return metrics, sleeve


def ann_sharpe(x, bpy):
    from dual8h.reports.metrics import ann_sharpe as _sh

    return _sh(x, bpy)


def max_drawdown_log(x):
    from dual8h.reports.metrics import max_drawdown_log as _dd

    return _dd(x)


def calmar_log(x):
    from dual8h.reports.metrics import calmar_log as _c

    return _c(x)


def backtest_hybrid_split(
    cfg: Config,
    exp: ExperimentSpec,
    model,
    split: str,
    *,
    fee_mult: float = 1.0,
) -> Dict[str, Any]:
    panel_8h = pd.read_parquet(_exp_dir(cfg, exp, "splits", f"{split}.parquet"))
    exec_1h = pd.read_parquet(exec_1h_dir(cfg, exp.tag) / "splits" / f"{split}.parquet")
    scores = score_panel(model, panel_8h, cfg.model_features)
    metrics, sleeve = evaluate_hybrid_sleeve(
        cfg, exp, panel_8h, exec_1h, scores, fee_mult=fee_mult,
    )
    w_ml, w_c = cfg.sleeve_weights
    summary = {
        "profile": cfg.profile,
        "mode": MODE,
        "cadence_tag": exp.tag,
        "model": MODEL_NAME,
        "split": split,
        "w_ml": w_ml,
        "w_carry": w_c,
        **metrics,
        "pnl_grid": "1h",
        "decision_grid": "8h",
        "execution_lag": "1h (1 bar)",
        "rebalance_grid": "8h (every 15 bars = 5d)",
    }
    if split == "test" and fee_mult == 1.0:
        bpy = _BARS_PER_YEAR["1h"]
        summary["bootstrap"] = block_bootstrap_sharpe(
            sleeve["net"].to_numpy(),
            block_size=int(cfg.raw["reports"]["bootstrap_block_days"]) * 24,
            n_iter=int(cfg.raw["reports"]["bootstrap_iterations"]),
            bars_per_year=bpy,
            seed=int(cfg.raw.get("random_seed", 42)),
        )
        res_dir = cfg.artifact_dir("results", cfg.profile)
        res_dir.mkdir(parents=True, exist_ok=True)
        _buf = io.BytesIO(); sleeve.to_parquet(_buf, index=False)
        with open(res_dir / CANONICAL_TEST_BARS, "wb") as _h:
            _h.write(_buf.getvalue()); _h.flush(); os.fsync(_h.fileno())
    return summary


def _selection_block(cfg: Config) -> Dict[str, Any]:
    """Build the (8h-only) horizon-selection summary from the leaderboard, if present.

    Source of truth = selection_leaderboard.csv. Keeps the manifest self-documenting
    and reproducible whenever the pipeline regenerates artifacts.
    """
    lb_path = cfg.artifact_dir("experiments", cfg.profile, "selection_leaderboard.csv")
    block: Dict[str, Any] = {
        "protocol": "pre-registered, validation-only",
        "selectable_family": "8h cadence horizons fwd1d..fwd7d (only candidates)",
        "criterion": "validation Sharpe-to-maxDrawdown (validation_sleeve_sharpe / |validation_max_dd|)",
        "chosen_tag": MODEL_TAG_CHOSEN,
        "cross_timeframe_note": "12h/1d candidates removed from selection; production profile is 8h-cadence.",
        "evaluator_note": ("holdout_test_evaluation.csv reports the 8h-bar selection simulator "
                           "(used only to rank horizons); the canonical test headline (Sharpe ~1.70) is the "
                           "8h_decide_1h_exec hybrid in production_summary.csv — same spec, different evaluator."),
    }
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        if "validation_sharpe_dd" not in lb.columns:
            lb["validation_sharpe_dd"] = lb["validation_sleeve_sharpe"] / lb["validation_max_dd"].abs()
        lb = lb[lb["interval"] == "8h"].sort_values("validation_sharpe_dd", ascending=False)
        block["ranked_candidates"] = [
            {
                "tag": r["tag"],
                "rank": int(i + 1),
                "validation_sharpe": round(float(r["validation_sleeve_sharpe"]), 3),
                "validation_max_dd": round(float(r["validation_max_dd"]), 4),
                "validation_sharpe_dd": round(float(r["validation_sharpe_dd"]), 3),
            }
            for i, (_, r) in enumerate(lb.iterrows())
        ]
        if block["ranked_candidates"]:
            block["chosen_tag"] = block["ranked_candidates"][0]["tag"]
    return block


def _write_canonical_artifacts(cfg: Config, exp: ExperimentSpec, rows: list, info) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    res_dir = cfg.artifact_dir("results", cfg.profile)
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / CANONICAL_SUMMARY, "w", encoding="utf-8", newline="") as _h:
        df.to_csv(_h, index=False); _h.flush(); os.fsync(_h.fileno())

    manifest = {
        "profile": cfg.profile,
        "mode": MODE,
        "experiment": exp.tag,
        "canonical_summary": CANONICAL_SUMMARY,
        "leakage_controls": {
            "model_fit_split": "train_only",
            "model_features_panel": "8h",
            "decisions_and_rebalance": "8h_every_15_bars",
            "pnl_simulation": "1h_hourly_returns",
            "execution_lag": "1h_one_bar",
            "validation_used_for": "horizon_selection_only_not_sleeve_weights",
            "selection_family": "8h_horizons_only",
            "test_evaluated_once": True,
            "do_not_add_val_test_to_train": True,
        },
        "selection": _selection_block(cfg),
        "sleeve_weights": {"ml": cfg.sleeve_weights[0], "carry": cfg.sleeve_weights[1]},
        "catboost_info": info,
        "splits": rows,
    }
    with open(res_dir / CANONICAL_MANIFEST, "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, default=str); h.flush(); os.fsync(h.fileno())
    return df


def run_production_hybrid(
    cfg: Config | None = None,
    *,
    fetch: bool = False,
    retrain: bool = False,
) -> pd.DataFrame:
    from dual8h.ingest.run import run_ingest

    cfg = cfg or load_config(profile=PROFILE)
    exp = experiment_spec(cfg)
    logger.info("Canonical production %s / %s", exp.tag, MODE)

    if fetch:
        run_ingest(cfg, force_fetch=True)

    model, info = ensure_model_trained(cfg, exp, retrain=retrain)

    build_exec_panel_1h(cfg, exp_tag=exp.tag)
    build_exec_splits_1h(cfg, exp_tag=exp.tag)

    rows = []
    for split in ("validation", "test"):
        s = backtest_hybrid_split(cfg, exp, model, split)
        rows.append(s)
        logger.info(
            "%s %s: sleeve=%.3f ml=%.3f carry=%.3f n_bars=%d",
            exp.tag, split, s["sleeve_sharpe"], s["ml_sharpe"], s["carry_sharpe"], s["n_bars"],
        )
    return _write_canonical_artifacts(cfg, exp, rows, info)
