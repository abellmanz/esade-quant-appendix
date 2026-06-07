"""Production pipeline: default 8h_decide_1h_exec; optional 8h-baseline sensitivity."""
from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, Literal

import joblib
import pandas as pd

from dual8h.cadence import ExperimentSpec
from dual8h.config import Config, load_config
from dual8h.experiments.granularity_sweep import (
    _exp_dir,
    _sim_overrides,
    build_panel_for_experiment,
    build_splits_for_experiment,
    evaluate_sleeve_at_weights,
)
from dual8h.ingest.run import run_ingest
from dual8h.pipeline.catboost_train import train_reg_catboost
from dual8h.pipeline.execution_modes import (
    CANONICAL_MANIFEST,
    CANONICAL_SUMMARY,
    CANONICAL_TEST_BARS,
    EXECUTION_MODE_8H_BASELINE,
    EXECUTION_MODE_HYBRID,
    SENSITIVITY_8H_SUMMARY,
)
from dual8h.reports.metrics import block_bootstrap_sharpe
from dual8h.strategy.simulator import _BARS_PER_YEAR

logger = logging.getLogger(__name__)

MODEL_NAME = "catboost_reg"
ExecutionMode = Literal["8h_decide_1h_exec", "8h_baseline"]


def experiment_spec(cfg: Config) -> ExperimentSpec:
    c = cfg.cadence
    return ExperimentSpec(
        c["interval"],
        int(c["target_days"]),
        profile=cfg.profile,
        rebalance_days=int(c.get("rebalance_days", c["target_days"])),
    )


PROFILE_8H = "wide20_8h_dual"


def default_execution_mode(cfg: Config) -> str:
    prof = cfg.raw.get("profiles", {}).get(cfg.profile, {})
    if prof.get("production_execution_mode"):
        return str(prof["production_execution_mode"])
    if cfg.profile == PROFILE_8H:
        return str(cfg.raw.get("production", {}).get("execution_mode", EXECUTION_MODE_HYBRID))
    return EXECUTION_MODE_8H_BASELINE


def ensure_model_trained(
    cfg: Config,
    exp: ExperimentSpec,
    *,
    retrain: bool = False,
) -> tuple[Any, Dict[str, Any] | None]:
    """Fit on train split only (8h panel). Reuse saved model unless retrain=True."""
    model_path = cfg.artifact_dir("models", cfg.profile, f"{MODEL_NAME}.joblib")
    info_path = model_path.with_name(f"{MODEL_NAME}_info.json")
    if model_path.exists() and not retrain:
        logger.info("Loading frozen model from %s (train split only was used for fit)", model_path)
        return joblib.load(model_path), None

    build_panel_for_experiment(cfg, exp)
    build_splits_for_experiment(cfg, exp)
    train = pd.read_parquet(_exp_dir(cfg, exp, "splits", "train.parquet"))
    logger.info("Training CatBoost on train split only (%d rows)", len(train))
    model, info = train_reg_catboost(cfg, train, exp.target_col)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _mbuf = io.BytesIO(); joblib.dump(model, _mbuf)
    with open(model_path, "wb") as _mh:
        _mh.write(_mbuf.getvalue()); _mh.flush(); os.fsync(_mh.fileno())
    with open(info_path, "w", encoding="utf-8") as h:
        json.dump(info, h, indent=2)
    return model, info


def backtest_split_8h_baseline(
    cfg: Config,
    exp: ExperimentSpec,
    model,
    split: str,
) -> Dict[str, Any]:
    panel = pd.read_parquet(_exp_dir(cfg, exp, "splits", f"{split}.parquet"))
    sim_ov = _sim_overrides(cfg, exp)
    w_ml, w_c = cfg.sleeve_weights
    metrics = evaluate_sleeve_at_weights(panel, model, cfg.model_features, sim_ov, w_ml, w_c)
    bpy = _BARS_PER_YEAR[sim_ov["interval"]]
    summary = {
        "profile": cfg.profile,
        "mode": EXECUTION_MODE_8H_BASELINE,
        "cadence_tag": exp.tag,
        "model": MODEL_NAME,
        "split": split,
        "w_ml": w_ml,
        "w_carry": w_c,
        **metrics,
        "sleeve_net_sharpe": metrics["sleeve_sharpe"],
        "sleeve_net_max_dd_log": metrics["sleeve_max_dd_log"],
        "sleeve_calmar": metrics.get("sleeve_calmar"),
        "n_symbols": len(cfg.symbols),
        "pnl_grid": "8h",
        "decision_grid": "8h",
    }
    if split == "test":
        from dual8h.models.common import score_panel
        from dual8h.strategy.scoring import compute_carry_score
        from dual8h.strategy.simulator import simulate_sleeve_leg

        scores = score_panel(model, panel, cfg.model_features)
        r_panel = panel.merge(scores, on=["timestamp", "symbol"], how="left")
        ml_sim = simulate_sleeve_leg(r_panel, cfg_overrides=sim_ov)
        c_scores = compute_carry_score(panel)
        c_panel = panel.merge(c_scores[["timestamp", "symbol", "score"]], on=["timestamp", "symbol"], how="left")
        carry_sim = simulate_sleeve_leg(c_panel, cfg_overrides=sim_ov)
        sleeve = ml_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "ml"}).merge(
            carry_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "carry"}),
            on="timestamp",
        )
        sleeve["net"] = w_ml * sleeve["ml"] + w_c * sleeve["carry"]
        summary["bootstrap"] = block_bootstrap_sharpe(
            sleeve["net"].to_numpy(),
            block_size=int(cfg.raw["reports"]["bootstrap_block_days"]),
            n_iter=int(cfg.raw["reports"]["bootstrap_iterations"]),
            bars_per_year=bpy,
            seed=int(cfg.raw.get("random_seed", 42)),
        )
        res_dir = cfg.artifact_dir("results", cfg.profile)
        sleeve.to_parquet(res_dir / "test_catboost_reg_8h_baseline_bars.parquet", index=False)
    return summary


def run_production_8h_baseline(cfg: Config, *, fetch: bool = False, retrain: bool = False) -> pd.DataFrame:
    """Sensitivity only — does not overwrite canonical production_summary.csv."""
    if fetch:
        run_ingest(cfg, force_fetch=True)
    exp = experiment_spec(cfg)
    model, info = ensure_model_trained(cfg, exp, retrain=retrain)
    rows = []
    for split in ("validation", "test"):
        rows.append(backtest_split_8h_baseline(cfg, exp, model, split))
    df = pd.DataFrame(rows)
    res_dir = cfg.artifact_dir("results", cfg.profile)
    if cfg.profile == PROFILE_8H:
        out = res_dir / SENSITIVITY_8H_SUMMARY
        logger.info("Wrote 8h panel baseline sensitivity to %s (canonical is hybrid)", out)
    else:
        out = res_dir / CANONICAL_SUMMARY
        manifest = {
            "profile": cfg.profile,
            "mode": EXECUTION_MODE_8H_BASELINE,
            "experiment": exp.tag,
            "splits": rows,
        }
        with open(res_dir / CANONICAL_MANIFEST, "w", encoding="utf-8") as h:
            json.dump(manifest, h, indent=2, default=str)
        logger.info("Wrote panel-native production to %s", out)
    df.to_csv(out, index=False)
    return df


def run_production(
    cfg: Config,
    *,
    fetch: bool = False,
    execution_mode: str | None = None,
    retrain: bool = False,
) -> pd.DataFrame:
    """Canonical production. Default: 8h_decide_1h_exec (hybrid path)."""
    mode = execution_mode or default_execution_mode(cfg)
    if mode == EXECUTION_MODE_HYBRID:
        if cfg.raw["profiles"][cfg.profile]["cadence"]["interval"] != "8h":
            raise ValueError(f"Hybrid execution requires 8h cadence profile, got {cfg.profile}")
        from dual8h.pipeline.production_hybrid import run_production_hybrid

        return run_production_hybrid(cfg, fetch=fetch, retrain=retrain)
    if mode == EXECUTION_MODE_8H_BASELINE:
        return run_production_8h_baseline(cfg, fetch=fetch, retrain=retrain)
    raise ValueError(f"Unknown execution_mode: {mode}")


def run_production_8h(cfg: Config, *, fetch: bool = False) -> pd.DataFrame:
    """Backward-compatible alias — runs canonical production (hybrid by default)."""
    return run_production(cfg, fetch=fetch)
