"""Tune CatBoost/carry sleeve weights on TRAIN only (bar Sharpe)."""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Tuple

import joblib
import pandas as pd

from dual8h.config import Config
from dual8h.models.common import score_panel
from dual8h.reports.metrics import ann_sharpe, block_bootstrap_sharpe, calmar_log, max_drawdown_log
from dual8h.strategy.compare import _evaluate_sleeve, _sim_cfg
from dual8h.strategy.scoring import compute_carry_score
from dual8h.strategy.simulator import simulate_sleeve_leg, _BARS_PER_YEAR

logger = logging.getLogger(__name__)


def _sleeve_from_legs(
    ml_sim: pd.DataFrame, carry_sim: pd.DataFrame, w_ml: float, w_carry: float
) -> pd.Series:
    merged = ml_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "ml"}).merge(
        carry_sim[["timestamp", "port_logret_net"]].rename(columns={"port_logret_net": "carry"}),
        on="timestamp",
        how="inner",
    )
    return w_ml * merged["ml"] + w_carry * merged["carry"]


def tune_sleeve_weights(
    cfg: Config,
    model_name: str = "catboost_tuned",
    weight_grid: List[float] | None = None,
) -> Tuple[float, float, pd.DataFrame]:
    """Pick w_ml on train; w_carry = 1 - w_ml."""
    train = pd.read_parquet(cfg.data_dir("splits", cfg.profile, "train.parquet"))
    model = joblib.load(cfg.artifact_dir("models", cfg.profile, f"{model_name}.joblib"))
    ml_scores = score_panel(model, train, cfg.model_features)
    r_panel = train.merge(ml_scores, on=["timestamp", "symbol"], how="left")
    ml_sim = simulate_sleeve_leg(r_panel, cfg_overrides=_sim_cfg(cfg))
    c_scores = compute_carry_score(train)
    c_panel = train.merge(c_scores[["timestamp", "symbol", "score"]], on=["timestamp", "symbol"], how="left")
    carry_sim = simulate_sleeve_leg(c_panel, cfg_overrides=_sim_cfg(cfg))

    grid = weight_grid or cfg.raw.get("sleeve_tune", {}).get(
        "ml_weights", [0.0, 0.15, 0.25, 0.35, 0.5]
    )
    bpy = _BARS_PER_YEAR[cfg.interval]
    rows = []
    best_w = 0.5
    best_sh = float("-inf")
    for w_ml in grid:
        w_ml = float(w_ml)
        w_carry = 1.0 - w_ml
        net = _sleeve_from_legs(ml_sim, carry_sim, w_ml, w_carry)
        sh = ann_sharpe(net, bpy)
        rows.append({"w_ml": w_ml, "w_carry": w_carry, "train_sharpe": sh})
        if sh > best_sh:
            best_sh = sh
            best_w = w_ml

    df = pd.DataFrame(rows)
    out_dir = cfg.artifact_dir("results", cfg.profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": cfg.profile,
        "model": model_name,
        "best_w_ml": best_w,
        "best_w_carry": 1.0 - best_w,
        "best_train_sharpe": best_sh,
        "grid": df.to_dict(orient="records"),
    }
    with open(out_dir / "sleeve_weights_tune.json", "w") as h:
        json.dump(manifest, h, indent=2)
    logger.info(
        "Sleeve weights (train): w_ml=%.2f w_carry=%.2f train Sharpe=%.3f",
        best_w, 1.0 - best_w, best_sh,
    )
    return best_w, 1.0 - best_w, df


def evaluate_sleeve_at_weights(
    cfg: Config,
    w_ml: float,
    w_carry: float,
    model_name: str = "catboost_tuned",
    splits: List[str] | None = None,
) -> Dict[str, dict]:
    model = joblib.load(cfg.artifact_dir("models", cfg.profile, f"{model_name}.joblib"))
    bpy = _BARS_PER_YEAR[cfg.interval]
    out = {}
    for split in splits or ["validation", "test"]:
        panel = pd.read_parquet(cfg.data_dir("splits", cfg.profile, f"{split}.parquet"))
        ml_scores = score_panel(model, panel, cfg.model_features)
        r_panel = panel.merge(ml_scores, on=["timestamp", "symbol"], how="left")
        ml_sim = simulate_sleeve_leg(r_panel, cfg_overrides=_sim_cfg(cfg))
        c_scores = compute_carry_score(panel)
        c_panel = panel.merge(c_scores[["timestamp", "symbol", "score"]], on=["timestamp", "symbol"], how="left")
        carry_sim = simulate_sleeve_leg(c_panel, cfg_overrides=_sim_cfg(cfg))
        net = _sleeve_from_legs(ml_sim, carry_sim, w_ml, w_carry)
        summary = {
            "profile": cfg.profile,
            "split": split,
            "w_ml": w_ml,
            "w_carry": w_carry,
            "ml_sharpe": ann_sharpe(ml_sim["port_logret_net"], bpy),
            "carry_sharpe": ann_sharpe(carry_sim["port_logret_net"], bpy),
            "sleeve_sharpe": ann_sharpe(net, bpy),
            "sleeve_max_dd_log": max_drawdown_log(net),
            "sleeve_total_log": float(net.sum()),
            "sleeve_hit_rate": float((net > 0).mean()),
            "sleeve_calmar": calmar_log(net),
            "n_bars": int(len(net)),
        }
        if split == "test":
            summary["bootstrap"] = block_bootstrap_sharpe(
                net.to_numpy(),
                block_size=int(cfg.raw["reports"]["bootstrap_block_days"]),
                n_iter=int(cfg.raw["reports"]["bootstrap_iterations"]),
                bars_per_year=bpy,
                seed=int(cfg.raw.get("random_seed", 42)),
            )
        res_dir = cfg.artifact_dir("results", cfg.profile)
        with open(res_dir / f"{split}_sleeve_tuned_weights.json", "w") as h:
            json.dump(summary, h, indent=2)
        out[split] = summary
        logger.info(
            "%s %s w_ml=%.2f: sleeve=%.3f ml=%.3f carry=%.3f",
            cfg.profile, split, w_ml, summary["sleeve_sharpe"],
            summary["ml_sharpe"], summary["carry_sharpe"],
        )
    return out
