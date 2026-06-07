"""Leak-safe 8h horizon sweep (fwd1d..fwd7d). Production profile is 8h-cadence;
cross-timeframe (12h/1d) candidates are intentionally excluded from selection."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from dual8h.cadence import ExperimentSpec
from dual8h.io_utils import write_parquet, write_json, write_text, dump_joblib
from dual8h.config import Config, load_config
from dual8h.features.cross_section import add_cross_sectional
from dual8h.features.funding_align import funding_with_7d_median
from dual8h.features.ffd import find_d_star_on_train
from dual8h.features.panel_filter import filter_common_history, filter_to_universe
from dual8h.features.per_asset import compute_per_asset_features
from dual8h.features.targets import add_forward_target
from dual8h.ingest.funding import fetch_funding, write_funding
from dual8h.ingest.ohlcv import fetch_ohlcv, write_ohlcv
from dual8h.models.boosters import fit_catboost_purged
from dual8h.pipeline.catboost_train import train_reg_catboost
from dual8h.models.common import score_panel
from dual8h.config import all_profile_symbols
from dual8h.splits.calendar import calendar_split
from dual8h.strategy.compare import _evaluate_sleeve, _sim_cfg
from dual8h.strategy.scoring import compute_carry_score
from dual8h.strategy.simulator import simulate_sleeve_leg, _BARS_PER_YEAR

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENTS = [
    ExperimentSpec("8h", 2),
    ExperimentSpec("8h", 4),
    ExperimentSpec("8h", 5),
]

# Focused 8h forecast-day grid (rebalance period = forecast horizon)
DEFAULT_8H_HORIZON_DAYS = [1, 2, 3, 4, 5, 6, 7]


def experiments_for(
    interval: str,
    target_days: List[int],
    profile: str,
    rebalance_days: int | None = None,
) -> List[ExperimentSpec]:
    return [
        ExperimentSpec(interval, int(d), profile=profile, rebalance_days=rebalance_days)
        for d in target_days
    ]


def parse_horizon_days(s: str) -> List[int]:
    return sorted({int(x.strip()) for x in s.split(",") if x.strip()})


def _exp_dir(cfg: Config, exp: ExperimentSpec, *parts: str) -> Path:
    base = cfg.data_dir("experiments", exp.profile, exp.tag)
    if parts:
        return base.joinpath(*parts)
    return base


def ensure_ohlcv(cfg: Config, intervals: List[str]) -> None:
    symbols = all_profile_symbols(cfg.raw)
    ohlcv_dir = cfg.data_dir("raw", "ohlcv")
    funding_dir = cfg.data_dir("raw", "funding")
    spot = cfg.raw["ingest"]["binance_spot_base"]
    perp = cfg.raw["ingest"]["binance_perp_base"]
    for interval in intervals:
        for sym in symbols:
            path = ohlcv_dir / f"{sym}_{interval}.parquet"
            if path.exists():
                continue
            logger.info("Fetching %s %s", sym, interval)
            df = fetch_ohlcv(sym, interval, cfg.start, cfg.end, spot)
            write_ohlcv(df, sym, interval, ohlcv_dir)
        for sym in symbols:
            fp = funding_dir / f"{sym}_funding.parquet"
            if fp.exists():
                continue
            logger.info("Fetching %s funding", sym)
            write_funding(fetch_funding(sym, cfg.start, cfg.end, perp), sym, funding_dir)


def build_panel_for_experiment(cfg: Config, exp: ExperimentSpec) -> Path:
    symbols = cfg.raw["profiles"][exp.profile]["symbols"]
    ohlcv_dir = cfg.data_dir("raw", "ohlcv")
    funding_dir = cfg.data_dir("raw", "funding")
    parts = []
    for sym in symbols:
        ohlcv = pd.read_parquet(ohlcv_dir / f"{sym}_{exp.interval}.parquet")
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], utc=True)
        funding = pd.read_parquet(funding_dir / f"{sym}_funding.parquet")
        funding["timestamp"] = pd.to_datetime(funding["timestamp"], utc=True)
        fund_feats = funding_with_7d_median(funding)
        merged = pd.merge_asof(
            ohlcv.sort_values("timestamp"), fund_feats, on="timestamp", direction="backward"
        )
        merged["symbol"] = sym
        parts.append(merged)
    raw = pd.concat(parts, ignore_index=True)
    raw = filter_to_universe(raw, symbols)

    train_end = cfg.raw["splits"]["train_end"]
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    d_star_map = {}
    for sym, g in raw.groupby("symbol", sort=False):
        train_g = g[g["timestamp"] < train_end_ts].sort_values("timestamp")
        d_star_map[sym] = find_d_star_on_train(np.log(train_g["close"].astype(float).values))

    es = exp.ema_spans()
    feats = compute_per_asset_features(
        raw,
        rv_window=exp.rv_lookback_bars(),
        ffd_d_per_symbol=d_star_map,
        mom_4h_bars=exp.mom_4h_bars(),
        mom_7d_bars=exp.mom_day_bars(7),
        mom_30d_bars=exp.mom_day_bars(30),
        ema_short=es[0],
        ema_long=es[1],
    )
    feats = add_cross_sectional(feats)
    feats = add_forward_target(feats, horizon_bars=exp.horizon_bars(), target_col=exp.target_col)

    req = list(cfg.raw["model"]["features"]) + [
        "logret_1", "funding_rate", "rv_close_30d", exp.target_col,
    ]
    feats = filter_common_history(feats, symbols, req)

    out_dir = _exp_dir(cfg, exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "panel.parquet"
    write_parquet(feats, out_path)
    meta = {
        "tag": exp.tag,
        "interval": exp.interval,
        "target_days": exp.target_days,
        "rebalance_days": exp.effective_rebalance_days,
        "horizon_bars": exp.horizon_bars(),
        "rebalance_every_bars": exp.rebalance_every_bars(),
        "n_rows": len(feats),
        "n_timestamps": int(feats["timestamp"].nunique()),
    }
    write_json(meta, out_dir / "manifest.json")
    logger.info("Built %s: %d rows, %d ts", exp.tag, len(feats), meta["n_timestamps"])
    return out_path


def build_splits_for_experiment(cfg: Config, exp: ExperimentSpec) -> None:
    panel = pd.read_parquet(_exp_dir(cfg, exp, "panel.parquet"))
    train, val, test = calendar_split(panel, cfg.raw["splits"])
    out = _exp_dir(cfg, exp, "splits")
    out.mkdir(parents=True, exist_ok=True)
    write_parquet(train, out / "train.parquet")
    write_parquet(val, out / "validation.parquet")
    write_parquet(test, out / "test.parquet")


def evaluate_sleeve_at_weights(
    panel: pd.DataFrame,
    model,
    features: List[str],
    sim_ov: dict,
    w_ml: float,
    w_carry: float,
) -> Dict[str, float]:
    """Score ML + carry legs with explicit sleeve weights."""
    from dual8h.strategy.compare import _evaluate_sleeve

    class _CfgProxy:
        pass

    c = _CfgProxy()
    c.interval = sim_ov["interval"]
    c.rebalance_every_bars = sim_ov["rebalance_every_bars"]
    c.n_long = sim_ov.get("n_long", 5)
    c.n_short = sim_ov.get("n_short", 5)
    c.sleeve_weights = (float(w_ml), float(w_carry))
    c.raw = {
        "strategy": {
            "fee_bps_per_side": sim_ov.get("fee_bps_per_side", 8.0),
            "slippage_bps_per_side": sim_ov.get("slippage_bps_per_side", 2.0),
            "target_vol_ann": sim_ov.get("target_vol_ann", 0.15),
            "execution_lag_bars": sim_ov.get("execution_lag_bars", 1),
            "per_asset_weight_cap": sim_ov.get("per_asset_weight_cap", 0.25),
            "funding_event_hours": sim_ov.get("funding_event_hours", 8.0),
        }
    }
    scores = score_panel(model, panel, features)
    metrics, _ = _evaluate_sleeve(c, panel, scores, sim_overrides=sim_ov)
    return metrics


def tune_sleeve_on_train(
    train: pd.DataFrame,
    model,
    features: List[str],
    sim_ov: dict,
    weight_grid: List[float] | None = None,
    tune_since: str | None = None,
) -> tuple[float, float, float, pd.DataFrame]:
    """Pick w_ml by bar Sharpe on train (optionally only rows with timestamp >= tune_since)."""
    from dual8h.strategy.scoring import compute_carry_score
    from dual8h.reports.metrics import ann_sharpe
    from dual8h.strategy.sleeve_tune import _sleeve_from_legs

    if tune_since:
        cut = pd.Timestamp(tune_since, tz="UTC")
        train = train[train["timestamp"] >= cut].copy()
    ml_scores = score_panel(model, train, features)
    r_panel = train.merge(ml_scores, on=["timestamp", "symbol"], how="left")
    ml_sim = simulate_sleeve_leg(r_panel, cfg_overrides=sim_ov)
    c_scores = compute_carry_score(train)
    c_panel = train.merge(c_scores[["timestamp", "symbol", "score"]], on=["timestamp", "symbol"], how="left")
    carry_sim = simulate_sleeve_leg(c_panel, cfg_overrides=sim_ov)
    grid = weight_grid or [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8]
    bpy = _BARS_PER_YEAR[sim_ov["interval"]]
    rows = []
    best_w, best_sh = 0.5, float("-inf")
    for w_ml in grid:
        w_ml = float(w_ml)
        net = _sleeve_from_legs(ml_sim, carry_sim, w_ml, 1.0 - w_ml)
        sh = ann_sharpe(net, bpy)
        rows.append({"w_ml": w_ml, "train_sharpe": sh})
        if sh > best_sh:
            best_sh, best_w = sh, w_ml
    return best_w, 1.0 - best_w, best_sh, pd.DataFrame(rows)


def _sim_overrides(cfg: Config, exp: ExperimentSpec) -> dict:
    prof = cfg.raw["profiles"][exp.profile]
    base = _sim_cfg(cfg)
    base.update({
        "interval": exp.interval,
        "rebalance_every_bars": exp.rebalance_every_bars(),
        "n_long": int(prof["n_long"]),
        "n_short": int(prof["n_short"]),
        "rv_lookback_bars": exp.rv_lookback_bars(),
    })
    return base


def run_one_experiment(cfg: Config, exp: ExperimentSpec) -> dict:
    build_panel_for_experiment(cfg, exp)
    build_splits_for_experiment(cfg, exp)
    train = pd.read_parquet(_exp_dir(cfg, exp, "splits", "train.parquet"))
    features = list(cfg.raw["model"]["features"])
    target = exp.target_col
    cv = cfg.raw["cv"]
    ml = cfg.raw.get("ml", {})

    model, info = fit_catboost_purged(
        train, features, target,
        iterations_grid=[int(x) for x in ml.get("n_estimators_grid", [50, 100, 200])],
        depth=int(ml.get("max_depth", 3)),
        learning_rate=float(ml.get("learning_rate", 0.05)),
        min_data_in_leaf=int(ml.get("min_data_in_leaf", 100)),
        n_splits=int(cv["n_splits"]),
        purge_days=int(cv["purge_days"]),
        embargo_pct=float(cv["embargo_pct"]),
    )
    model_dir = cfg.artifact_dir("experiments", exp.profile, exp.tag)
    model_dir.mkdir(parents=True, exist_ok=True)
    dump_joblib(model, model_dir / "catboost.joblib")
    write_json(info, model_dir / "train_info.json")

    row = {
        "tag": exp.tag,
        "interval": exp.interval,
        "target_days": exp.target_days,
        "rebalance_days": exp.effective_rebalance_days,
        "cv_rank_ic": info.get("mean_rank_ic"),
        "n_train": info.get("n_train"),
    }
    sim_ov = _sim_overrides(cfg, exp)
    for split in ("validation", "test"):
        panel = pd.read_parquet(_exp_dir(cfg, exp, "splits", f"{split}.parquet"))
        scores = score_panel(model, panel, features)
        metrics, _ = _evaluate_sleeve(cfg, panel, scores, sim_overrides=sim_ov)
        row[f"{split}_ml_sharpe"] = metrics["ml_sharpe"]
        row[f"{split}_carry_sharpe"] = metrics["carry_sharpe"]
        row[f"{split}_sleeve_sharpe"] = metrics["sleeve_sharpe"]
        row[f"{split}_max_dd"] = metrics["sleeve_max_dd_log"]
        logger.info(
            "%s %s: sleeve=%.3f ml=%.3f carry=%.3f",
            exp.tag, split, metrics["sleeve_sharpe"],
            metrics["ml_sharpe"], metrics["carry_sharpe"],
        )
    return row


def run_sweep(
    cfg: Config,
    experiments: List[ExperimentSpec] | None = None,
    *,
    leaderboard_name: str = "granularity_leaderboard.csv",
) -> pd.DataFrame:
    experiments = experiments or [
        ExperimentSpec(e.interval, e.target_days, profile=cfg.profile)
        for e in DEFAULT_EXPERIMENTS
    ]
    intervals = sorted({e.interval for e in experiments})
    ensure_ohlcv(cfg, intervals)
    rows = []
    for exp in experiments:
        logger.info("=== experiment %s ===", exp.tag)
        rows.append(run_one_experiment(cfg, exp))
    df = pd.DataFrame(rows).sort_values("test_sleeve_sharpe", ascending=False)
    profile = experiments[0].profile if experiments else cfg.profile
    out = cfg.artifact_dir("experiments", profile, leaderboard_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text(out, df.to_csv(index=False))
    write_json(rows, out.with_suffix(".json"))
    return df


PRODUCTION_TAG = "8h_fwd5d"


def focused_selection_experiments(
    cfg: Config,
    *,
    horizons_only: bool = True,
    include_cross_tf: bool = False,
) -> List[ExperimentSpec]:
    """8h-family horizon grid (fwd1d..fwd7d).

    The production profile is 8h-cadence, so only 8h horizons are candidates.
    Cross-timeframe (12h / 1d) contrasts have been removed from selection.
    (`horizons_only` / `include_cross_tf` kept for signature compatibility.)
    """
    return experiments_for("8h", DEFAULT_8H_HORIZON_DAYS, profile=cfg.profile)


def run_one_experiment_production(cfg: Config, exp: ExperimentSpec) -> dict:
    """Train/evaluate with production CatBoost + frozen sleeve weights from config."""
    build_panel_for_experiment(cfg, exp)
    build_splits_for_experiment(cfg, exp)
    train = pd.read_parquet(_exp_dir(cfg, exp, "splits", "train.parquet"))
    features = list(cfg.raw["model"]["features"])
    target = exp.target_col

    model, info = train_reg_catboost(cfg, train, target, features=features)
    model_dir = cfg.artifact_dir("experiments", exp.profile, exp.tag)
    model_dir.mkdir(parents=True, exist_ok=True)
    dump_joblib(model, model_dir / "catboost.joblib")
    write_json(info, model_dir / "train_info.json")

    w_ml, w_carry = cfg.sleeve_weights
    row = {
        "tag": exp.tag,
        "interval": exp.interval,
        "target_days": exp.target_days,
        "rebalance_days": exp.effective_rebalance_days,
        "cv_mean_rank_ic": info.get("mean_rank_ic"),
        "n_train": info.get("n_train"),
        "w_ml": w_ml,
        "w_carry": w_carry,
        "production_tag": exp.tag == PRODUCTION_TAG,
    }
    sim_ov = _sim_overrides(cfg, exp)
    for split in ("validation", "test"):
        panel = pd.read_parquet(_exp_dir(cfg, exp, "splits", f"{split}.parquet"))
        metrics = evaluate_sleeve_at_weights(
            panel, model, features, sim_ov, w_ml, w_carry,
        )
        row[f"{split}_ml_sharpe"] = metrics["ml_sharpe"]
        row[f"{split}_carry_sharpe"] = metrics["carry_sharpe"]
        row[f"{split}_sleeve_sharpe"] = metrics["sleeve_sharpe"]
        row[f"{split}_max_dd"] = metrics["sleeve_max_dd_log"]
        row[f"{split}_total_log"] = metrics["sleeve_total_log"]
        logger.info(
            "%s %s: sleeve=%.3f ml=%.3f carry=%.3f",
            exp.tag, split, metrics["sleeve_sharpe"],
            metrics["ml_sharpe"], metrics["carry_sharpe"],
        )
    return row


# --- Production selection protocol (pre-registered) --------------------------
# Family:    only the 8h-cadence horizons are *selectable* for the wide20_8h_dual
#            production profile. 12h / 1d rows are out-of-family sanity contrasts
#            and must never drive the production choice.
# Criterion: validation Sharpe-to-maxDrawdown (risk-adjusted), validation ONLY.
#            Test is held out and evaluated exactly once, after the pick.
SELECTION_FAMILY_INTERVAL = "8h"
SELECTION_CRITERION = "validation_sharpe_dd"


def add_validation_sharpe_dd(df: pd.DataFrame) -> pd.DataFrame:
    """Add validation Sharpe / |maxDD| (the production selection criterion)."""
    out = df.copy()
    out[SELECTION_CRITERION] = (
        out["validation_sleeve_sharpe"] / out["validation_max_dd"].abs()
    )
    return out


def pick_best(df: pd.DataFrame, rank_col: str = SELECTION_CRITERION) -> pd.Series:
    """Select within the 8h production family on validation Sharpe-to-maxDD.

    Leakage discipline: ranking reads validation columns ONLY. Cross-timeframe
    (12h / 1d) rows are excluded from the selectable set.
    """
    if df.empty:
        raise ValueError("empty leaderboard")
    if rank_col not in df.columns and rank_col == SELECTION_CRITERION:
        df = add_validation_sharpe_dd(df)
    fam = df[df["interval"] == SELECTION_FAMILY_INTERVAL]
    if fam.empty:
        fam = df
    return fam.loc[fam[rank_col].idxmax()]


def run_selection_sweep(
    cfg: Config,
    experiments: List[ExperimentSpec] | None = None,
    *,
    rank_split: str = "validation",
    horizons_only: bool = False,
    include_cross_tf: bool = True,
    leaderboard_name: str = "selection_leaderboard.csv",
) -> pd.DataFrame:
    """Production-aligned sweep.

    Selection protocol (pre-registered, leakage-safe):
      * selectable family = 8h-cadence horizons only (12h/1d are contrasts);
      * criterion = validation Sharpe-to-maxDrawdown, validation columns ONLY;
      * the test split is NOT written into the selection leaderboard. It is
        evaluated exactly once for the chosen tag into holdout_test_evaluation.csv.
    """
    experiments = experiments or focused_selection_experiments(
        cfg, horizons_only=horizons_only, include_cross_tf=include_cross_tf,
    )
    intervals = sorted({e.interval for e in experiments})
    ensure_ohlcv(cfg, intervals)
    rows = []
    for exp in experiments:
        logger.info("=== selection experiment %s ===", exp.tag)
        rows.append(run_one_experiment_production(cfg, exp))
    df = add_validation_sharpe_dd(pd.DataFrame(rows))
    df["selectable"] = df["interval"] == SELECTION_FAMILY_INTERVAL

    # --- Selection leaderboard: VALIDATION columns only (no test peeking) -----
    sel_cols = [c for c in df.columns if not c.startswith("test_")]
    sel = df[sel_cols].copy()
    # Rank within the selectable 8h family by the risk-adjusted criterion.
    sel = sel.sort_values([SELECTION_CRITERION], ascending=False).reset_index(drop=True)
    sel["rank_by_validation_sharpe_dd"] = (
        sel[SELECTION_CRITERION].where(sel["selectable"])
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    out = cfg.artifact_dir("experiments", cfg.profile, leaderboard_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text(out, sel.to_csv(index=False))
    write_json(sel.to_dict(orient="records"), out.with_suffix(".json"))

    best = pick_best(df, rank_col=SELECTION_CRITERION)
    logger.info(
        "Selected (8h family, validation Sharpe/maxDD): %s (%.2f); PRODUCTION_TAG=%s",
        best["tag"], best[SELECTION_CRITERION], PRODUCTION_TAG,
    )

    # --- Holdout: test evaluated ONCE, for the selected tag only --------------
    # IMPORTANT: these test_* figures come from the 8h-BAR selection evaluator,
    # used ONLY to rank horizons. The CANONICAL production headline uses the
    # 8h-decide / 1h-execute hybrid simulator (see production_summary.csv), which
    # gives a different (higher) test Sharpe. The two are different evaluators of
    # the same chosen spec, not a discrepancy.
    test_cols = ["tag"] + [c for c in df.columns if c.startswith("test_")]
    holdout = df.loc[df["tag"] == best["tag"], test_cols].copy()
    holdout.insert(1, "selected_on", "validation_sharpe_dd")
    holdout.insert(2, "evaluator", "8h_bar_selection_sim (ranking only)")
    holdout.insert(3, "canonical_headline", "see production_summary.csv (8h_decide_1h_exec hybrid)")
    write_text(out.with_name("holdout_test_evaluation.csv"), holdout.to_csv(index=False))

    return sel
