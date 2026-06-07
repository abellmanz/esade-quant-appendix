"""Compare 8h baseline, full-1h, and hybrid (8h decide / 1h exec) backtests.

Loads existing artifacts when possible — does not re-run canonical hybrid twice.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

import pandas as pd

from dual8h.cadence import ExperimentSpec, days_to_bars
from dual8h.config import Config, load_config
from dual8h.experiments.granularity_sweep import _exp_dir, build_panel_for_experiment, build_splits_for_experiment
from dual8h.pipeline.execution_modes import (
    CANONICAL_SUMMARY,
    EXECUTION_MODE_8H_BASELINE,
    EXECUTION_MODE_HYBRID,
    SENSITIVITY_8H_SUMMARY,
)
from dual8h.pipeline.production_8h import run_production_8h_baseline
from dual8h.pipeline.production_hybrid import MODE as HYBRID_MODE

logger = logging.getLogger(__name__)

PROFILE_8H = "wide20_8h_dual"
PROFILE_1H = "wide20_1h_dual"


def split_row_counts(cfg: Config, exp: ExperimentSpec) -> Dict[str, Any]:
    split_dir = _exp_dir(cfg, exp, "splits")
    if not (split_dir / "train.parquet").exists():
        build_panel_for_experiment(cfg, exp)
        build_splits_for_experiment(cfg, exp)
    panel = pd.read_parquet(_exp_dir(cfg, exp, "panel.parquet"))
    counts = {"panel_rows": int(len(panel)), "n_timestamps": int(panel["timestamp"].nunique())}
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(split_dir / f"{split}.parquet")
        counts[f"{split}_rows"] = int(len(df))
    return counts


def _append_rows(
    rows: list,
    df: pd.DataFrame,
    *,
    mode: str,
    interval: str,
    lag_label: str,
    rebalance_bars: int,
    horizon_bars: int,
    profile: str,
    counts: Dict[str, Any],
) -> None:
    for _, r in df.iterrows():
        rows.append({
            "mode": mode,
            "profile": profile,
            "interval": interval,
            "execution_lag_wall_clock": lag_label,
            "rebalance_bars": rebalance_bars,
            "horizon_bars": horizon_bars,
            "split": r["split"],
            "sleeve_sharpe": r["sleeve_sharpe"],
            "sleeve_max_dd_pct": float(r["sleeve_max_dd_log"]) * 100,
            "ml_sharpe": r["ml_sharpe"],
            "carry_sharpe": r["carry_sharpe"],
            "n_bars": int(r["n_bars"]),
            "panel_rows": counts.get("panel_rows"),
            "panel_n_timestamps": counts.get("n_timestamps"),
        })


def compare_execution_modes(*, fetch: bool = False, run_1h: bool = True) -> pd.DataFrame:
    cfg8 = load_config(profile=PROFILE_8H)
    res8 = cfg8.artifact_dir("results", PROFILE_8H)
    rows: list = []

    exp8 = ExperimentSpec("8h", 5, profile=PROFILE_8H, rebalance_days=5)
    counts8 = split_row_counts(cfg8, exp8)

    canonical = res8 / CANONICAL_SUMMARY
    if fetch or not canonical.exists():
        from dual8h.pipeline.production_8h import run_production

        logger.info("Running canonical production (hybrid)...")
        run_production(cfg8, fetch=fetch)
    df_canon = pd.read_csv(canonical)
    canon_mode = str(df_canon["mode"].iloc[0]) if "mode" in df_canon.columns else HYBRID_MODE
    if canon_mode != HYBRID_MODE:
        logger.warning(
            "Canonical summary mode=%s; expected %s. Re-run: python -m dual8h production",
            canon_mode,
            HYBRID_MODE,
        )
    _append_rows(
        rows, df_canon,
        mode=HYBRID_MODE,
        interval="8h+1h",
        lag_label="1h (1 bar on 1h grid); 8h rebalance",
        rebalance_bars=15,
        horizon_bars=15,
        profile=PROFILE_8H,
        counts=counts8,
    )

    baseline_path = res8 / SENSITIVITY_8H_SUMMARY
    if fetch or not baseline_path.exists():
        logger.info("Running 8h panel baseline (sensitivity)...")
        run_production_8h_baseline(cfg8, fetch=fetch)
    df8 = pd.read_csv(baseline_path)
    _append_rows(
        rows, df8,
        mode=EXECUTION_MODE_8H_BASELINE,
        interval="8h",
        lag_label="8h (1 bar on 8h grid)",
        rebalance_bars=15,
        horizon_bars=15,
        profile=PROFILE_8H,
        counts=counts8,
    )

    counts1 = None
    if run_1h:
        cfg1 = load_config(profile=PROFILE_1H)
        summary_1h = cfg1.artifact_dir("results", PROFILE_1H, CANONICAL_SUMMARY)
        if fetch or not summary_1h.exists():
            from dual8h.pipeline.production_8h import run_production

            logger.info("Running full 1h panel production...")
            run_production(cfg1, fetch=fetch)
        df1 = pd.read_csv(summary_1h)
        exp1 = ExperimentSpec("1h", 5, profile=PROFILE_1H, rebalance_days=5)
        counts1 = split_row_counts(cfg1, exp1)
        reb = days_to_bars("1h", 5)
        _append_rows(
            rows, df1,
            mode="1h_full_features",
            interval="1h",
            lag_label="1h (1 bar); 1h features + 120-bar rebalance",
            rebalance_bars=reb,
            horizon_bars=reb,
            profile=PROFILE_1H,
            counts=counts1,
        )

    out_df = pd.DataFrame(rows)
    out_dir = cfg8.artifact_dir("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "execution_lag_comparison.csv"
    out_df.to_csv(out_csv, index=False)
    legacy = out_dir / "execution_lag_8h_vs_1h.csv"
    out_df.to_csv(legacy, index=False)

    payload = {
        "splits_calendar": cfg8.raw["splits"],
        "canonical_production": {
            "summary": str(canonical),
            "mode": HYBRID_MODE,
            "note": "Headline metrics use this file only; do not also report 8h_baseline as primary.",
        },
        "modes": {
            EXECUTION_MODE_8H_BASELINE: "Sensitivity: 8h panel + 8h simulator",
            HYBRID_MODE: "Canonical: 8h scores/rebalance; 1h returns; 1h execution lag",
            "1h_full_features": "Sensitivity: full 1h feature panel",
        },
        "counts_8h": counts8,
        "counts_1h": counts1,
        "comparison": rows,
    }
    out_json = out_dir / "execution_lag_comparison.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", out_csv)
    return out_df


def compare_8h_1h_execution(**kwargs) -> pd.DataFrame:
    """Backward-compatible alias."""
    return compare_execution_modes(**kwargs)
