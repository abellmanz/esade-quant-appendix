"""Shared production CatBoost training (used by production + selection sweep)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from dual8h.config import Config
from dual8h.models.boosters import fit_catboost_purged


def catboost_params(cfg: Config) -> Dict[str, Any]:
    cb_cfg = cfg.profile_block.get("catboost", {})
    cv = cfg.raw["cv"]
    ml = cfg.raw.get("ml", {})
    depth = int(cb_cfg.get("depth", 2))
    min_leaf = int(cb_cfg.get("min_data_in_leaf", max(150, 10 * len(cfg.symbols))))
    iters = [int(x) for x in cb_cfg.get("iterations_grid", ml.get("n_estimators_grid", [100, 200]))]
    return {
        "iterations_grid": iters,
        "depth": depth,
        "learning_rate": float(cb_cfg.get("learning_rate", ml.get("learning_rate", 0.05))),
        "min_data_in_leaf": min_leaf,
        "n_splits": int(cv["n_splits"]),
        "purge_days": int(cv["purge_days"]),
        "embargo_pct": float(cv["embargo_pct"]),
    }


def train_reg_catboost(
    cfg: Config,
    train: pd.DataFrame,
    target: str,
    features: List[str] | None = None,
) -> Tuple[Any, Dict]:
    feats = features or cfg.model_features
    p = catboost_params(cfg)
    seed = int(cfg.raw.get("random_seed", 42))
    model, info = fit_catboost_purged(
        train, feats, target,
        iterations_grid=p["iterations_grid"],
        depth=p["depth"],
        learning_rate=p["learning_rate"],
        min_data_in_leaf=p["min_data_in_leaf"],
        n_splits=p["n_splits"],
        purge_days=p["purge_days"],
        embargo_pct=p["embargo_pct"],
        seed=seed,
    )
    info["random_seed"] = seed
    info["depth"] = p["depth"]
    info["min_data_in_leaf"] = p["min_data_in_leaf"]
    return model, info
