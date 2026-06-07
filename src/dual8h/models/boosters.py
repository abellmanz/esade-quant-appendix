"""Gradient boosting trainers with purged k-fold rank-IC selection (train only)."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from dual8h.models.common import cv_rank_ic, prepare_xy


def fit_ridge_purged(
    train_panel: pd.DataFrame,
    features: List[str],
    target: str,
    alphas: List[float],
    n_splits: int,
    purge_days: int,
    embargo_pct: float,
) -> Tuple[Ridge, Dict[str, Any]]:
    z, y, X, timestamps = prepare_xy(train_panel, features, target)
    cv_rows = []
    best = {"alpha": alphas[0], "mean_rank_ic": float("-inf")}

    for alpha in alphas:
        def fp(train_m, test_m):
            m = Ridge(alpha=alpha, random_state=0)
            m.fit(X[train_m], y.values[train_m])
            return m.predict(X[test_m])

        mean_ic = cv_rank_ic(X, y.values, timestamps, n_splits, purge_days, embargo_pct, fp)
        cv_rows.append({"alpha": alpha, "mean_rank_ic": mean_ic})
        if mean_ic > best["mean_rank_ic"]:
            best = {"alpha": alpha, "mean_rank_ic": mean_ic}

    model = Ridge(alpha=best["alpha"], random_state=0)
    model.fit(X, y.values)
    return model, {
        "model_type": "ridge",
        "alpha": best["alpha"],
        "cv_results": cv_rows,
        "mean_rank_ic": best["mean_rank_ic"],
        "n_train": len(X),
        "features": list(features),
        "target": target,
        "coefficients": dict(zip(features, model.coef_.tolist())),
        "intercept": float(model.intercept_),
    }


def fit_lightgbm_purged(
    train_panel: pd.DataFrame,
    features: List[str],
    target: str,
    n_estimators_grid: List[int],
    max_depth: int,
    learning_rate: float,
    min_data_in_leaf: int,
    n_splits: int,
    purge_days: int,
    embargo_pct: float,
) -> Tuple[Any, Dict[str, Any]]:
    import lightgbm as lgb

    z, y, X, timestamps = prepare_xy(train_panel, features, target)
    cv_rows = []
    best = {"n_estimators": n_estimators_grid[0], "mean_rank_ic": float("-inf")}

    for n_est in n_estimators_grid:
        def fp(train_m, test_m):
            m = lgb.LGBMRegressor(
                n_estimators=n_est,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=max(2 ** max_depth - 1, 2),
                min_data_in_leaf=min_data_in_leaf,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            m.fit(X[train_m], y.values[train_m])
            return m.predict(X[test_m])

        mean_ic = cv_rank_ic(X, y.values, timestamps, n_splits, purge_days, embargo_pct, fp)
        cv_rows.append({"n_estimators": n_est, "mean_rank_ic": mean_ic})
        if mean_ic > best["mean_rank_ic"]:
            best = {"n_estimators": n_est, "mean_rank_ic": mean_ic}

    model = lgb.LGBMRegressor(
        n_estimators=best["n_estimators"],
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=max(2 ** max_depth - 1, 2),
        min_data_in_leaf=min_data_in_leaf,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X, y.values)
    return model, {
        "model_type": "lightgbm",
        "n_estimators": best["n_estimators"],
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "cv_results": cv_rows,
        "mean_rank_ic": best["mean_rank_ic"],
        "n_train": len(X),
        "features": list(features),
        "target": target,
        "feature_importance": dict(zip(features, model.feature_importances_.tolist())),
    }


def fit_xgboost_purged(
    train_panel: pd.DataFrame,
    features: List[str],
    target: str,
    n_estimators_grid: List[int],
    max_depth: int,
    learning_rate: float,
    min_child_weight: int,
    n_splits: int,
    purge_days: int,
    embargo_pct: float,
) -> Tuple[Any, Dict[str, Any]]:
    import xgboost as xgb

    z, y, X, timestamps = prepare_xy(train_panel, features, target)
    cv_rows = []
    best = {"n_estimators": n_estimators_grid[0], "mean_rank_ic": float("-inf")}

    for n_est in n_estimators_grid:
        def fp(train_m, test_m):
            m = xgb.XGBRegressor(
                n_estimators=n_est,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            m.fit(X[train_m], y.values[train_m])
            return m.predict(X[test_m])

        mean_ic = cv_rank_ic(X, y.values, timestamps, n_splits, purge_days, embargo_pct, fp)
        cv_rows.append({"n_estimators": n_est, "mean_rank_ic": mean_ic})
        if mean_ic > best["mean_rank_ic"]:
            best = {"n_estimators": n_est, "mean_rank_ic": mean_ic}

    model = xgb.XGBRegressor(
        n_estimators=best["n_estimators"],
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y.values)
    return model, {
        "model_type": "xgboost",
        "n_estimators": best["n_estimators"],
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "min_child_weight": min_child_weight,
        "cv_results": cv_rows,
        "mean_rank_ic": best["mean_rank_ic"],
        "n_train": len(X),
        "features": list(features),
        "target": target,
        "feature_importance": dict(zip(features, model.feature_importances_.tolist())),
    }


def fit_catboost_purged(
    train_panel: pd.DataFrame,
    features: List[str],
    target: str,
    iterations_grid: List[int],
    depth: int,
    learning_rate: float,
    min_data_in_leaf: int,
    n_splits: int,
    purge_days: int,
    embargo_pct: float,
    seed: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    from catboost import CatBoostRegressor

    z, y, X, timestamps = prepare_xy(train_panel, features, target)
    cv_rows = []
    best = {"iterations": iterations_grid[0], "mean_rank_ic": float("-inf")}

    for iters in iterations_grid:
        def fp(train_m, test_m):
            m = CatBoostRegressor(
                iterations=iters,
                depth=depth,
                learning_rate=learning_rate,
                min_data_in_leaf=min_data_in_leaf,
                subsample=0.8,
                random_seed=seed,
                verbose=False,
            )
            m.fit(X[train_m], y.values[train_m])
            return m.predict(X[test_m])

        mean_ic = cv_rank_ic(X, y.values, timestamps, n_splits, purge_days, embargo_pct, fp)
        cv_rows.append({"iterations": iters, "mean_rank_ic": mean_ic})
        if mean_ic > best["mean_rank_ic"]:
            best = {"iterations": iters, "mean_rank_ic": mean_ic}

    model = CatBoostRegressor(
        iterations=best["iterations"],
        depth=depth,
        learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf,
        subsample=0.8,
        random_seed=seed,
        verbose=False,
    )
    model.fit(X, y.values)
    return model, {
        "model_type": "catboost",
        "iterations": best["iterations"],
        "depth": depth,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "cv_results": cv_rows,
        "mean_rank_ic": best["mean_rank_ic"],
        "n_train": len(X),
        "features": list(features),
        "target": target,
        "feature_importance": dict(zip(features, model.feature_importances_.tolist())),
    }
