"""Performance metrics and block bootstrap CI on bar PnL."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def ann_sharpe(rets: pd.Series, bars_per_year: float) -> float:
    sd = float(rets.std(ddof=1))
    return float("nan") if sd <= 1e-12 else float(rets.mean() / sd * np.sqrt(bars_per_year))


def max_drawdown_log(rets: pd.Series) -> float:
    cum = rets.cumsum()
    return float((cum - cum.cummax()).min())


def calmar_log(rets: pd.Series) -> float:
    dd = max_drawdown_log(rets)
    return float(rets.sum() / abs(dd)) if dd < 0 else float("nan")


def block_bootstrap_sharpe(
    rets: np.ndarray,
    block_size: int,
    n_iter: int,
    bars_per_year: float,
    seed: int = 42,
) -> Dict[str, Any]:
    """Block bootstrap on a 1d bar net-return series."""
    rng = np.random.default_rng(seed)
    n = len(rets)
    if n < block_size + 1:
        return {"error": "series too short for bootstrap"}
    sharpes = []
    for _ in range(n_iter):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, max(n - block_size, 1)))
            idx.extend(range(start, min(start + block_size, n)))
        sample = rets[np.array(idx[:n])]
        sd = sample.std(ddof=1)
        if sd > 1e-12:
            sharpes.append(float(sample.mean() / sd * np.sqrt(bars_per_year)))
    arr = np.array(sharpes)
    return {
        "n_iterations": n_iter,
        "block_size_bars": block_size,
        "sharpe_median": float(np.median(arr)),
        "sharpe_ci_95_low": float(np.quantile(arr, 0.025)),
        "sharpe_ci_95_high": float(np.quantile(arr, 0.975)),
        "p_sharpe_gt_1": float((arr > 1.0).mean()),
        "p_sharpe_gt_0": float((arr > 0.0).mean()),
    }
