"""Per-asset FFD with d_star selected by ADF stationarity test on TRAIN-only log_close.

Following Lopez de Prado: choose the smallest d in a grid such that the FFD series
is stationary (ADF p-value < threshold). Fallback to a fixed d if no d passes.

Leakage discipline: d_star is fit on the TRAIN slice of log_close, then APPLIED
unchanged to the full series for feature computation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

D_GRID: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
FFD_THRESH = 1e-4
ADF_P_THRESHOLD = 0.05
FALLBACK_D = 0.95


def compute_ffd_weights(d: float, tau: float = FFD_THRESH) -> np.ndarray:
    w = [1.0]; k = 1
    while True:
        w_k = w[-1] * (k - 1 - d) / k
        if abs(w_k) < tau: break
        w.append(w_k); k += 1
        if k > 5000: break
    return np.array(w, dtype=np.float64)


def apply_ffd(series: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = len(series); width = len(weights)
    result = np.full(n, np.nan)
    if n < width: return result
    conv = np.convolve(series, weights[::-1], mode="full")
    result[width - 1: n] = conv[width - 1: n]
    return result


def find_d_star_on_train(log_close_train: np.ndarray, d_grid: List[float] = D_GRID,
                          adf_p: float = ADF_P_THRESHOLD, fallback: float = FALLBACK_D) -> float:
    valid = log_close_train[np.isfinite(log_close_train)]
    if len(valid) < 100:
        return fallback
    for d in d_grid:
        w = compute_ffd_weights(d)
        ffd = apply_ffd(valid, w)
        ffd_v = ffd[np.isfinite(ffd)]
        if len(ffd_v) < 30: continue
        try:
            p = adfuller(ffd_v, autolag="AIC")[1]
        except Exception:
            continue
        if p < adf_p:
            return d
    return fallback
