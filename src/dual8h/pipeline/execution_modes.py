"""Canonical production execution modes (leakage-safe labeling)."""
from __future__ import annotations

EXECUTION_MODE_HYBRID = "8h_decide_1h_exec"
EXECUTION_MODE_8H_BASELINE = "8h_baseline"
EXECUTION_MODE_1H_FULL = "1h_full_features"

CANONICAL_SUMMARY = "production_summary.csv"
CANONICAL_MANIFEST = "production_manifest.json"
CANONICAL_TEST_BARS = "test_catboost_reg_bars.parquet"
SENSITIVITY_8H_SUMMARY = "production_summary_8h_baseline.csv"
