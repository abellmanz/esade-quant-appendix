# ESADE Quant Fund — Final Project Submission

**Student submission package** — Quant Appendix (full pipeline backtest)

> **SIMULATED BACKTEST ONLY** — No live or paper trading. All metrics are from historical simulation.

---

## For the instructor — start here

### Main deliverable

The complete analysis is in:

**`notebooks/ESADE_quant_appendix.ipynb`**

Please open this notebook and use **Run All**. The notebook is the final report, but it **requires this entire folder** to run (source code, data, and cached outputs are all included in this zip).

### How to run (3 steps)

1. **Unzip** this folder anywhere on your computer.
2. **Install Python 3.10 or newer** if not already available.
3. Open the notebook (Jupyter, VS Code, or Cursor) from this folder and click **Run All**.

The **first cell** of the notebook automatically installs all required Python libraries via `pip`. You should see `Package installed OK` before the rest of the pipeline runs.

If you prefer installing from the terminal first:

```bash
cd ESADE_quant_appendix_submission
pip install -e ".[notebook]"
jupyter notebook notebooks/ESADE_quant_appendix.ipynb
```

---

## Why is this zip ~150 MB?

Most of the file size is **pre-downloaded market data**, not source code.

| Contents | Approx. size | Purpose |
|----------|--------------|---------|
| `data/raw/ohlcv/` | ~87 MB | 8h and 1h OHLCV for 20 crypto symbols (Binance) |
| `data/raw/funding/` | ~2 MB | Perpetual funding rates for 20 symbols |
| `data/experiments/` | ~53 MB | Pre-built feature panels and train/val/test splits |
| `artifacts/` | ~1 MB | Frozen CatBoost model, leaderboards, back-test results |
| `src/` + notebook | <1 MB | Python pipeline code |

**Why include the data?** The notebook can download everything from Binance APIs (`DATA_SOURCE = "download"`), but a full fresh download takes **2+ hours**. The bundled data lets you reproduce results immediately with the default setting `DATA_SOURCE = "bundled"`.

**What is NOT in this zip (and not needed):** Python virtual environments (`.venv/`), temporary CatBoost logs, or backup copies of data. Those are recreated locally when libraries are installed.

---

## Required Python libraries

Libraries are **not** bundled in the zip. They are installed automatically by the notebook's first cell, or manually via `pip install -e ".[notebook]"`.

| Library | Role |
|---------|------|
| Python 3.10+ | Runtime |
| numpy, pandas, pyarrow | Data handling |
| scikit-learn, catboost, joblib | Machine learning |
| statsmodels | Statistical tests |
| matplotlib | Charts in the notebook |
| httpx, PyYAML | Data ingest and configuration |
| jupyter / ipykernel | Running the notebook (if using classic Jupyter) |

After installation, roughly **1–1.5 GB of disk space** may be used locally for the Python packages (similar to any data-science project). This is normal and happens on the reviewer's machine, not inside the zip.

---

## What is included in this submission

```
ESADE_quant_appendix_submission/
├── README.md                          ← this file
├── config.yaml                        ← project parameters
├── pyproject.toml                     ← dependencies list
├── notebooks/
│   └── ESADE_quant_appendix.ipynb     ← MAIN NOTEBOOK (final deliverable)
├── src/dual8h/                        ← pipeline source code
├── data/
│   ├── raw/                           ← bundled market data
│   └── experiments/                   ← pre-built panels & splits
└── artifacts/                         ← frozen model & cached results
```

Everything listed above is **required** for the notebook to run with default settings.

---

## Notebook settings (defaults are submission-ready)

| Flag | Default | Meaning |
|------|---------|---------|
| `DATA_SOURCE` | `"bundled"` | Use included parquet data (fast) |
| `RUN_SELECTION_SWEEP` | `False` | Use cached cadence leaderboard |
| `RETRAIN_PRODUCTION_MODEL` | `False` | Use frozen CatBoost model |
| `FORCE_REBUILD_PANEL` | `False` | Reuse pre-built panels |

No changes are needed for a standard review run.

---

## GitHub mirror

An identical copy of this submission is available at:

**https://github.com/abellmanz/esade-quant-appendix**

---

## AI usage disclosure

AI coding assistants in **Cursor** were used during development of this project. All modeling choices, back-test design, quantitative results, and final submission content were reviewed and validated by the author.

| Tool | Model | Typical use |
|------|-------|-------------|
| Cursor | **Claude Opus 4.7** | Pipeline structure, feature engineering, debugging |
| Cursor | **Claude Opus 4.8** | Refactoring, documentation, submission packaging |
| Cursor | **Composer 2.5** | Rapid iteration on scripts, notebook cells, and README |

The author remains solely responsible for the methodology, analysis, and reported metrics.

---

## Data sources

- **Crypto OHLCV & funding rates:** Binance public APIs (historical)
- **S&P 500 benchmark:** Yahoo Finance (`^GSPC` daily closes), file `data/raw/benchmark/SPX_daily.csv`
