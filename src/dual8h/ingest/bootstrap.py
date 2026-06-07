"""Bootstrap importer: copy parquet from sibling repos into raw/ layout."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _workspace_root() -> Path:
    # .../binance_8h_dual_standalone/src/dual8h/ingest/bootstrap.py -> parents[4] = workspace
    return Path(__file__).resolve().parents[4]


def _sister_brc_root() -> Optional[Path]:
    """Sibling binance_ridge_carry project in the same workspace."""
    root = _workspace_root()
    dual8h = root / "binance_ridge_carry"
    raw = dual8h / "data" / "raw"
    if raw.is_dir():
        return dual8h
    return None


def _legacy_research_root() -> Optional[Path]:
    """Original BinanceDatabase research repo layout (1d resampled data)."""
    root = _workspace_root()
    if (root / "data").is_dir() and (root / "src" / "btc_pipeline").is_dir():
        return root
    return None


def bootstrap_from_brc_sibling(
    symbols: List[str],
    out_ohlcv: Path,
    out_funding: Path,
    *,
    intervals: tuple[str, ...] = ("8h",),
) -> bool:
    """Copy OHLCV + funding from ../binance_ridge_carry/data/raw/."""
    dual8h = _sister_brc_root()
    if dual8h is None:
        return False

    src_ohlcv = dual8h / "data" / "raw" / "ohlcv"
    src_funding = dual8h / "data" / "raw" / "funding"
    if not src_ohlcv.is_dir():
        return False

    out_ohlcv.mkdir(parents=True, exist_ok=True)
    out_funding.mkdir(parents=True, exist_ok=True)
    wrote = False

    for sym in symbols:
        for interval in intervals:
            src = src_ohlcv / f"{sym}_{interval}.parquet"
            if not src.exists():
                continue
            dst = out_ohlcv / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            logger.info("Bootstrapped %s from binance_ridge_carry", dst.name)
            wrote = True

        src_f = src_funding / f"{sym}_funding.parquet"
        if src_f.exists():
            dst_f = out_funding / src_f.name
            if not dst_f.exists():
                shutil.copy2(src_f, dst_f)
                logger.info("Bootstrapped %s funding from binance_ridge_carry", sym)

    return wrote


def bootstrap_from_legacy_research(
    symbols: List[str], out_ohlcv: Path, out_funding: Path
) -> bool:
    """Copy 1d OHLCV + funding from legacy research repo (not sufficient for 8h alone)."""
    import pandas as pd

    root = _legacy_research_root()
    if root is None:
        logger.info("Legacy research repo not found; skip bootstrap.")
        return False

    ohlcv_src = root / "data/Asset Data/clean/resampled data"
    funding_src = root / "data/Funding Rate Data/clean"
    if not ohlcv_src.is_dir():
        logger.warning("OHLCV source missing: %s", ohlcv_src)
        return False

    out_ohlcv.mkdir(parents=True, exist_ok=True)
    out_funding.mkdir(parents=True, exist_ok=True)
    wrote = False

    for sym in symbols:
        src = ohlcv_src / f"binance_{sym}_1d.parquet"
        if not src.exists():
            continue
        df = pd.read_parquet(src)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        out_path = out_ohlcv / f"{sym}_1d.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Bootstrapped %s OHLCV (1d only): %d rows", sym, len(df))
        wrote = True

        fp = funding_src / f"binance_{sym}_funding.parquet"
        if fp.exists():
            fdf = pd.read_parquet(fp)
            fdf["timestamp"] = pd.to_datetime(fdf["timestamp"], utc=True)
            fdf = fdf[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)
            fdf.to_parquet(out_funding / f"{sym}_funding.parquet", index=False)

    return wrote


def bootstrap_from_sister_repo(
    symbols: List[str],
    out_ohlcv: Path,
    out_funding: Path,
    sister_root: Optional[Path] = None,
    *,
    intervals: tuple[str, ...] = ("8h",),
) -> bool:
    """Try binance_ridge_carry sibling first, then legacy research repo."""
    if sister_root is not None:
        brc_raw = sister_root / "data" / "raw"
        if brc_raw.is_dir():
            return bootstrap_from_brc_sibling(symbols, out_ohlcv, out_funding, intervals=intervals)
    if bootstrap_from_brc_sibling(symbols, out_ohlcv, out_funding, intervals=intervals):
        return True
    return bootstrap_from_legacy_research(symbols, out_ohlcv, out_funding)
