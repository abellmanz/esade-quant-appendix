"""dual8h ingest: bootstrap from research repo if present, else fetch from Binance."""
from __future__ import annotations

import logging

from dual8h.config import Config, all_profile_symbols
from dual8h.ingest.bootstrap import bootstrap_from_sister_repo
from dual8h.ingest.funding import fetch_funding, write_funding
from dual8h.ingest.ohlcv import fetch_ohlcv, write_ohlcv

logger = logging.getLogger(__name__)


def run_ingest(cfg: Config, force_fetch: bool = False) -> None:
    ohlcv_dir = cfg.data_dir("raw", "ohlcv")
    funding_dir = cfg.data_dir("raw", "funding")
    all_syms = all_profile_symbols(cfg.raw)

    intervals = tuple(cfg.raw.get("ingest", {}).get("intervals", ["8h", "1h"]))
    if not force_fetch:
        bootstrap_from_sister_repo(all_syms, ohlcv_dir, funding_dir, intervals=intervals)
    spot = cfg.raw["ingest"]["binance_spot_base"]
    perp = cfg.raw["ingest"]["binance_perp_base"]
    fetched = 0
    for sym in all_syms:
        for interval in intervals:
            path = ohlcv_dir / f"{sym}_{interval}.parquet"
            if path.exists() and not force_fetch:
                continue
            ohlcv = fetch_ohlcv(sym, interval, cfg.start, cfg.end, spot)
            write_ohlcv(ohlcv, sym, interval, ohlcv_dir)
            logger.info("%s %s OHLCV: %d bars", sym, interval, len(ohlcv))
            fetched += 1
        fp = funding_dir / f"{sym}_funding.parquet"
        if not fp.exists() or force_fetch:
            write_funding(fetch_funding(sym, cfg.start, cfg.end, perp), sym, funding_dir)
    logger.info("Ingest complete (%d OHLCV series written/updated).", fetched)
