"""Fetch Binance Spot OHLCV for a single symbol."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import httpx
import pandas as pd

from dual8h.cadence import BAR_HOURS

logger = logging.getLogger(__name__)

_KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "trade_count",
               "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"]

_INTERVAL_MS = {k: int(h * 3600 * 1000) for k, h in BAR_HOURS.items()}


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    base_url: str,
    chunk: int = 1000,
    sleep_s: float = 0.2,
) -> pd.DataFrame:
    """Fetch all klines for [start, end) at the given interval."""
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval {interval}")
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    step_ms = _INTERVAL_MS[interval]
    out_rows: List[List] = []
    cur = start_ts
    with httpx.Client(timeout=30.0) as client:
        while cur < end_ts:
            params = {"symbol": symbol, "interval": interval, "startTime": cur, "limit": chunk}
            r = client.get(f"{base_url}/api/v3/klines", params=params)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            out_rows.extend(batch)
            last_open = batch[-1][0]
            cur = last_open + step_ms
            if len(batch) < chunk:
                break
            time.sleep(sleep_s)
    df = pd.DataFrame(out_rows, columns=_KLINE_COLS)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(float)
    df["trade_count"] = df["trade_count"].astype(int)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume",
             "trade_count", "taker_buy_base_volume", "taker_buy_quote_volume"]]
    df = df[df["timestamp"] < pd.Timestamp(end, tz="UTC")]
    return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def write_ohlcv(df: pd.DataFrame, symbol: str, interval: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{interval}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def fetch_ohlcv_1d(symbol: str, start: str, end: str, base_url: str, **kw) -> pd.DataFrame:
    return fetch_ohlcv(symbol, "1d", start, end, base_url, **kw)
