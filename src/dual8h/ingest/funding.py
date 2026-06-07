"""Fetch USDT-M perpetual funding rate history via /fapi/v1/fundingRate."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_funding(symbol: str, start: str, end: str, base_url: str,
                  chunk: int = 1000, sleep_s: float = 0.2) -> pd.DataFrame:
    """Fetch all funding rate events for [start, end). 8h native cadence."""
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    rows: List[dict] = []
    cur = start_ts
    with httpx.Client(timeout=30.0) as client:
        while cur < end_ts:
            params = {"symbol": symbol, "startTime": cur, "endTime": end_ts, "limit": chunk}
            r = client.get(f"{base_url}/fapi/v1/fundingRate", params=params)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows.extend(batch)
            last_time = batch[-1]["fundingTime"]
            cur = last_time + 1
            if len(batch) < chunk:
                break
            time.sleep(sleep_s)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def write_funding(df: pd.DataFrame, symbol: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_funding.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path
