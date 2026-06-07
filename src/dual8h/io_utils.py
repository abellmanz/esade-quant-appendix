"""Durable artifact writes: buffer then flush+fsync so a partial/truncated file
can never be left behind (e.g. on network/async-flush filesystems)."""
from __future__ import annotations

import io
import json as _json
import os
from pathlib import Path

import joblib
import pandas as pd


def write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as h:
        h.write(data); h.flush(); os.fsync(h.fileno())
    return path


def write_text(path: str | Path, text: str) -> Path:
    return write_bytes(path, text.encode("utf-8"))


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    buf = io.BytesIO(); df.to_parquet(buf, index=False)
    return write_bytes(path, buf.getvalue())


def write_json(obj, path: str | Path) -> Path:
    return write_text(path, _json.dumps(obj, indent=2, default=str))


def dump_joblib(obj, path: str | Path) -> Path:
    buf = io.BytesIO(); joblib.dump(obj, buf)
    return write_bytes(path, buf.getvalue())
