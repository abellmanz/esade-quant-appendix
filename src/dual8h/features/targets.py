"""Forward target for training. NEVER consumed by the simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_target(
    panel: pd.DataFrame,
    horizon_bars: int = 7,
    target_col: str | None = None,
) -> pd.DataFrame:
    """Forward log return over horizon_bars. Column name defaults to fwd_logret_{horizon_bars}d."""
    col = target_col or f"fwd_logret_{horizon_bars}d"
    out = panel.copy().sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(drop=True)
    out[col] = (
        out.groupby("symbol", sort=False)["close"]
           .transform(lambda s: np.log(s.shift(-horizon_bars) / s))
    )
    return out.sort_values(["timestamp", "symbol"], kind="mergesort").reset_index(drop=True)
