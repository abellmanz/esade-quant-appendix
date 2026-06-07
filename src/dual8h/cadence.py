"""Cadence helpers: convert calendar days to bar counts per interval."""
from __future__ import annotations

from dataclasses import dataclass

# Binance spot kline intervals used in this project
BAR_HOURS = {"1h": 1.0, "8h": 8.0, "12h": 12.0, "1d": 24.0}
BARS_PER_YEAR = {k: (365.25 * 24) / h for k, h in BAR_HOURS.items()}
BARS_PER_DAY = {k: int(24 / h) for k, h in BAR_HOURS.items()}


def days_to_bars(interval: str, days: float) -> int:
    return max(int(round(days * BARS_PER_DAY[interval])), 1)


@dataclass(frozen=True)
class ExperimentSpec:
    interval: str
    target_days: int
    profile: str = "wide20"
    rebalance_days: int | None = None  # None => rebalance every target_days

    @property
    def effective_rebalance_days(self) -> int:
        return self.target_days if self.rebalance_days is None else self.rebalance_days

    @property
    def tag(self) -> str:
        base = f"{self.interval}_fwd{self.target_days}d"
        if self.rebalance_days is not None and self.rebalance_days != self.target_days:
            return f"{base}_reb{self.rebalance_days}d"
        return base

    def horizon_bars(self) -> int:
        return days_to_bars(self.interval, self.target_days)

    def rebalance_every_bars(self) -> int:
        return days_to_bars(self.interval, self.effective_rebalance_days)

    @property
    def target_col(self) -> str:
        return f"fwd_logret_{self.target_days}d"

    def rv_lookback_bars(self, calendar_days: int = 30) -> int:
        return days_to_bars(self.interval, calendar_days)

    def ema_spans(self, short_days: int = 16, long_days: int = 64) -> tuple[int, int]:
        return days_to_bars(self.interval, short_days), days_to_bars(self.interval, long_days)

    def mom_4h_bars(self) -> int:
        return max(int(round(4.0 / BAR_HOURS[self.interval])), 1)

    def mom_day_bars(self, days: int) -> int:
        return days_to_bars(self.interval, days)
