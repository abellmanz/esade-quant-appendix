"""Load config.yaml into a typed namespace. Single source of truth for constants."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as h:
        return yaml.safe_load(h)


@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]
    repo_root: Path
    profile: str

    @property
    def symbols(self) -> List[str]:
        profiles = self.raw.get("profiles", {})
        if self.profile in profiles:
            return list(profiles[self.profile]["symbols"])
        return list(self.raw["universe"]["symbols"])

    @property
    def n_long(self) -> int:
        profiles = self.raw.get("profiles", {})
        if self.profile in profiles:
            return int(profiles[self.profile]["n_long"])
        return int(self.raw["strategy"]["n_long"])

    @property
    def n_short(self) -> int:
        profiles = self.raw.get("profiles", {})
        if self.profile in profiles:
            return int(profiles[self.profile]["n_short"])
        return int(self.raw["strategy"]["n_short"])

    @property
    def profile_block(self) -> Dict[str, Any]:
        return self.raw["profiles"][self.profile]

    @property
    def cadence(self) -> Dict[str, Any]:
        return dict(self.profile_block.get("cadence", {}))

    @property
    def interval(self) -> str:
        return str(self.cadence.get("interval", self.raw["ingest"]["interval"]))

    @property
    def start(self) -> str:
        return self.raw["ingest"]["start"]

    @property
    def end(self) -> str:
        return self.raw["ingest"]["end"]

    @property
    def model_features(self) -> List[str]:
        return list(self.raw["model"]["features"])

    @property
    def alpha(self) -> float:
        return float(self.raw["model"]["alpha"])

    @property
    def target_col(self) -> str:
        if self.cadence:
            return f"fwd_logret_{int(self.cadence['target_days'])}d"
        return f"fwd_logret_{self.forward_horizon_bars}d"

    @property
    def purge_days(self) -> int:
        return int(self.raw["splits"]["purge_days"])

    @property
    def forward_horizon_bars(self) -> int:
        if self.cadence:
            from dual8h.cadence import days_to_bars
            return days_to_bars(self.interval, int(self.cadence["target_days"]))
        return int(self.raw["target"]["forward_horizon_bars"])

    @property
    def rebalance_every_bars(self) -> int:
        if self.cadence:
            from dual8h.cadence import days_to_bars
            reb = int(self.cadence.get("rebalance_days", self.cadence["target_days"]))
            return days_to_bars(self.interval, reb)
        return int(self.raw["strategy"]["rebalance_every_bars"])

    @property
    def require_common_history(self) -> bool:
        return bool(self.raw.get("panel", {}).get("require_common_history", False))

    @property
    def sleeve_weights(self) -> Tuple[float, float]:
        sw = self.profile_block.get("sleeve_weights") or self.raw["strategy"]["sleeve_weights"]
        return float(sw["ml"]), float(sw["carry"])

    def data_dir(self, *parts: str) -> Path:
        return self.repo_root.joinpath("data", *parts)

    def artifact_dir(self, *parts: str) -> Path:
        return self.repo_root.joinpath("artifacts", *parts)


def all_profile_symbols(raw: Dict[str, Any]) -> List[str]:
    syms: set[str] = set()
    for prof in raw.get("profiles", {}).values():
        syms.update(prof["symbols"])
    if not syms:
        syms.update(raw.get("universe", {}).get("symbols", []))
    return sorted(syms)


def load_config(path: Path | None = None, profile: str | None = None) -> Config:
    cfg_path = path or (_REPO_ROOT / "config.yaml")
    raw = _load_yaml(cfg_path)
    prof = profile or raw.get("active_profile", "wide20")
    return Config(raw=raw, repo_root=_REPO_ROOT, profile=prof)
