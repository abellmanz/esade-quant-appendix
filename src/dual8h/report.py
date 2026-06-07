"""Write human-readable run outputs under output/."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dual8h.config import Config


def output_dir(cfg: Config) -> Path:
    d = cfg.repo_root / "output"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_run_report(
    cfg: Config,
    summary: pd.DataFrame,
    *,
    leakage_passed: bool | None = None,
    leakage_detail: str = "",
) -> Path:
    out = output_dir(cfg)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(run_dir / "summary.csv", index=False)
    summary.to_json(run_dir / "summary.json", orient="records", indent=2)

    manifest_src = cfg.artifact_dir("results", cfg.profile) / "production_manifest.json"
    mode = "8h_decide_1h_exec"
    pnl_grid = "1h"
    if manifest_src.exists():
        manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        mode = manifest.get("mode", mode)
        lc = manifest.get("leakage_controls", {})
        pnl_grid = lc.get("pnl_simulation", pnl_grid)

    lines = [
        "# wide20_8h_dual run report",
        "",
        f"- Profile: `{cfg.profile}`",
        f"- Execution mode: `{mode}`",
        f"- Decisions: 8h features, 5d rebalance (15 bars)",
        f"- PnL grid: {pnl_grid} (1-bar execution lag)",
        f"- Sleeve: 35% ML / 65% carry",
        f"- Symbols: {len(cfg.symbols)}",
        f"- Generated: {ts} UTC",
        "",
        "## Performance",
        "",
        "| Split | Sleeve Sharpe | ML | Carry | Max DD | Calmar | n_bars |",
        "|-------|---------------|-----|-------|--------|--------|--------|",
    ]
    for _, row in summary.iterrows():
        nb = int(row.get("n_bars", 0))
        lines.append(
            f"| {row['split']} | {row['sleeve_sharpe']:.3f} | "
            f"{row['ml_sharpe']:.3f} | {row['carry_sharpe']:.3f} | "
            f"{row['sleeve_max_dd_log']:.1%} | {row.get('sleeve_calmar', float('nan')):.2f} | {nb} |"
        )

    if leakage_passed is not None:
        lines.extend(["", "## Leakage verify", ""])
        status = "PASS" if leakage_passed else "FAIL"
        lines.append(f"- Status: **{status}**")
        if leakage_detail:
            lines.append(f"- Detail: {leakage_detail}")

    md_path = run_dir / "report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    latest = out / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink(missing_ok=True)
    try:
        latest.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        (out / "LATEST.txt").write_text(str(run_dir), encoding="utf-8")

    return run_dir
