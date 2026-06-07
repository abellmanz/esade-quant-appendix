"""CLI for binance_8h_dual_standalone."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from dual8h.config import load_config

PROFILE = "wide20_8h_dual"


def _run_leakage_tests() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/"],
        capture_output=True,
        text=True,
    )
    detail = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(detail.strip().splitlines()[-3:])
    return proc.returncode == 0, tail


def cmd_run_all(*, fetch: bool, skip_verify: bool) -> int:
    cfg = load_config(profile=PROFILE)
    logging.info("=== run-all: %s (%d symbols) ===", PROFILE, len(cfg.symbols))

    from dual8h.ingest.run import run_ingest
    from dual8h.pipeline.production_8h import run_production
    from dual8h.report import write_run_report

    run_ingest(cfg, force_fetch=fetch)
    summary = run_production(cfg, fetch=False)

    print("\n--- Production summary ---")
    print(summary[["split", "sleeve_sharpe", "ml_sharpe", "carry_sharpe"]].to_string(index=False))

    leakage_passed = None
    leakage_detail = ""
    if not skip_verify:
        logging.info("Running leakage test battery...")
        leakage_passed, leakage_detail = _run_leakage_tests()
        status = "PASS" if leakage_passed else "FAIL"
        print(f"\n--- Leakage verify: {status} ---")
        print(leakage_detail)

    run_dir = write_run_report(
        cfg, summary, leakage_passed=leakage_passed, leakage_detail=leakage_detail,
    )
    print(f"\nReport written to: {run_dir}")
    return 1 if leakage_passed is False else 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(
        prog="dual8h",
        description="Standalone 8h dual book (wide20_8h_dual): ingest → panel → production → report",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Fetch 8h OHLCV + funding (or bootstrap from sibling repo)")
    ing.add_argument("--fetch", action="store_true", help="Force re-fetch from Binance API")

    prod = sub.add_parser(
        "production",
        help="Canonical production (default: 8h_decide_1h_exec for wide20_8h_dual)",
    )
    prod.add_argument(
        "--profile",
        default=PROFILE,
        help="Config profile (wide20_8h_dual or wide20_1h_dual)",
    )
    prod.add_argument("--fetch", action="store_true", help="Re-ingest before production")
    prod.add_argument(
        "--baseline-8h",
        action="store_true",
        help="8h panel simulator only (sensitivity; does not overwrite canonical summary)",
    )
    prod.add_argument("--retrain", action="store_true", help="Refit CatBoost on train split")

    ph = sub.add_parser(
        "production-hybrid",
        help="Alias for production on wide20_8h_dual (8h decide / 1h exec)",
    )
    ph.add_argument("--fetch", action="store_true")
    ph.add_argument("--retrain", action="store_true")

    ce = sub.add_parser(
        "compare-execution",
        help="Compare 8h vs full-1h vs hybrid execution modes",
    )
    ce.add_argument("--fetch", action="store_true", help="Re-ingest and re-run all profiles")
    sub.add_parser("verify-leakage", help="Run pytest leakage battery")

    ss = sub.add_parser(
        "selection-sweep",
        help="Focused cadence sweep (val-ranked); writes selection_leaderboard.csv",
    )
    ss.add_argument(
        "--horizons-only",
        action="store_true",
        help="Only 8h_fwd1d..7d (skip cross-timeframe contrast rows)",
    )
    ss.add_argument(
        "--no-cross-tf",
        action="store_true",
        help="Skip 12h/1d contrast rows (8h horizons only)",
    )

    ra = sub.add_parser("run-all", help="ingest → production → report → verify-leakage")
    ra.add_argument("--fetch", action="store_true", help="Force Binance API fetch")
    ra.add_argument("--skip-verify", action="store_true", help="Skip leakage pytest step")

    args = p.parse_args()
    cfg = load_config(profile=PROFILE)

    if args.cmd == "ingest":
        from dual8h.ingest.run import run_ingest
        run_ingest(cfg, force_fetch=args.fetch)
        return

    if args.cmd == "production":
        from dual8h.pipeline.production_8h import run_production, run_production_8h_baseline

        prof = getattr(args, "profile", PROFILE)
        cfg_p = load_config(profile=prof)
        if getattr(args, "baseline_8h", False):
            df = run_production_8h_baseline(
                cfg_p, fetch=getattr(args, "fetch", False), retrain=getattr(args, "retrain", False),
            )
        else:
            df = run_production(
                cfg_p,
                fetch=getattr(args, "fetch", False),
                retrain=getattr(args, "retrain", False),
            )
        cols = ["split", "sleeve_sharpe", "ml_sharpe", "carry_sharpe", "n_bars"]
        cols = [c for c in cols if c in df.columns]
        print(df[cols].to_string(index=False))
        return

    if args.cmd == "production-hybrid":
        from dual8h.pipeline.production_hybrid import run_production_hybrid

        df = run_production_hybrid(
            fetch=getattr(args, "fetch", False),
            retrain=getattr(args, "retrain", False),
        )
        print(df[["split", "sleeve_sharpe", "ml_sharpe", "carry_sharpe", "n_bars"]].to_string(index=False))
        return

    if args.cmd == "compare-execution":
        from dual8h.pipeline.compare_execution import compare_execution_modes

        df = compare_execution_modes(fetch=args.fetch)
        print("\n--- Execution mode comparison ---")
        print(df.to_string(index=False))
        return

    if args.cmd == "verify-leakage":
        ok, detail = _run_leakage_tests()
        print(detail)
        sys.exit(0 if ok else 1)

    if args.cmd == "selection-sweep":
        from dual8h.experiments.granularity_sweep import (
            PRODUCTION_TAG,
            pick_best,
            run_selection_sweep,
        )

        df = run_selection_sweep(
            cfg,
            horizons_only=args.horizons_only,
            include_cross_tf=not args.no_cross_tf,
        )
        best = pick_best(df)
        print("\n--- Selection leaderboard (validation rank) ---")
        cols = ["tag", "validation_sleeve_sharpe", "test_sleeve_sharpe", "production_tag"]
        print(df[cols].to_string(index=False))
        print(f"\nBest on validation: {best['tag']} (Sharpe={best['validation_sleeve_sharpe']:.3f})")
        print(f"Production spec tag: {PRODUCTION_TAG}")
        return

    if args.cmd == "run-all":
        sys.exit(cmd_run_all(fetch=args.fetch, skip_verify=args.skip_verify))


if __name__ == "__main__":
    main()
