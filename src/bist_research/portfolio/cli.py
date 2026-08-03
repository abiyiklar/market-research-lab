from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from bist_research.config import INITIAL_BIST_PANEL
from bist_research.symbols import panel_configs

from .models import PortfolioConfig
from .pipeline import run_portfolio_pipeline


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the causal long-only BIST panel portfolio backtest."
    )
    parser.add_argument("--universe", choices=[INITIAL_BIST_PANEL])
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=Path("data/signals/panel_daily_scores.parquet"),
    )
    parser.add_argument(
        "--feature-dir", type=Path, default=Path("data/features/equities")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/portfolio"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/portfolio_backtest_research.md"),
    )
    parser.add_argument("--starting-capital", type=float, default=1_000_000.0)
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument("--cost-rate", type=float, default=0.0015)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configs = panel_configs(args.universe, args.symbols)
    config = PortfolioConfig(
        starting_capital=args.starting_capital,
        max_positions=args.max_positions,
        transaction_cost_rate=args.cost_rate,
    )
    try:
        outputs = run_portfolio_pipeline(
            configs,
            panel_path=args.panel_path,
            feature_dir=args.feature_dir,
            output_dir=args.output_dir,
            report_path=args.report_path,
            config=config,
        )
    except Exception as exc:
        print(f"Portfolio pipeline failed: {exc}")
        return 1
    primary = outputs.performance_summary.loc[
        outputs.performance_summary["portfolio"].eq("primary_portfolio")
    ].iloc[0]
    print(f"Backtest dates: {outputs.start_date.date()} to {outputs.end_date.date()}")
    print(f"Completed trades: {outputs.completed_trades}")
    print(f"Open positions at end: {outputs.open_positions}")
    print(f"Ending value: TRY {primary['ending_value']:,.2f}")
    print(f"Total return: {primary['total_return']:.2%}")
    print(f"Report: {outputs.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
