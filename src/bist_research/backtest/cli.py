from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .models import BacktestConfig
from .report import run_backtest_pipeline


def configure_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bist_research.backtest")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "backtest.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TUPRS Baseline V1 daily backtest.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/features/tuprs_features.parquet"),
        help="Input feature Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backtest"),
        help="Directory for backtest CSV outputs.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for Markdown and chart outputs.",
    )
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--commission-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging()
    try:
        config = BacktestConfig(
            initial_capital=args.initial_capital,
            commission_rate=args.commission_rate,
            slippage_rate=args.slippage_rate,
        )
        result = run_backtest_pipeline(
            input_path=args.input,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            config=config,
            logger=logger,
        )
    except Exception:
        logger.exception("Backtest failed")
        return 1

    all_benchmarks = result.benchmark_table.loc[result.benchmark_table["period"].eq("all")]
    print("Baseline V1 backtest completed")
    print(f"Trades: {result.metrics['total_trades']}")
    print(f"Total return: {float(result.metrics['total_return']) * 100:.2f}%")
    print(f"Maximum drawdown: {float(result.metrics['maximum_drawdown']) * 100:.2f}%")
    for row in all_benchmarks.itertuples(index=False):
        print(f"{row.benchmark}: {row.total_return * 100:.2f}%")
    print(f"Report: {result.artifacts.markdown_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
