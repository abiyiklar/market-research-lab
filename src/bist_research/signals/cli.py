from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from bist_research.config import INITIAL_BIST_PANEL
from bist_research.symbols import panel_configs

from .pipeline import run_signal_pipeline
from .scoring import MINIMUM_CROSS_SECTION


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build transparent cross-sectional scores and causal signal diagnostics."
    )
    parser.add_argument("--universe", choices=[INITIAL_BIST_PANEL], help="Configured research universe.")
    parser.add_argument("--symbols", nargs="+", help="Optional symbol override.")
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("data/features/equities"),
        help="Directory containing per-symbol feature folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/signals"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/score_signal_research.md"),
    )
    parser.add_argument(
        "--minimum-cross-section",
        type=int,
        default=MINIMUM_CROSS_SECTION,
        help="Minimum valid same-date symbols required for scoring.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configs = panel_configs(args.universe, args.symbols)
    try:
        outputs = run_signal_pipeline(
            configs,
            feature_dir=args.feature_dir,
            output_dir=args.output_dir,
            report_path=args.report_path,
            minimum_cross_section=args.minimum_cross_section,
        )
    except Exception as exc:
        print(f"Signal pipeline failed: {exc}")
        return 1

    latest = outputs.latest_score_date.date() if outputs.latest_score_date is not None else "none"
    print(f"Loaded symbols: {len(outputs.loaded_symbols)}")
    print(f"Missing symbols: {', '.join(outputs.missing_symbols) or 'none'}")
    print(f"Latest scored date: {latest}")
    print(f"BUY events: {outputs.buy_event_count}")
    print(f"SELL events: {outputs.sell_event_count}")
    print(f"Panel scores: {outputs.panel_parquet}")
    print(f"Report: {outputs.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
