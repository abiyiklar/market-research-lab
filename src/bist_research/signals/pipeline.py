from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from bist_research.config import SymbolConfig
from bist_research.symbols import safe_symbol_name

from .report import build_score_signal_report, write_score_signal_report
from .rules import add_signal_rules
from .scoring import MINIMUM_CROSS_SECTION, build_score_panel
from .study import (
    build_buy_event_study,
    build_sell_event_study,
    build_signal_quality_summary,
)


CURRENT_SIGNAL_COLUMNS = [
    "date",
    "symbol",
    "cross_section_rank",
    "final_score",
    "score_state",
    "decision_label",
    "signal_type",
    "has_confirmed_exit",
    "primary_exit_reason",
    "exit_reasons",
    "executable_entry_date",
    "executable_entry_signal_open",
    "sector_score_available",
    "sector_fallback_neutralized",
]


@dataclass(frozen=True)
class SignalPipelineOutputs:
    panel_csv: Path
    panel_parquet: Path
    current_scores: Path
    current_signals: Path
    buy_events: Path
    sell_events: Path
    event_study: Path
    quality_summary: Path
    report: Path
    loaded_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    latest_score_date: pd.Timestamp | None
    buy_event_count: int
    sell_event_count: int


def load_feature_frames(
    configs: Sequence[SymbolConfig],
    feature_dir: Path = Path("data/features/equities"),
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for config in configs:
        path = feature_dir / safe_symbol_name(config.symbol) / "features.parquet"
        if not path.exists():
            missing.append(config.symbol)
            continue
        frames[config.symbol] = pd.read_parquet(path)
    return frames, missing


def _current_cross_section(panel: pd.DataFrame) -> pd.DataFrame:
    scored = panel.loc[panel["final_score"].notna()]
    if scored.empty:
        return scored.copy()
    latest_date = scored["date"].max()
    return scored.loc[scored["date"].eq(latest_date)].sort_values(
        ["cross_section_rank", "symbol"]
    ).reset_index(drop=True)


def _validate_outputs(panel: pd.DataFrame, current: pd.DataFrame) -> None:
    if panel.duplicated(["symbol", "date"]).any():
        raise ValueError("Signal output contains duplicate symbol/date rows")
    scores = pd.to_numeric(panel["final_score"], errors="coerce").dropna()
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Signal output contains non-finite final scores")
    if not scores.between(0, 100, inclusive="both").all():
        raise ValueError("Signal output contains scores outside 0-100")
    fallback = panel["sector_fallback_neutralized"].fillna(False).astype(bool) & panel[
        "final_score"
    ].notna()
    if fallback.any() and not panel.loc[fallback, "sector_relative_strength_score"].eq(50.0).all():
        raise ValueError("Sector fallback rows were not neutralized at 50")
    if not current.empty and current["cross_section_rank"].isna().any():
        raise ValueError("Current ranking contains missing ranks")


def run_signal_pipeline(
    configs: Sequence[SymbolConfig],
    feature_dir: Path = Path("data/features/equities"),
    output_dir: Path = Path("data/signals"),
    report_path: Path = Path("reports/score_signal_research.md"),
    minimum_cross_section: int = MINIMUM_CROSS_SECTION,
) -> SignalPipelineOutputs:
    frames, missing = load_feature_frames(configs, feature_dir)
    scored = build_score_panel(frames, configs, minimum_cross_section=minimum_cross_section)
    if scored.empty:
        raise ValueError("No feature files were available for signal scoring")
    panel = add_signal_rules(scored)
    current = _current_cross_section(panel)
    current_signals = current.reindex(columns=CURRENT_SIGNAL_COLUMNS)
    buy_events, event_study = build_buy_event_study(panel)
    sell_events = build_sell_event_study(panel)
    quality_summary = build_signal_quality_summary(panel, event_study, sell_events)
    _validate_outputs(panel, current)

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_csv = output_dir / "panel_daily_scores.csv"
    panel_parquet = output_dir / "panel_daily_scores.parquet"
    current_scores_path = output_dir / "current_scores.csv"
    current_signals_path = output_dir / "current_signals.csv"
    buy_events_path = output_dir / "buy_signal_events.csv"
    sell_events_path = output_dir / "sell_signal_events.csv"
    event_study_path = output_dir / "signal_event_study.csv"
    quality_summary_path = output_dir / "signal_quality_summary.csv"

    panel.to_csv(panel_csv, index=False)
    panel.to_parquet(panel_parquet, index=False)
    current.to_csv(current_scores_path, index=False)
    current_signals.to_csv(current_signals_path, index=False)
    buy_events.to_csv(buy_events_path, index=False)
    sell_events.to_csv(sell_events_path, index=False)
    event_study.to_csv(event_study_path, index=False)
    quality_summary.to_csv(quality_summary_path, index=False)

    report = build_score_signal_report(
        current,
        current_signals,
        panel,
        buy_events,
        sell_events,
        event_study,
        quality_summary,
        missing,
    )
    write_score_signal_report(report, report_path)
    latest = current["date"].max() if not current.empty else None
    return SignalPipelineOutputs(
        panel_csv=panel_csv,
        panel_parquet=panel_parquet,
        current_scores=current_scores_path,
        current_signals=current_signals_path,
        buy_events=buy_events_path,
        sell_events=sell_events_path,
        event_study=event_study_path,
        quality_summary=quality_summary_path,
        report=report_path,
        loaded_symbols=tuple(sorted(frames)),
        missing_symbols=tuple(missing),
        latest_score_date=latest,
        buy_event_count=len(buy_events),
        sell_event_count=len(sell_events),
    )
