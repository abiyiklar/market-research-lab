from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No observations._"
    selected = frame.reindex(columns=columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_format_value(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _summary_slice(
    summary: pd.DataFrame,
    analysis: str,
    group: str,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary.loc[
        summary["analysis"].eq(analysis)
        & summary["group"].eq(group)
        & summary["horizon_sessions"].isin(horizons)
    ].copy()


def build_score_signal_report(
    current_scores: pd.DataFrame,
    current_signals: pd.DataFrame,
    panel: pd.DataFrame,
    buy_events: pd.DataFrame,
    sell_events: pd.DataFrame,
    event_study: pd.DataFrame,
    quality_summary: pd.DataFrame,
    missing_symbols: list[str],
) -> str:
    latest_date = current_scores["date"].max() if not current_scores.empty else pd.NaT
    ranking_columns = [
        "cross_section_rank",
        "symbol",
        "final_score",
        "score_state",
        "decision_label",
        "signal_type",
        "primary_exit_reason",
    ]
    component_columns = [
        "symbol",
        "market_relative_strength_score",
        "sector_relative_strength_score",
        "momentum_score",
        "trend_quality_score",
        "volume_confirmation_score",
        "risk_quality_score",
        "sector_score_available",
    ]
    signal_counts = (
        buy_events.groupby(["symbol", "signal_type"], observed=True)
        .size()
        .rename("buy_event_count")
        .reset_index()
        if not buy_events.empty
        else pd.DataFrame()
    )
    exit_counts = (
        sell_events.groupby(["symbol", "primary_exit_reason"], observed=True)
        .size()
        .rename("sell_event_count")
        .reset_index()
        if not sell_events.empty
        else pd.DataFrame()
    )
    forward = _summary_slice(quality_summary, "buy_event", "overall", (5, 10, 20, 40, 60))
    buckets = _summary_slice(quality_summary, "score_bucket", "score_state", (20, 60))
    fallbacks = current_scores.loc[
        current_scores.get("sector_fallback_neutralized", False).fillna(False).astype(bool),
        ["symbol", "market_benchmark", "sector_benchmark", "sector_relative_strength_score"],
    ] if not current_scores.empty else pd.DataFrame()

    efficiency = pd.DataFrame()
    if not event_study.empty:
        efficiency = pd.DataFrame(
            [
                {
                    "buy_events": len(event_study),
                    "overlapping_events": int(event_study["overlapping_signal"].sum()),
                    "mean_mfe_60": event_study["mfe_60"].mean(),
                    "mean_mae_60": event_study["mae_60"].mean(),
                    "mean_entry_efficiency_20": event_study["entry_efficiency_20"].mean(),
                    "median_distance_from_low_20": event_study[
                        "distance_from_subsequent_low_20"
                    ].median(),
                    "median_captured_upside_20": event_study[
                        "captured_potential_upside_20"
                    ].median(),
                }
            ]
        )
    exit_efficiency = pd.DataFrame()
    if not sell_events.empty:
        exit_efficiency = pd.DataFrame(
            [
                {
                    "sell_events": len(sell_events),
                    "mean_post_exit_decline_20": sell_events[
                        "maximum_post_exit_decline_20"
                    ].mean(),
                    "mean_missed_upside_20": sell_events["missed_upside_after_exit_20"].mean(),
                }
            ]
        )

    decision_counts = (
        current_signals["decision_label"].value_counts().rename_axis("state").reset_index(name="count")
        if not current_signals.empty
        else pd.DataFrame()
    )
    high_bucket = buckets.loc[buckets["value"].isin(["STRONG", "POSITIVE"])] if not buckets.empty else buckets
    low_bucket = buckets.loc[buckets["value"].isin(["WEAKENING", "WEAK"])] if not buckets.empty else buckets
    high_median = high_bucket.loc[high_bucket["horizon_sessions"].eq(20), "median_forward_return"].mean()
    low_median = low_bucket.loc[low_bucket["horizon_sessions"].eq(20), "median_forward_return"].mean()
    bucket_conclusion = (
        "High-score buckets produced a higher median 20-session return than low-score buckets."
        if np.isfinite([high_median, low_median]).all() and high_median > low_median
        else "The available sample does not show a higher median 20-session return for high-score buckets."
    )

    return f"""# Score and Signal Research

Generated for the latest scored cross-section: **{_format_value(latest_date)}**.

This is a transparent historical research diagnostic. The labels are not investment advice, and this output is not deployment-ready.

## Latest Ranking

{_markdown_table(current_scores, ranking_columns)}

## Current States

{_markdown_table(decision_counts, ["state", "count"])}

## Component Breakdown

{_markdown_table(current_scores, component_columns)}

## Sector Fallback Warnings

Fallback rows receive a neutral sector score of 50 and are excluded from sector percentile ranking, preventing XU100 information from being rewarded twice.

{_markdown_table(fallbacks, ["symbol", "market_benchmark", "sector_benchmark", "sector_relative_strength_score"])}

## Signal Counts

### BUY events

{_markdown_table(signal_counts, ["symbol", "signal_type", "buy_event_count"])}

### SELL events

{_markdown_table(exit_counts, ["symbol", "primary_exit_reason", "sell_event_count"])}

## Forward Return and Benchmark Excess

{_markdown_table(forward, ["horizon_sessions", "observation_count", "positive_return_pct", "outperform_xu100_pct", "median_forward_return", "mean_forward_return", "median_xu100_excess_return", "mean_xu100_excess_return", "median_sector_excess_return"])}

## MFE, MAE and Entry Efficiency

{_markdown_table(efficiency, ["buy_events", "overlapping_events", "mean_mfe_60", "mean_mae_60", "mean_entry_efficiency_20", "median_distance_from_low_20", "median_captured_upside_20"])}

## Exit Efficiency

{_markdown_table(exit_efficiency, ["sell_events", "mean_post_exit_decline_20", "mean_missed_upside_20"])}

## Score Bucket Outcomes

{_markdown_table(buckets, ["value", "horizon_sessions", "observation_count", "positive_return_pct", "outperform_xu100_pct", "median_forward_return", "mean_forward_return", "median_xu100_excess_return"])}

**Research observation:** {bucket_conclusion}

## Limitations

- The universe is the fixed current 12-stock research panel and does not reconstruct historical index membership.
- Cross-sectional scores are absent when fewer than eight stocks have valid same-date post-warm-up inputs.
- Missing stock sessions are not forward-filled; index values were forward-filled causally by the upstream feature pipeline.
- Sector fallback uses a neutral 50 and is not a second market-relative-strength reward.
- Forward returns, excess returns, MFE, MAE, and efficiency fields are retrospective diagnostics only and are never signal inputs.
- Overlapping BUY events are marked and must not be interpreted as independent portfolio trades.
- Thresholds are fixed research assumptions and were not optimized.
- No transaction costs, capacity limits, portfolio construction, live execution, broker integration, or notification delivery is modeled.
- Missing feature inputs at runtime: {", ".join(missing_symbols) if missing_symbols else "none"}.
- No deployment-ready or investment-advice claim is made.
"""


def write_score_signal_report(report: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
