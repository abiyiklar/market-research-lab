from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


BUY_HORIZONS = (5, 10, 20, 40, 60)
SELL_HORIZONS = (5, 10, 20)
SUMMARY_COLUMNS = (
    "analysis",
    "group",
    "value",
    "horizon_sessions",
    "observation_count",
    "positive_return_pct",
    "outperform_xu100_pct",
    "mean_forward_return",
    "median_forward_return",
    "mean_xu100_excess_return",
    "median_xu100_excess_return",
    "mean_sector_excess_return",
    "median_sector_excess_return",
    "mean_mfe_60",
    "mean_mae_60",
    "mean_entry_efficiency_20",
    "mean_post_exit_decline_20",
    "mean_missed_upside_after_exit_20",
)
BUY_EVENT_COLUMNS = (
    "symbol",
    "signal_date",
    "signal_type",
    "final_score",
    "score_state",
    "calendar_period",
    "entry_date",
    "entry_signal_open",
    "has_executable_entry",
    "sector_benchmark",
    "sector_score_available",
    "overlap_cluster_id",
    "overlap_cluster_size",
    "overlapping_signal",
)
BUY_STUDY_COLUMNS = (
    *BUY_EVENT_COLUMNS,
    "signal_session_index",
    "entry_market_close",
    "entry_sector_close",
    *(f"forward_return_{horizon}" for horizon in BUY_HORIZONS),
    *(f"market_return_{horizon}" for horizon in BUY_HORIZONS),
    *(f"xu100_excess_return_{horizon}" for horizon in BUY_HORIZONS),
    *(f"sector_return_{horizon}" for horizon in BUY_HORIZONS),
    *(f"sector_excess_return_{horizon}" for horizon in BUY_HORIZONS),
    "mfe_20",
    "mae_20",
    "mfe_60",
    "mae_60",
    "entry_efficiency_20",
    "distance_from_subsequent_low_20",
    "captured_potential_upside_20",
)
SELL_EVENT_COLUMNS = (
    "symbol",
    "signal_date",
    "signal_session_index",
    "primary_exit_reason",
    "exit_reasons",
    "final_score",
    "score_state",
    "calendar_period",
    "exit_signal_close",
    *(f"return_after_exit_{horizon}" for horizon in SELL_HORIZONS),
    "maximum_post_exit_decline_20",
    "missed_upside_after_exit_20",
)


def _calendar_period(date: pd.Timestamp) -> str:
    if date.year <= 2019:
        return "2013-2019"
    if date.year <= 2022:
        return "2020-2022"
    return "2023-latest"


def _safe_return(end_value: float, start_value: float) -> float:
    if not np.isfinite([end_value, start_value]).all() or start_value <= 0:
        return np.nan
    return end_value / start_value - 1.0


def _cluster_overlapping_buys(events: pd.DataFrame, horizon: int = 60) -> pd.DataFrame:
    clustered = events.copy()
    clustered["overlap_cluster_id"] = ""
    clustered["overlap_cluster_size"] = 1
    clustered["overlapping_signal"] = False
    for symbol, indices in clustered.groupby("symbol", sort=True).groups.items():
        ordered = sorted(indices, key=lambda index: int(clustered.loc[index, "signal_session_index"]))
        clusters: list[list[int]] = []
        active: list[int] = []
        active_end = -1
        for index in ordered:
            position = int(clustered.loc[index, "signal_session_index"])
            if active and position > active_end:
                clusters.append(active)
                active = []
            active.append(index)
            active_end = max(active_end, position + horizon)
        if active:
            clusters.append(active)
        for number, cluster in enumerate(clusters, start=1):
            cluster_id = f"{symbol}-C{number:04d}"
            clustered.loc[cluster, "overlap_cluster_id"] = cluster_id
            clustered.loc[cluster, "overlap_cluster_size"] = len(cluster)
            clustered.loc[cluster, "overlapping_signal"] = len(cluster) > 1
    return clustered


def build_buy_event_study(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if panel.empty or "is_buy_event" not in panel:
        return pd.DataFrame(columns=BUY_EVENT_COLUMNS), pd.DataFrame(columns=BUY_STUDY_COLUMNS)
    records: list[dict[str, object]] = []
    for symbol, source in panel.groupby("symbol", sort=True, observed=True):
        frame = source.sort_values("date").reset_index(drop=True)
        event_positions = frame.index[frame["is_buy_event"].fillna(False)]
        for signal_position in event_positions:
            entry_position = signal_position + 1
            row = frame.loc[signal_position]
            record: dict[str, object] = {
                "symbol": symbol,
                "signal_date": row["date"],
                "signal_session_index": int(signal_position),
                "signal_type": row["signal_type"],
                "final_score": row["final_score"],
                "score_state": row["score_state"],
                "calendar_period": _calendar_period(pd.Timestamp(row["date"])),
                "sector_benchmark": row["sector_benchmark"],
                "sector_score_available": bool(row["sector_score_available"]),
                "entry_date": pd.NaT,
                "entry_signal_open": np.nan,
                "has_executable_entry": entry_position < len(frame),
            }
            if entry_position >= len(frame):
                records.append(record)
                continue

            entry = frame.loc[entry_position]
            entry_open = float(entry["signal_open"])
            record["entry_date"] = entry["date"]
            record["entry_signal_open"] = entry_open
            record["entry_market_close"] = entry["market_close"]
            record["entry_sector_close"] = entry["sector_close"]

            for horizon in BUY_HORIZONS:
                exit_position = entry_position + horizon - 1
                if exit_position >= len(frame):
                    record[f"forward_return_{horizon}"] = np.nan
                    record[f"market_return_{horizon}"] = np.nan
                    record[f"xu100_excess_return_{horizon}"] = np.nan
                    record[f"sector_return_{horizon}"] = np.nan
                    record[f"sector_excess_return_{horizon}"] = np.nan
                    continue
                exit_row = frame.loc[exit_position]
                stock_return = _safe_return(float(exit_row["signal_close"]), entry_open)
                market_return = _safe_return(
                    float(exit_row["market_close"]),
                    float(entry["market_close"]),
                )
                sector_return = (
                    _safe_return(
                        float(exit_row["sector_close"]),
                        float(entry["sector_close"]),
                    )
                    if bool(row["sector_score_available"])
                    else np.nan
                )
                record[f"forward_return_{horizon}"] = stock_return
                record[f"market_return_{horizon}"] = market_return
                record[f"xu100_excess_return_{horizon}"] = stock_return - market_return
                record[f"sector_return_{horizon}"] = sector_return
                record[f"sector_excess_return_{horizon}"] = stock_return - sector_return

            for horizon in (20, 60):
                window = frame.iloc[entry_position : min(entry_position + horizon, len(frame))]
                if window.empty or not np.isfinite(entry_open) or entry_open <= 0:
                    record[f"mfe_{horizon}"] = np.nan
                    record[f"mae_{horizon}"] = np.nan
                else:
                    record[f"mfe_{horizon}"] = window["signal_high"].max() / entry_open - 1.0
                    record[f"mae_{horizon}"] = window["signal_low"].min() / entry_open - 1.0

            efficiency_window = frame.iloc[entry_position : min(entry_position + 20, len(frame))]
            if efficiency_window.empty:
                record["entry_efficiency_20"] = np.nan
                record["distance_from_subsequent_low_20"] = np.nan
                record["captured_potential_upside_20"] = np.nan
            else:
                subsequent_low = float(efficiency_window["signal_low"].min())
                subsequent_high = float(efficiency_window["signal_high"].max())
                price_range = subsequent_high - subsequent_low
                record["entry_efficiency_20"] = (
                    (entry_open - subsequent_low) / price_range if price_range > 0 else np.nan
                )
                record["distance_from_subsequent_low_20"] = _safe_return(
                    entry_open,
                    subsequent_low,
                )
                potential_upside = _safe_return(subsequent_high, entry_open)
                realized = record.get("forward_return_20", np.nan)
                record["captured_potential_upside_20"] = (
                    float(realized) / potential_upside
                    if np.isfinite([realized, potential_upside]).all() and potential_upside > 0
                    else np.nan
                )
            records.append(record)

    study = pd.DataFrame(records)
    if study.empty:
        return pd.DataFrame(columns=BUY_EVENT_COLUMNS), pd.DataFrame(columns=BUY_STUDY_COLUMNS)
    study = _cluster_overlapping_buys(study)
    study = study.reindex(columns=BUY_STUDY_COLUMNS)
    return study.loc[:, BUY_EVENT_COLUMNS].copy(), study


def build_sell_event_study(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or "is_sell_event" not in panel:
        return pd.DataFrame(columns=SELL_EVENT_COLUMNS)
    records: list[dict[str, object]] = []
    for symbol, source in panel.groupby("symbol", sort=True, observed=True):
        frame = source.sort_values("date").reset_index(drop=True)
        for position in frame.index[frame["is_sell_event"].fillna(False)]:
            row = frame.loc[position]
            exit_basis = float(row["signal_close"])
            record: dict[str, object] = {
                "symbol": symbol,
                "signal_date": row["date"],
                "signal_session_index": int(position),
                "primary_exit_reason": row["primary_exit_reason"],
                "exit_reasons": row["exit_reasons"],
                "final_score": row["final_score"],
                "score_state": row["score_state"],
                "calendar_period": _calendar_period(pd.Timestamp(row["date"])),
                "exit_signal_close": exit_basis,
            }
            for horizon in SELL_HORIZONS:
                future_position = position + horizon
                record[f"return_after_exit_{horizon}"] = (
                    _safe_return(float(frame.loc[future_position, "signal_close"]), exit_basis)
                    if future_position < len(frame)
                    else np.nan
                )
            future_window = frame.iloc[position + 1 : min(position + 21, len(frame))]
            if future_window.empty:
                record["maximum_post_exit_decline_20"] = np.nan
                record["missed_upside_after_exit_20"] = np.nan
            else:
                record["maximum_post_exit_decline_20"] = (
                    future_window["signal_low"].min() / exit_basis - 1.0
                )
                record["missed_upside_after_exit_20"] = (
                    future_window["signal_high"].max() / exit_basis - 1.0
                )
            records.append(record)
    return pd.DataFrame(records).reindex(columns=SELL_EVENT_COLUMNS)


def _percent(series: pd.Series) -> float:
    usable = series.dropna()
    return float(usable.mean() * 100.0) if not usable.empty else np.nan


def _buy_summary_row(
    frame: pd.DataFrame,
    analysis: str,
    group: str,
    value: str,
    horizon: int,
) -> dict[str, object]:
    returns = pd.to_numeric(frame[f"forward_return_{horizon}"], errors="coerce")
    excess = pd.to_numeric(frame[f"xu100_excess_return_{horizon}"], errors="coerce")
    sector_excess = pd.to_numeric(frame[f"sector_excess_return_{horizon}"], errors="coerce")
    return {
        "analysis": analysis,
        "group": group,
        "value": value,
        "horizon_sessions": horizon,
        "observation_count": int(returns.notna().sum()),
        "positive_return_pct": _percent(returns.gt(0).where(returns.notna())),
        "outperform_xu100_pct": _percent(excess.gt(0).where(excess.notna())),
        "mean_forward_return": returns.mean(),
        "median_forward_return": returns.median(),
        "mean_xu100_excess_return": excess.mean(),
        "median_xu100_excess_return": excess.median(),
        "mean_sector_excess_return": sector_excess.mean(),
        "median_sector_excess_return": sector_excess.median(),
        "mean_mfe_60": pd.to_numeric(frame.get("mfe_60"), errors="coerce").mean(),
        "mean_mae_60": pd.to_numeric(frame.get("mae_60"), errors="coerce").mean(),
        "mean_entry_efficiency_20": pd.to_numeric(
            frame.get("entry_efficiency_20"), errors="coerce"
        ).mean(),
    }


def _iter_groups(frame: pd.DataFrame, column: str) -> Iterable[tuple[str, pd.DataFrame]]:
    for value, group in frame.groupby(column, dropna=False, sort=True, observed=True):
        yield str(value), group


def _score_bucket_forward_data(panel: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, source in panel.groupby("symbol", sort=True, observed=True):
        frame = source.sort_values("date").copy()
        for horizon in (20, 60):
            stock_forward = frame["signal_close"].shift(-horizon) / frame["signal_close"] - 1.0
            market_forward = frame["market_close"].shift(-horizon) / frame["market_close"] - 1.0
            frame[f"forward_return_{horizon}"] = stock_forward
            frame[f"xu100_excess_return_{horizon}"] = stock_forward - market_forward
            sector_forward = frame["sector_close"].shift(-horizon) / frame["sector_close"] - 1.0
            frame[f"sector_excess_return_{horizon}"] = (stock_forward - sector_forward).where(
                frame["sector_score_available"]
            )
        frame["mfe_60"] = np.nan
        frame["mae_60"] = np.nan
        frame["entry_efficiency_20"] = np.nan
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_signal_quality_summary(
    panel: pd.DataFrame,
    buy_study: pd.DataFrame,
    sell_events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not buy_study.empty:
        grouped = [("overall", "all", buy_study)]
        for column in ("signal_type", "score_state", "symbol", "calendar_period"):
            grouped.extend((column, value, group) for value, group in _iter_groups(buy_study, column))
        for group_name, value, group in grouped:
            for horizon in BUY_HORIZONS:
                rows.append(_buy_summary_row(group, "buy_event", group_name, value, horizon))

    score_forward = _score_bucket_forward_data(panel)
    if not score_forward.empty:
        scored = score_forward.loc[score_forward["final_score"].notna()]
        for value, group in _iter_groups(scored, "score_state"):
            for horizon in (20, 60):
                rows.append(_buy_summary_row(group, "score_bucket", "score_state", value, horizon))

    if not sell_events.empty:
        for group_name, value, group in [
            ("overall", "all", sell_events),
            *(
                ("primary_exit_reason", value, grouped)
                for value, grouped in _iter_groups(sell_events, "primary_exit_reason")
            ),
        ]:
            for horizon in SELL_HORIZONS:
                returns = pd.to_numeric(group[f"return_after_exit_{horizon}"], errors="coerce")
                rows.append(
                    {
                        "analysis": "sell_event",
                        "group": group_name,
                        "value": value,
                        "horizon_sessions": horizon,
                        "observation_count": int(returns.notna().sum()),
                        "positive_return_pct": _percent(returns.gt(0).where(returns.notna())),
                        "mean_forward_return": returns.mean(),
                        "median_forward_return": returns.median(),
                        "mean_post_exit_decline_20": pd.to_numeric(
                            group["maximum_post_exit_decline_20"], errors="coerce"
                        ).mean(),
                        "mean_missed_upside_after_exit_20": pd.to_numeric(
                            group["missed_upside_after_exit_20"], errors="coerce"
                        ).mean(),
                    }
                )
    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)
