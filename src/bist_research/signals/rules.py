from __future__ import annotations

import numpy as np
import pandas as pd


EXIT_PRIORITY = (
    "BREAKDOWN_EXIT",
    "TREND_EXIT",
    "SCORE_EXIT",
    "RELATIVE_STRENGTH_EXIT",
)


def _join_exit_reasons(row: pd.Series) -> str:
    return "|".join(reason for reason in EXIT_PRIORITY if bool(row[f"is_{reason.lower()}"]))


def _primary_exit_reason(row: pd.Series) -> str | None:
    for reason in EXIT_PRIORITY:
        if bool(row[f"is_{reason.lower()}"]):
            return reason
    return None


def _decision_labels(frame: pd.DataFrame) -> pd.Series:
    decisions: list[str] = []
    active = False
    for row in frame.itertuples(index=False):
        if bool(row.has_confirmed_exit):
            active = False
            decisions.append("SELL")
        elif bool(row.is_buy_event):
            active = True
            decisions.append("BUY")
        elif bool(row.early_exit_warning) or row.score_state == "WEAKENING":
            decisions.append("WEAKENING")
        elif active and np.isfinite(row.final_score) and row.final_score >= 50:
            decisions.append("HOLD")
        elif np.isfinite(row.final_score) and row.final_score >= 70:
            decisions.append("WATCH")
        else:
            decisions.append("NO_SIGNAL")
    return pd.Series(decisions, index=frame.index, dtype="string")


def _add_symbol_rules(symbol_frame: pd.DataFrame) -> pd.DataFrame:
    frame = symbol_frame.sort_values("date").reset_index(drop=True).copy()
    close = pd.to_numeric(frame["signal_close"], errors="coerce")
    atr = pd.to_numeric(frame["atr_14"], errors="coerce")
    score = pd.to_numeric(frame["final_score"], errors="coerce")
    repaired = frame.get("quality_repaired", frame.get("repaired_data", False))
    if not isinstance(repaired, pd.Series):
        repaired = pd.Series(bool(repaired), index=frame.index)
    repaired = repaired.fillna(False).astype(bool)

    frame["prior_55_session_high"] = (
        pd.to_numeric(frame["signal_high"], errors="coerce")
        .rolling(55, min_periods=55)
        .max()
        .shift(1)
    )
    frame["prior_20_session_low"] = (
        pd.to_numeric(frame["signal_low"], errors="coerce")
        .rolling(20, min_periods=20)
        .min()
        .shift(1)
    )

    distance_ema20 = (close - frame["ema_20"]).abs()
    distance_ema50 = (close - frame["ema_50"]).abs()
    frame["pullback_buy_condition"] = (
        score.ge(70)
        & close.gt(frame["ema_100"])
        & frame["ema_50"].gt(frame["ema_100"])
        & frame["ema_100"].gt(frame["ema_200"])
        & frame["rsi_14"].between(40, 60, inclusive="both")
        & (distance_ema20.le(0.75 * atr) | distance_ema50.le(0.75 * atr))
        & close.ge(frame["ema_50"] - 0.25 * atr)
        & close.le(frame["ema_20"] + 2.5 * atr)
        & frame["relative_momentum_market_60"].gt(0)
        & ~repaired
    ).fillna(False)
    frame["breakout_buy_condition"] = (
        score.ge(75)
        & close.gt(frame["prior_55_session_high"])
        & frame["volume_ratio_20"].ge(1.20)
        & close.gt(frame["ema_50"])
        & frame["ema_50"].gt(frame["ema_100"])
        & frame["ema_100"].gt(frame["ema_200"])
        & frame["relative_momentum_market_60"].gt(0)
        & close.le(frame["ema_20"] + 2.5 * atr)
        & ~repaired
    ).fillna(False)

    frame["new_pullback_buy"] = (
        frame["pullback_buy_condition"]
        & ~frame["pullback_buy_condition"].shift(1, fill_value=False)
    )
    frame["new_breakout_buy"] = (
        frame["breakout_buy_condition"]
        & ~frame["breakout_buy_condition"].shift(1, fill_value=False)
    )
    frame["is_buy_event"] = frame["new_pullback_buy"] | frame["new_breakout_buy"]
    frame["signal_type"] = np.select(
        [
            frame["is_buy_event"]
            & frame["pullback_buy_condition"]
            & frame["breakout_buy_condition"],
            frame["new_pullback_buy"],
            frame["new_breakout_buy"],
        ],
        ["COMBINED_BUY", "PULLBACK_BUY", "BREAKOUT_BUY"],
        default="",
    )

    valid_score = score.notna()
    below_50 = valid_score & score.lt(50)
    previous_below_50 = below_50.shift(1, fill_value=False) & valid_score.shift(
        1, fill_value=False
    )
    frame["is_score_exit"] = below_50 & previous_below_50
    frame["is_trend_exit"] = (
        close.lt(frame["ema_50"]) & frame["ema_20_slope_10"].lt(0)
    ).fillna(False)
    frame["is_relative_strength_exit"] = (
        frame["relative_momentum_market_60"].lt(0) & score.lt(55)
    ).fillna(False)
    frame["is_breakdown_exit"] = close.lt(frame["prior_20_session_low"]).fillna(False)
    exit_columns = [f"is_{reason.lower()}" for reason in EXIT_PRIORITY]
    frame["has_confirmed_exit"] = frame[exit_columns].any(axis=1)
    frame["is_sell_event"] = frame["has_confirmed_exit"] & ~frame[
        "has_confirmed_exit"
    ].shift(1, fill_value=False)
    frame["exit_reasons"] = frame.apply(_join_exit_reasons, axis=1)
    frame["primary_exit_reason"] = frame.apply(_primary_exit_reason, axis=1)
    frame["early_exit_warning"] = below_50 & ~frame["is_score_exit"]

    next_date = frame["date"].shift(-1)
    next_open = pd.to_numeric(frame["signal_open"], errors="coerce").shift(-1)
    frame["executable_entry_date"] = next_date.where(frame["is_buy_event"])
    frame["executable_entry_signal_open"] = next_open.where(frame["is_buy_event"])
    frame["decision_label"] = _decision_labels(frame)
    return frame


def add_signal_rules(scored_panel: pd.DataFrame) -> pd.DataFrame:
    if scored_panel.empty:
        return scored_panel.copy()
    if scored_panel.duplicated(["symbol", "date"]).any():
        raise ValueError("Scored panel contains duplicate symbol/date rows")
    frames = [
        _add_symbol_rules(group)
        for _, group in scored_panel.groupby("symbol", sort=True, observed=True)
    ]
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)
