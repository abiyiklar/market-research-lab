from __future__ import annotations

import math

import pandas as pd

from .models import BacktestConfig


XU100_EMA_SPAN = 200
RELATIVE_STRENGTH_WINDOW = 20


def prepare_strategy_data(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.sort_values("date").reset_index(drop=True).copy()
    if "xu100_ema_200" not in prepared.columns:
        prepared["xu100_ema_200"] = prepared["xu100_close"].ewm(
            span=XU100_EMA_SPAN,
            adjust=False,
            min_periods=XU100_EMA_SPAN,
        ).mean()
    if "relative_strength_xu100_mean_20d" not in prepared.columns:
        prepared["relative_strength_xu100_mean_20d"] = prepared[
            "relative_strength_xu100"
        ].rolling(RELATIVE_STRENGTH_WINDOW, min_periods=RELATIVE_STRENGTH_WINDOW).mean()
    return prepared


def _finite_values(row: pd.Series, columns: tuple[str, ...]) -> bool:
    return all(math.isfinite(float(row[column])) for column in columns)


def baseline_entry_signal(row: pd.Series, config: BacktestConfig) -> bool:
    required = (
        "xu100_close",
        "xu100_ema_200",
        "xu100_return_1d",
        "tuprs_signal_close",
        "ema_20",
        "ema_50",
        "ema_100",
        "ema_200",
        "relative_momentum_20d",
        "relative_momentum_60d",
        "relative_strength_xu100",
        "relative_strength_xu100_mean_20d",
        "rsi_14",
        "volume_ratio_20d",
        "atr_14",
        "tuprs_volume",
    )
    if bool(row["is_indicator_warmup"]) or not _finite_values(row, required):
        return False

    positive_market_regime = (
        row["xu100_close"] > row["xu100_ema_200"]
        and row["xu100_return_1d"] >= config.minimum_market_return
    )
    tuprs_trend = (
        row["tuprs_signal_close"] > row["ema_200"]
        and row["ema_20"] > row["ema_50"] > row["ema_100"]
    )
    relative_strength = (
        row["relative_momentum_20d"] > 0
        and row["relative_momentum_60d"] > 0
        and row["relative_strength_xu100"] > row["relative_strength_xu100_mean_20d"]
    )
    momentum = config.minimum_rsi <= row["rsi_14"] <= config.maximum_entry_rsi
    volume = row["volume_ratio_20d"] >= config.minimum_volume_ratio and row["tuprs_volume"] > 0
    return bool(positive_market_regime and tuprs_trend and relative_strength and momentum and volume)


def close_exit_reason(
    row: pd.Series,
    holding_days: int,
    config: BacktestConfig,
) -> str | None:
    if (
        math.isfinite(float(row["ema_50"]))
        and row["tuprs_signal_close"] < row["ema_50"]
    ):
        return "trend_exit"
    if math.isfinite(float(row["relative_momentum_20d"])) and row["relative_momentum_20d"] < 0:
        return "relative_strength_exit"
    if math.isfinite(float(row["rsi_14"])) and row["rsi_14"] > config.maximum_exit_rsi:
        return "rsi_exit"
    if holding_days >= config.maximum_holding_days:
        return "maximum_holding_days"
    return None
