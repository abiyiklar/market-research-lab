from __future__ import annotations

import pandas as pd

from bist_research.backtest.models import BacktestConfig
from bist_research.backtest.strategy import baseline_entry_signal, prepare_strategy_data


def _strategy_frame(row_count: int = 220) -> pd.DataFrame:
    index = pd.Series(range(row_count), dtype="float64")
    close = 200.0 + index
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-02", periods=row_count, freq="B"),
            "xu100_close": 1_000.0 + index,
            "xu100_return_1d": 0.001,
            "tuprs_close": close,
            "tuprs_volume": 1_000.0,
            "ema_20": close - 1,
            "ema_50": close - 2,
            "ema_100": close - 3,
            "ema_200": close - 4,
            "relative_momentum_20d": 0.10,
            "relative_momentum_60d": 0.08,
            "relative_strength_xu100": 1.0 + index / 100,
            "rsi_14": 60.0,
            "volume_ratio_20d": 1.20,
            "atr_14": 2.0,
            "is_indicator_warmup": [True] * 200 + [False] * (row_count - 200),
        }
    )


def test_baseline_signal_requires_all_conditions() -> None:
    prepared = prepare_strategy_data(_strategy_frame())
    row = prepared.iloc[-1]

    assert baseline_entry_signal(row, BacktestConfig())

    failed = row.copy()
    failed["volume_ratio_20d"] = 1.09
    assert not baseline_entry_signal(failed, BacktestConfig())


def test_baseline_signal_rejects_extreme_market_loss_and_warmup() -> None:
    prepared = prepare_strategy_data(_strategy_frame())
    extreme_loss = prepared.iloc[-1].copy()
    extreme_loss["xu100_return_1d"] = -0.031
    warmup = prepared.iloc[-1].copy()
    warmup["is_indicator_warmup"] = True

    assert not baseline_entry_signal(extreme_loss, BacktestConfig())
    assert not baseline_entry_signal(warmup, BacktestConfig())


def test_strategy_columns_do_not_change_when_future_data_changes() -> None:
    original = _strategy_frame()
    changed = original.copy()
    changed.loc[len(changed) - 1, "xu100_close"] *= 10
    changed.loc[len(changed) - 1, "relative_strength_xu100"] *= 10

    original_prepared = prepare_strategy_data(original)
    changed_prepared = prepare_strategy_data(changed)

    pd.testing.assert_series_equal(
        original_prepared["xu100_ema_200"].iloc[:-1],
        changed_prepared["xu100_ema_200"].iloc[:-1],
    )
    pd.testing.assert_series_equal(
        original_prepared["relative_strength_xu100_mean_20d"].iloc[:-1],
        changed_prepared["relative_strength_xu100_mean_20d"].iloc[:-1],
    )
