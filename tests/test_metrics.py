from __future__ import annotations

import pandas as pd
import pytest

from bist_research.backtest.metrics import (
    calculate_drawdowns,
    calculate_tuprs_buy_hold,
    calculate_xu100_buy_hold,
)
from bist_research.backtest.models import BacktestConfig, PERIOD_DEFINITIONS
from bist_research.backtest.report import slice_period


def _equity(values: list[float]) -> pd.DataFrame:
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=len(values), freq="B"),
            "total_equity": values,
        }
    )
    equity["daily_return"] = equity["total_equity"].pct_change(fill_method=None).fillna(0.0)
    equity["running_peak"] = equity["total_equity"].cummax()
    equity["drawdown"] = equity["total_equity"] / equity["running_peak"] - 1
    return equity


def test_drawdown_value_and_dates() -> None:
    equity = _equity([100.0, 120.0, 90.0, 110.0])

    drawdowns = calculate_drawdowns(equity)
    trough = drawdowns.loc[drawdowns["drawdown"].idxmin()]

    assert trough["drawdown"] == pytest.approx(-0.25)
    assert trough["drawdown_start"] == equity.loc[1, "date"]
    assert trough["date"] == equity.loc[2, "date"]


def test_tuprs_and_xu100_benchmarks() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "tuprs_open": [100.0, 105.0, 110.0],
            "tuprs_close": [100.0, 105.0, 110.0],
            "tuprs_volume": [1_000.0, 1_000.0, 1_000.0],
            "xu100_close": [200.0, 210.0, 220.0],
            "is_indicator_warmup": [False, False, False],
        }
    )
    config = BacktestConfig(initial_capital=1_000, commission_rate=0, slippage_rate=0)

    tuprs = calculate_tuprs_buy_hold(frame, config)
    xu100 = calculate_xu100_buy_hold(frame, config)

    assert tuprs.daily_equity["total_equity"].iloc[-1] == pytest.approx(1_100.0)
    assert xu100.daily_equity["total_equity"].iloc[-1] == pytest.approx(1_100.0)
    assert tuprs.total_commission == 0
    assert xu100.total_slippage == 0


def test_train_validation_and_test_periods_do_not_overlap() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2019-12-31", "2020-01-01", "2022-12-31", "2023-01-02"]
            )
        }
    )

    slices = {period.name: slice_period(frame, period) for period in PERIOD_DEFINITIONS}

    assert slices["train"]["date"].dt.year.tolist() == [2019]
    assert slices["validation"]["date"].dt.year.tolist() == [2020, 2022]
    assert slices["test"]["date"].dt.year.tolist() == [2023]
    combined = pd.concat(slices.values(), ignore_index=True)
    assert not combined["date"].duplicated().any()
