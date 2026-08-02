from __future__ import annotations

import pandas as pd
import pytest

from bist_research.backtest.engine import BacktestEngine
from bist_research.backtest.models import BacktestConfig


def _engine_frame(row_count: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=row_count, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "tuprs_open": [100.0] * row_count,
            "tuprs_high": [102.0] * row_count,
            "tuprs_low": [98.0] * row_count,
            "tuprs_close": [101.0] * row_count,
            "tuprs_volume": [1_000.0] * row_count,
            "xu100_close": [1_000.0 + index for index in range(row_count)],
            "xu100_return_1d": [0.001] * row_count,
            "relative_strength_xu100": [1.0 + index / 100 for index in range(row_count)],
            "relative_momentum_20d": [0.10] * row_count,
            "relative_momentum_60d": [0.10] * row_count,
            "volume_ratio_20d": [1.20] * row_count,
            "ema_20": [99.0] * row_count,
            "ema_50": [98.0] * row_count,
            "ema_100": [97.0] * row_count,
            "ema_200": [96.0] * row_count,
            "rsi_14": [60.0] * row_count,
            "atr_14": [2.0] * row_count,
            "is_indicator_warmup": [False] * row_count,
            "force_signal": [False] * row_count,
        }
    )


def _forced_signal(row: pd.Series, config: BacktestConfig) -> bool:
    del config
    return bool(row["force_signal"])


def _zero_cost_config(**overrides: object) -> BacktestConfig:
    values: dict[str, object] = {"commission_rate": 0.0, "slippage_rate": 0.0}
    values.update(overrides)
    return BacktestConfig(**values)


def test_signal_enters_next_day_open_not_same_day_close() -> None:
    frame = _engine_frame(3)
    frame.loc[0, "force_signal"] = True
    frame.loc[0, "tuprs_close"] = 90.0
    frame.loc[1, "tuprs_open"] = 105.0
    frame.loc[1, "tuprs_high"] = 107.0
    frame.loc[1, "tuprs_low"] = 103.0
    frame.loc[1, "tuprs_close"] = 106.0

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["signal_date"] == frame.loc[0, "date"]
    assert trade["entry_date"] == frame.loc[1, "date"]
    assert trade["entry_price_raw"] == 105.0
    assert trade["entry_price_raw"] != frame.loc[0, "tuprs_close"]


def test_gap_below_stop_exits_at_open() -> None:
    frame = _engine_frame(4)
    frame.loc[0, ["force_signal", "atr_14"]] = [True, 4.0]
    frame.loc[1, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close", "atr_14"]] = [
        100,
        104,
        96,
        102,
        10,
    ]
    frame.loc[2, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [85, 88, 84, 87]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["initial_stop"] == 90.0
    assert trade["exit_price_raw"] == 85.0
    assert trade["exit_reason"] == "atr_stop"


def test_intraday_stop_exits_at_stop_price() -> None:
    frame = _engine_frame(4)
    frame.loc[0, ["force_signal", "atr_14"]] = [True, 4.0]
    frame.loc[1, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close", "atr_14"]] = [
        100,
        104,
        96,
        102,
        10,
    ]
    frame.loc[2, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [98, 101, 89, 95]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["exit_price_raw"] == 90.0
    assert trade["exit_reason"] == "atr_stop"


def test_stop_and_close_exit_use_more_conservative_price() -> None:
    frame = _engine_frame(4)
    frame.loc[0, ["force_signal", "atr_14"]] = [True, 4.0]
    frame.loc[1, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close", "atr_14"]] = [
        100,
        104,
        96,
        102,
        10,
    ]
    frame.loc[2, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close", "ema_50"]] = [
        98,
        101,
        84,
        85,
        90,
    ]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["exit_price_raw"] == 85.0
    assert trade["exit_reason"] == "atr_stop+trend_exit"


def test_commission_slippage_and_whole_share_quantity() -> None:
    frame = _engine_frame(3)
    frame.loc[0, ["force_signal", "atr_14"]] = [True, 10.0]
    frame.loc[1, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [100, 111, 99, 110]
    frame.loc[2, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [110, 112, 108, 110]
    config = BacktestConfig(
        initial_capital=1_000,
        commission_rate=0.01,
        slippage_rate=0.01,
    )

    result = BacktestEngine(config, _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["quantity"] == 9
    assert trade["entry_price_effective"] == 101.0
    assert trade["entry_commission"] == pytest.approx(9.09)
    assert trade["exit_price_effective"] == pytest.approx(108.9)
    assert trade["exit_commission"] == pytest.approx(9.801)
    assert trade["slippage_cost"] == pytest.approx(18.9)


def test_zero_volume_execution_day_does_not_open_trade() -> None:
    frame = _engine_frame(3)
    frame.loc[0, "force_signal"] = True
    frame.loc[1, "tuprs_volume"] = 0

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)

    assert result.trades.empty
    assert not result.daily_equity["position_open"].any()


def test_warmup_signal_is_not_used() -> None:
    frame = _engine_frame(3)
    frame.loc[0, ["force_signal", "is_indicator_warmup"]] = [True, True]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)

    assert result.trades.empty
    assert frame.loc[0, "date"] not in result.daily_equity["date"].tolist()


def test_maximum_holding_period_closes_position() -> None:
    frame = _engine_frame(5)
    frame.loc[0, ["force_signal", "atr_14"]] = [True, 10.0]
    config = _zero_cost_config(maximum_holding_days=2)

    result = BacktestEngine(config, _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["holding_days"] == 2
    assert trade["exit_date"] == frame.loc[2, "date"]
    assert trade["exit_reason"] == "maximum_holding_days"


def test_trailing_stop_uses_prior_known_close_and_atr() -> None:
    frame = _engine_frame(4)
    frame.loc[0, ["force_signal", "atr_14"]] = [True, 2.0]
    frame.loc[1, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close", "atr_14"]] = [
        100,
        111,
        99,
        110,
        2,
    ]
    frame.loc[2, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [110, 112, 103, 105]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["initial_stop"] == 95.0
    assert trade["exit_price_raw"] == 104.0
    assert trade["exit_reason"] == "trailing_stop"


def test_engine_never_holds_more_than_one_position() -> None:
    frame = _engine_frame(8)
    frame["force_signal"] = True
    frame["atr_14"] = 20.0

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)

    assert len(result.trades) == 1
    assert result.daily_equity["position_quantity"].max() <= result.trades.iloc[0]["quantity"]


def test_future_price_change_does_not_change_prior_equity() -> None:
    frame = _engine_frame(5)
    frame.loc[0, ["force_signal", "atr_14"]] = [True, 20.0]
    changed = frame.copy()
    changed.loc[4, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [200, 210, 190, 205]

    original_result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    changed_result = BacktestEngine(_zero_cost_config(), _forced_signal).run(changed)

    pd.testing.assert_frame_equal(
        original_result.daily_equity.iloc[:-1].reset_index(drop=True),
        changed_result.daily_equity.iloc[:-1].reset_index(drop=True),
    )
