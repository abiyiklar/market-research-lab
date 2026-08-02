from __future__ import annotations

import pandas as pd
import pytest

from bist_research.backtest.corporate_actions import (
    PRICE_BASIS_SPLIT_ADJUSTED,
    PRICE_BASIS_UNADJUSTED,
    audit_corporate_actions,
    detect_price_basis,
    raw_strategy_prices_are_consistent,
)
from bist_research.backtest.engine import BacktestEngine
from bist_research.backtest.metrics import (
    benchmark_metrics,
    calculate_tuprs_adjusted_total_return,
    calculate_tuprs_explicit_dividend_total_return,
)
from bist_research.backtest.models import BacktestConfig


def _action_frame(row_count: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=row_count, freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "tuprs_open": [100.0] * row_count,
            "tuprs_high": [102.0] * row_count,
            "tuprs_low": [98.0] * row_count,
            "tuprs_close": [100.0] * row_count,
            "tuprs_adj_close": [100.0] * row_count,
            "tuprs_volume": [1_000.0] * row_count,
            "xu100_close": [1_000.0] * row_count,
            "xu100_return_1d": [0.0] * row_count,
            "relative_strength_xu100": [1.0] * row_count,
            "relative_momentum_20d": [0.1] * row_count,
            "relative_momentum_60d": [0.1] * row_count,
            "volume_ratio_20d": [1.2] * row_count,
            "ema_20": [90.0] * row_count,
            "ema_50": [80.0] * row_count,
            "ema_100": [70.0] * row_count,
            "ema_200": [60.0] * row_count,
            "rsi_14": [60.0] * row_count,
            "atr_14": [50.0] * row_count,
            "is_indicator_warmup": [False] * row_count,
            "force_signal": [False] * row_count,
            "dividend_per_share": [0.0] * row_count,
            "stock_split_factor": [0.0] * row_count,
            "corporate_action_flag": [False] * row_count,
            "price_basis": [PRICE_BASIS_SPLIT_ADJUSTED] * row_count,
        }
    )


def _forced_signal(row: pd.Series, config: BacktestConfig) -> bool:
    del config
    return bool(row["force_signal"])


def _never_exit(
    row: pd.Series,
    holding_days: int,
    config: BacktestConfig,
) -> None:
    del row, holding_days, config
    return None


def _zero_cost_config(**overrides: object) -> BacktestConfig:
    values: dict[str, object] = {"commission_rate": 0.0, "slippage_rate": 0.0}
    values.update(overrides)
    return BacktestConfig(**values)


def test_large_close_adjusted_ratio_change_is_corporate_action_candidate() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "tuprs_close": [100.0, 102.0, 120.0],
            "tuprs_adj_close": [100.0, 102.0, 100.0],
        }
    )

    audit = audit_corporate_actions(frame, ratio_change_threshold=0.10)

    assert audit["corporate_action_candidate"].tolist() == [False, False, True]
    assert audit.loc[2, "ratio_change"] > 0.10


def test_raw_strategy_price_consistency_uses_raw_ohlc() -> None:
    frame = pd.DataFrame(
        {
            "tuprs_open": [100.0],
            "tuprs_high": [105.0],
            "tuprs_low": [98.0],
            "tuprs_close": [103.0],
            "tuprs_adj_close": [90.0],
        }
    )

    assert raw_strategy_prices_are_consistent(frame)
    assert not raw_strategy_prices_are_consistent(frame.drop(columns="tuprs_open"))


def test_dividend_is_credited_while_long() -> None:
    frame = _action_frame()
    frame.loc[0, "force_signal"] = True
    frame.loc[2, ["dividend_per_share", "corporate_action_flag"]] = [2.0, True]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    trade = result.trades.iloc[0]
    dividend_row = result.daily_equity.loc[
        result.daily_equity["date"].eq(frame.loc[2, "date"])
    ].iloc[0]

    assert trade["dividend_cash"] == pytest.approx(trade["quantity"] * 2.0)
    assert dividend_row["dividend_cash"] == pytest.approx(trade["dividend_cash"])
    assert trade["net_pnl"] == pytest.approx(trade["dividend_cash"])


def test_no_dividend_is_credited_while_flat() -> None:
    frame = _action_frame()
    frame.loc[1, ["dividend_per_share", "corporate_action_flag"]] = [3.0, True]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)

    assert result.trades.empty
    assert result.daily_equity["dividend_cash"].sum() == 0.0
    assert result.daily_equity["cumulative_dividend_cash"].iloc[-1] == 0.0


def test_ex_dividend_cash_preserves_portfolio_equity() -> None:
    frame = _action_frame()
    frame.loc[0, "force_signal"] = True
    frame.loc[2:, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [
        90.0,
        92.0,
        88.0,
        90.0,
    ]
    frame.loc[2, ["dividend_per_share", "corporate_action_flag"]] = [10.0, True]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    equity = result.daily_equity.set_index("date")

    assert equity.loc[frame.loc[2, "date"], "total_equity"] == pytest.approx(
        equity.loc[frame.loc[1, "date"], "total_equity"]
    )
    assert equity.loc[frame.loc[2, "date"], "daily_return"] == pytest.approx(0.0)
    trade = result.trades.iloc[0]
    assert trade["entry_price_raw"] == 100.0
    assert trade["exit_price_raw"] == 90.0
    assert trade["dividend_cash"] == pytest.approx(trade["quantity"] * 10.0)
    assert result.daily_equity["dividend_cash"].sum() == pytest.approx(
        trade["dividend_cash"]
    )
    assert trade["net_pnl"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("config_overrides", "expected_stop_before_dividend"),
    [
        ({"atr_stop_multiplier": 5.0, "atr_trailing_multiplier": 100.0}, 90.0),
        ({"atr_stop_multiplier": 20.0, "atr_trailing_multiplier": 2.0}, 96.0),
    ],
    ids=["atr-stop", "trailing-stop"],
)
def test_pure_dividend_drop_does_not_trigger_raw_basis_stops(
    config_overrides: dict[str, float],
    expected_stop_before_dividend: float,
) -> None:
    frame = _action_frame()
    frame["atr_14"] = 2.0
    frame.loc[0, "force_signal"] = True
    frame.loc[2:, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [
        90.0,
        92.0,
        88.0,
        90.0,
    ]
    frame.loc[2, ["dividend_per_share", "corporate_action_flag"]] = [10.0, True]
    config = _zero_cost_config(**config_overrides)

    result = BacktestEngine(config, _forced_signal, _never_exit).run(frame)

    assert result.daily_equity.loc[1, "current_stop"] == pytest.approx(
        expected_stop_before_dividend
    )
    assert result.trades["exit_reason"].tolist() == ["end_of_period"]
    assert result.trades.loc[0, "dividend_cash"] == pytest.approx(
        result.trades.loc[0, "quantity"] * 10.0
    )


def test_pure_dividend_drop_does_not_trigger_false_trend_exit() -> None:
    frame = _action_frame()
    frame["ema_50"] = 95.0
    frame.loc[0, "force_signal"] = True
    frame.loc[2:, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [
        90.0,
        92.0,
        88.0,
        90.0,
    ]
    frame.loc[2, ["dividend_per_share", "corporate_action_flag"]] = [10.0, True]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)

    assert result.trades["exit_reason"].tolist() == ["end_of_period"]
    assert result.trades.loc[0, "exit_price_raw"] == 90.0


def test_split_adjusted_prices_do_not_double_quantity() -> None:
    frame = _action_frame()
    frame.loc[0, "force_signal"] = True
    frame.loc[2, ["stock_split_factor", "corporate_action_flag"]] = [7.0, True]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["quantity"] == 1_000
    assert trade["stock_split_factor"] == 1.0
    assert result.daily_equity.loc[2, "stock_split_factor"] == 7.0
    assert bool(trade["corporate_action_flag"])


def test_unadjusted_split_changes_quantity_once_and_keeps_equity_continuous() -> None:
    frame = _action_frame()
    frame.loc[0, "force_signal"] = True
    frame.loc[1, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [
        700.0,
        710.0,
        690.0,
        700.0,
    ]
    frame.loc[1:, "price_basis"] = PRICE_BASIS_UNADJUSTED
    frame.loc[2, ["stock_split_factor", "corporate_action_flag"]] = [7.0, True]

    result = BacktestEngine(_zero_cost_config(), _forced_signal).run(frame)
    trade = result.trades.iloc[0]

    assert trade["quantity"] == 994
    assert trade["stock_split_factor"] == 7.0
    assert trade["return_pct"] == pytest.approx(0.0)


def test_adjusted_benchmark_reconciles_with_explicit_dividend_cash() -> None:
    frame = _action_frame(2)
    frame.loc[1, ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] = [
        90.0,
        92.0,
        88.0,
        90.0,
    ]
    frame["tuprs_adj_close"] = [90.0, 90.0]
    frame.loc[1, ["dividend_per_share", "corporate_action_flag"]] = [10.0, True]
    config = _zero_cost_config()

    explicit = benchmark_metrics(
        calculate_tuprs_explicit_dividend_total_return(frame, config),
        config,
    )
    adjusted = benchmark_metrics(
        calculate_tuprs_adjusted_total_return(frame, config),
        config,
    )

    assert explicit["total_return"] == pytest.approx(adjusted["total_return"], abs=1e-12)


def test_price_basis_detection_distinguishes_split_adjustment() -> None:
    adjusted = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-04-03", "2023-04-04"]),
            "Open": [99.0, 101.0],
            "Close": [100.0, 102.0],
            "Stock Splits": [0.0, 7.0],
        }
    )
    unadjusted = adjusted.copy()
    unadjusted.loc[0, "Close"] = 700.0

    assert detect_price_basis(adjusted) == PRICE_BASIS_SPLIT_ADJUSTED
    assert detect_price_basis(unadjusted) == PRICE_BASIS_UNADJUSTED
