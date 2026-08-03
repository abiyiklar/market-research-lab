from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bist_research.backtest.corporate_actions import PRICE_BASIS_UNADJUSTED
from bist_research.config import SymbolConfig
from bist_research.portfolio.analytics import (
    build_calendar_year_contribution,
    build_contribution_tables,
)
from bist_research.portfolio.benchmarks import build_xu100_benchmark
from bist_research.portfolio.engine import (
    PortfolioBacktestEngine,
    build_portfolio_context,
    prepare_portfolio_inputs,
)
from bist_research.portfolio.models import PortfolioConfig


def _inputs(
    symbols: Sequence[str] = ("AAA.IS", "BBB.IS"),
    periods: int = 7,
    starting_price: float = 100.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], tuple[SymbolConfig, ...], pd.DatetimeIndex]:
    dates = pd.bdate_range("2024-01-02", periods=periods)
    configs = tuple(SymbolConfig(symbol, "equity", symbol) for symbol in symbols)
    features: dict[str, pd.DataFrame] = {}
    panel_rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        base = starting_price + symbol_index * 10.0
        frame = pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates,
                "execution_open": base,
                "execution_high": base + 1.0,
                "execution_low": base - 1.0,
                "execution_close": base,
                "volume": 1_000.0,
                "dividend": 0.0,
                "stock_split": 0.0,
                "source_symbol": symbol,
                "price_basis": "raw_ohlc_split_basis_unknown",
            }
        )
        features[symbol] = frame
        for date_index, date in enumerate(dates):
            panel_rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "decision_label": "NO_SIGNAL",
                    "is_buy_event": False,
                    "final_score": 50.0,
                    "cross_section_rank": float(symbol_index + 1),
                    "risk_quality_score": 50.0,
                    "score_state": "valid",
                    "signal_type": "",
                    "primary_exit_reason": "",
                    "exit_reasons": "",
                    "insufficient_cross_section": False,
                    "valid_score_input": True,
                    "risk_data_quality_problem": False,
                    "quality_repaired": False,
                    "quality_valid_ohlc": True,
                    "quality_positive_volume": True,
                    "executable_entry_date": (
                        dates[date_index + 1] if date_index + 1 < len(dates) else pd.NaT
                    ),
                    "executable_entry_signal_open": base,
                    "sector_fallback_neutralized": False,
                    "market_close": 10_000.0 + date_index,
                }
            )
    panel = pd.DataFrame(panel_rows).sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel, features, configs, dates


def _signal(
    panel: pd.DataFrame,
    symbol: str,
    date: pd.Timestamp,
    decision: str,
    *,
    score: float = 80.0,
    rank: float = 1.0,
    risk: float = 60.0,
) -> None:
    mask = panel["symbol"].eq(symbol) & panel["date"].eq(date)
    panel.loc[mask, "decision_label"] = decision
    panel.loc[mask, "is_buy_event"] = decision == "BUY"
    panel.loc[mask, "final_score"] = score
    panel.loc[mask, "cross_section_rank"] = rank
    panel.loc[mask, "risk_quality_score"] = risk
    panel.loc[mask, "signal_type"] = "trend_breakout" if decision == "BUY" else ""
    if decision == "SELL":
        panel.loc[mask, "primary_exit_reason"] = "trend_exit"
        panel.loc[mask, "exit_reasons"] = "trend_exit|risk_exit"


def _run(
    panel: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    configs: tuple[SymbolConfig, ...],
    **config_overrides: object,
):
    config = PortfolioConfig(**config_overrides)
    return PortfolioBacktestEngine(config).run(panel, features, configs)


def test_close_signal_executes_only_at_future_valid_raw_open() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    features["AAA.IS"].loc[1, "execution_open"] = 123.45

    result = _run(panel, features, configs)
    order = result.orders.iloc[0]

    assert order["signal_date"] == dates[0]
    assert order["actual_execution_date"] == dates[1]
    assert order["actual_execution_date"] > order["signal_date"]
    assert order["raw_execution_price"] == pytest.approx(123.45)
    assert result.daily.loc[result.daily["date"].eq(dates[0]), "number_open_positions"].item() == 0


def test_future_changes_do_not_alter_earlier_orders_or_equity() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",), periods=8)
    _signal(panel, "AAA.IS", dates[0], "BUY")
    baseline = _run(panel, features, configs)
    changed_features = {"AAA.IS": features["AAA.IS"].copy()}
    changed_features["AAA.IS"].loc[6:, ["execution_open", "execution_high", "execution_low", "execution_close"]] *= 3
    changed = _run(panel, changed_features, configs)

    pd.testing.assert_frame_equal(
        baseline.orders.loc[baseline.orders["signal_date"].le(dates[4])].reset_index(drop=True),
        changed.orders.loc[changed.orders["signal_date"].le(dates[4])].reset_index(drop=True),
    )
    pd.testing.assert_series_equal(
        baseline.daily.loc[baseline.daily["date"].le(dates[5]), "total_portfolio_equity"].reset_index(drop=True),
        changed.daily.loc[changed.daily["date"].le(dates[5]), "total_portfolio_equity"].reset_index(drop=True),
    )


def test_sells_execute_before_buys_on_same_open() -> None:
    panel, features, configs, dates = _inputs()
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "AAA.IS", dates[1], "SELL")
    _signal(panel, "BBB.IS", dates[1], "BUY")

    result = _run(
        panel,
        features,
        configs,
        starting_capital=1_000.0,
        max_positions=1,
        target_weight=1.0,
        transaction_cost_rate=0.0,
    )
    same_day = result.orders.loc[result.orders["actual_execution_date"].eq(dates[2])]

    assert same_day["side"].tolist() == ["SELL", "BUY"]
    assert same_day["order_status"].eq("filled").all()
    assert result.open_positions["symbol"].tolist() == ["BBB.IS"]


def test_repeated_buy_events_do_not_create_duplicate_positions() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "AAA.IS", dates[1], "BUY")

    result = _run(panel, features, configs)

    assert result.orders.query("side == 'BUY' and order_status == 'filled'").shape[0] == 1
    assert result.orders.query("rejection_reason == 'already_held'").shape[0] == 1
    assert not result.open_positions["symbol"].duplicated().any()


def test_repeated_sell_rows_do_not_duplicate_pending_exit() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "AAA.IS", dates[1], "SELL")
    _signal(panel, "AAA.IS", dates[2], "SELL")
    features["AAA.IS"].loc[2, "volume"] = 0.0

    result = _run(panel, features, configs)
    sell_orders = result.orders.loc[result.orders["side"].eq("SELL")]

    assert len(sell_orders) == 1
    assert sell_orders.iloc[0]["actual_execution_date"] == dates[3]


def test_position_cash_quantity_and_leverage_constraints_hold() -> None:
    symbols = tuple(f"S{index}.IS" for index in range(6))
    panel, features, configs, dates = _inputs(symbols)
    for index, symbol in enumerate(symbols):
        _signal(panel, symbol, dates[0], "BUY", score=90 - index)

    result = _run(panel, features, configs)

    assert result.daily["number_open_positions"].max() <= 4
    assert result.daily["cash"].min() >= 0
    assert result.daily["gross_exposure"].max() <= 1.0 + 1e-12
    assert result.open_positions["current_quantity"].ge(0).all()
    assert result.open_positions["current_quantity"].map(lambda value: float(value).is_integer()).all()


def test_transaction_cost_aware_position_sizing_and_cash_deduction() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")

    result = _run(
        panel,
        features,
        configs,
        starting_capital=1_000.0,
        max_positions=1,
        target_weight=1.0,
        transaction_cost_rate=0.10,
    )
    order = result.orders.iloc[0]

    assert order["quantity"] == 9
    assert order["gross_transaction_value"] == pytest.approx(900.0)
    assert order["transaction_cost"] == pytest.approx(90.0)
    assert order["cash_after"] == pytest.approx(10.0)


def test_zero_quantity_order_is_rejected_with_reason() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")

    result = _run(
        panel,
        features,
        configs,
        starting_capital=50.0,
        max_positions=1,
        target_weight=1.0,
    )

    assert result.orders.iloc[0]["order_status"] == "rejected"
    assert result.orders.iloc[0]["rejection_reason"] == "zero_quantity"


def test_entry_and_exit_use_raw_open_prices() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "AAA.IS", dates[2], "SELL")
    features["AAA.IS"].loc[1, "execution_open"] = 123.0
    features["AAA.IS"].loc[3, "execution_open"] = 111.0

    result = _run(panel, features, configs)

    assert result.trades.iloc[0]["entry_price"] == pytest.approx(123.0)
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(111.0)
    assert result.orders.loc[result.orders["side"].eq("BUY"), "raw_execution_price"].item() == pytest.approx(123.0)
    assert result.orders.loc[result.orders["side"].eq("SELL"), "raw_execution_price"].item() == pytest.approx(111.0)


def test_dividends_are_credited_once_and_only_to_eligible_shares() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    features["AAA.IS"].loc[0, "dividend"] = 5.0
    features["AAA.IS"].loc[2, "dividend"] = 2.0

    result = _run(
        panel,
        features,
        configs,
        starting_capital=1_000.0,
        max_positions=1,
        target_weight=1.0,
        transaction_cost_rate=0.0,
    )

    assert result.open_positions.iloc[0]["current_quantity"] == 10
    assert result.daily["cumulative_dividends"].iloc[-1] == pytest.approx(20.0)
    assert result.position_daily_audit["dividend_cash_day"].sum() == pytest.approx(20.0)


def test_unadjusted_split_changes_quantity_without_artificial_pnl() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",), periods=5)
    _signal(panel, "AAA.IS", dates[0], "BUY")
    frame = features["AAA.IS"]
    frame["price_basis"] = PRICE_BASIS_UNADJUSTED
    frame.loc[2, "stock_split"] = 2.0
    frame.loc[2:, ["execution_open", "execution_high", "execution_low", "execution_close"]] = [50.0, 51.0, 49.0, 50.0]

    result = _run(
        panel,
        features,
        configs,
        starting_capital=1_000.0,
        max_positions=1,
        target_weight=1.0,
        transaction_cost_rate=0.0,
    )

    assert result.open_positions.iloc[0]["current_quantity"] == 20
    assert result.daily.loc[result.daily["date"].eq(dates[2]), "total_portfolio_equity"].item() == pytest.approx(1_000.0)
    assert result.daily.loc[result.daily["date"].eq(dates[2]), "unrealized_pnl"].item() == pytest.approx(0.0)


def test_split_adjusted_prices_are_not_adjusted_twice() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",), periods=5)
    _signal(panel, "AAA.IS", dates[0], "BUY")
    features["AAA.IS"].loc[2, "stock_split"] = 2.0

    result = _run(
        panel,
        features,
        configs,
        starting_capital=1_000.0,
        max_positions=1,
        target_weight=1.0,
        transaction_cost_rate=0.0,
    )

    assert result.open_positions.iloc[0]["current_quantity"] == 10
    assert result.daily.loc[result.daily["date"].eq(dates[2]), "total_portfolio_equity"].item() == pytest.approx(1_000.0)


def test_future_split_metadata_cannot_change_earlier_split_accounting() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",), periods=7)
    _signal(panel, "AAA.IS", dates[0], "BUY")
    frame = features["AAA.IS"]
    frame.loc[2, "stock_split"] = 2.0
    frame.loc[2:, ["execution_open", "execution_high", "execution_low", "execution_close"]] = [50.0, 51.0, 49.0, 50.0]
    baseline = _run(panel, features, configs, transaction_cost_rate=0.0)
    changed_features = {"AAA.IS": frame.copy()}
    changed_features["AAA.IS"].loc[5, "stock_split"] = 3.0
    changed_features["AAA.IS"].loc[5, "price_basis"] = "raw_ohlc_split_adjusted"
    changed = _run(panel, changed_features, configs, transaction_cost_rate=0.0)

    pd.testing.assert_frame_equal(
        baseline.daily.loc[baseline.daily["date"].le(dates[4])].reset_index(drop=True),
        changed.daily.loc[changed.daily["date"].le(dates[4])].reset_index(drop=True),
    )


def test_missing_session_delays_execution_and_records_delay() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    features["AAA.IS"].loc[1, "volume"] = 0.0
    features["AAA.IS"].loc[1, "execution_open"] = 999.0
    features["AAA.IS"].loc[2, "execution_open"] = 105.0

    result = _run(panel, features, configs)
    order = result.orders.iloc[0]

    assert order["actual_execution_date"] == dates[2]
    assert order["raw_execution_price"] == pytest.approx(105.0)
    assert order["execution_delay_sessions"] == 1


def test_stale_valuation_is_not_used_as_an_execution_price() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "AAA.IS", dates[1], "SELL")
    features["AAA.IS"].loc[2, "volume"] = 0.0
    features["AAA.IS"].loc[2, "execution_open"] = 999.0
    features["AAA.IS"].loc[3, "execution_open"] = 101.0

    result = _run(panel, features, configs)
    stale = result.position_daily_audit.loc[result.position_daily_audit["date"].eq(dates[2])]
    sell = result.orders.loc[result.orders["side"].eq("SELL")].iloc[0]

    assert stale.iloc[0]["stale_valuation"]
    assert stale.iloc[0]["raw_close_used"] == pytest.approx(100.0)
    assert sell["actual_execution_date"] == dates[3]
    assert sell["raw_execution_price"] == pytest.approx(101.0)


def test_candidate_ties_use_symbol_as_deterministic_final_key() -> None:
    symbols = ("CCC.IS", "AAA.IS", "BBB.IS")
    panel, features, configs, dates = _inputs(symbols)
    for symbol in symbols:
        _signal(panel, symbol, dates[0], "BUY", score=80.0, rank=1.0, risk=50.0)

    result = _run(panel, features, configs, max_positions=2)
    filled = result.orders.query("side == 'BUY' and order_status == 'filled'")

    assert filled["symbol"].tolist() == ["AAA.IS", "BBB.IS"]


def test_open_position_and_daily_pnl_reconcile() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    features["AAA.IS"].loc[2:, "execution_close"] = 110.0

    result = _run(panel, features, configs)
    final = result.daily.iloc[-1]

    assert abs(final["reconciliation_error"]) < 1e-8
    assert final["total_portfolio_equity"] == pytest.approx(
        1_000_000.0
        + final["cumulative_realized_pnl"]
        + final["unrealized_pnl"]
        + final["cumulative_dividends"]
        - final["cumulative_transaction_costs"]
    )


def test_realized_unrealized_dividends_and_costs_reconcile() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",), periods=6)
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "AAA.IS", dates[3], "SELL")
    features["AAA.IS"].loc[2, "dividend"] = 1.5
    features["AAA.IS"].loc[4, "execution_open"] = 115.0
    features["AAA.IS"].loc[4:, "execution_close"] = 115.0

    result = _run(panel, features, configs, starting_capital=10_000.0)
    final = result.daily.iloc[-1]

    assert result.open_positions.empty
    assert final["unrealized_pnl"] == pytest.approx(0.0)
    assert abs(final["reconciliation_error"]) < 1e-8
    assert final["total_portfolio_equity"] == pytest.approx(
        10_000.0
        + final["cumulative_realized_pnl"]
        + final["cumulative_dividends"]
        - final["cumulative_transaction_costs"]
    )


def test_repeated_runs_are_identical() -> None:
    panel, features, configs, dates = _inputs()
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "AAA.IS", dates[3], "SELL")
    first = _run(panel, features, configs)
    second = _run(panel, features, configs)

    pd.testing.assert_frame_equal(first.daily, second.daily)
    pd.testing.assert_frame_equal(first.orders, second.orders)
    pd.testing.assert_frame_equal(first.trades, second.trades)


def test_audit_identifiers_and_position_rows_are_unique() -> None:
    panel, features, configs, dates = _inputs()
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "BBB.IS", dates[0], "BUY")
    result = _run(panel, features, configs)

    assert not result.orders["order_id"].duplicated().any()
    assert not result.open_positions["symbol"].duplicated().any()
    assert not result.position_daily_audit.duplicated(["symbol", "date"]).any()


def test_stress_runs_reuse_frozen_scores_without_mutation() -> None:
    panel, features, configs, dates = _inputs()
    _signal(panel, "AAA.IS", dates[0], "BUY")
    context = build_portfolio_context(panel, features, configs)
    panel_before = context.panel.copy(deep=True)
    features_before = {symbol: frame.copy(deep=True) for symbol, frame in context.features.items()}

    PortfolioBacktestEngine(PortfolioConfig(max_positions=3)).run_prepared(context)
    PortfolioBacktestEngine(PortfolioConfig(transaction_cost_rate=0.005)).run_prepared(context)

    pd.testing.assert_frame_equal(context.panel, panel_before)
    for symbol in features:
        pd.testing.assert_frame_equal(context.features[symbol], features_before[symbol])


def test_xu100_benchmark_alignment_uses_only_past_values() -> None:
    config = PortfolioConfig(starting_capital=1_000.0, transaction_cost_rate=0.0)
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "market_close": [100.0, 900.0],
        }
    )
    curve = build_xu100_benchmark(
        pd.Series(pd.to_datetime(["2024-01-03", "2024-01-04"])), panel, config
    )

    assert curve["market_close"].tolist() == [100.0, 100.0]
    assert curve["total_equity"].tolist() == [1_000.0, 1_000.0]


def test_cross_symbol_feature_contamination_is_rejected() -> None:
    panel, features, configs, _ = _inputs(("AAA.IS",))
    features["AAA.IS"].loc[2, "source_symbol"] = "BBB.IS"

    with pytest.raises(ValueError, match="Cross-symbol contamination"):
        prepare_portfolio_inputs(panel, features, configs)


def test_pending_exit_without_future_session_is_open_at_end() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",), periods=3)
    _signal(panel, "AAA.IS", dates[0], "BUY")
    _signal(panel, "AAA.IS", dates[1], "SELL")
    features["AAA.IS"].loc[2, "volume"] = 0.0

    result = _run(panel, features, configs)
    sell = result.orders.loc[result.orders["side"].eq("SELL")].iloc[0]

    assert sell["order_status"] == "open_at_end"
    assert sell["rejection_reason"] == "no_future_valid_session"
    assert result.open_positions.iloc[0]["open_at_end"]


def test_calendar_year_contribution_reconciles_to_equity_change() -> None:
    panel, features, configs, dates = _inputs(("AAA.IS",))
    _signal(panel, "AAA.IS", dates[0], "BUY")
    features["AAA.IS"].loc[2, "dividend"] = 1.0
    features["AAA.IS"].loc[3:, "execution_close"] = 110.0

    result = _run(panel, features, configs)
    contribution = build_calendar_year_contribution(result)

    assert contribution["reconciliation_error"].abs().max() < 1e-8
    assert contribution["net_contribution"].sum() == pytest.approx(
        result.daily["total_portfolio_equity"].iloc[-1]
        - result.config.starting_capital
    )


def test_trade_concentration_uses_total_net_completed_trade_profit() -> None:
    symbols = ("AAA.IS", "BBB.IS", "CCC.IS", "DDD.IS")
    panel, features, configs, dates = _inputs(symbols)
    for symbol in symbols:
        _signal(panel, symbol, dates[0], "BUY")
        _signal(panel, symbol, dates[1], "SELL")
    result = _run(panel, features, configs, transaction_cost_rate=0.0)
    trades = result.trades.copy()
    trades["net_pnl"] = [100.0, 80.0, 60.0, -100.0]
    daily = result.daily.copy()
    daily.loc[daily.index[-1], "total_portfolio_equity"] = (
        result.config.starting_capital + trades["net_pnl"].sum()
    )
    synthetic = replace(result, trades=trades, daily=daily)

    *_, concentration = build_contribution_tables(synthetic)

    assert concentration["best_three_trade_concentration"] == pytest.approx(
        (100.0 + 80.0 + 60.0) / (100.0 + 80.0 + 60.0 - 100.0)
    )
