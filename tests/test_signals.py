from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bist_research.config import INITIAL_BIST_SYMBOLS, SymbolConfig
from bist_research.signals.pipeline import CURRENT_SIGNAL_COLUMNS, run_signal_pipeline
from bist_research.signals.rules import add_signal_rules
from bist_research.signals.scoring import COMPONENT_COLUMNS, build_score_panel, prepare_symbol_inputs
from bist_research.signals.study import build_buy_event_study


def _feature_frame(
    config: SymbolConfig,
    strength: int,
    periods: int = 320,
    fallback: bool = False,
) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    growth = 0.0003 + strength * 0.00008
    close = pd.Series(100.0 * np.exp(np.arange(periods) * growth))
    market = pd.Series(1_000.0 * np.exp(np.arange(periods) * 0.00035))
    sector = pd.Series(700.0 * np.exp(np.arange(periods) * (0.00032 + strength * 0.00001)))
    daily_return = close.pct_change(fill_method=None)
    market_benchmark = config.market_benchmark
    sector_benchmark = market_benchmark if fallback else config.preferred_sector_benchmark or market_benchmark
    return pd.DataFrame(
        {
            "symbol": config.symbol,
            "source_symbol": config.symbol,
            "date": dates,
            "execution_open": close * 0.999,
            "execution_high": close * 1.01,
            "execution_low": close * 0.99,
            "execution_close": close,
            "adjusted_close": close,
            "volume": 1_000_000.0 + strength * 100_000 + np.arange(periods) * 100,
            "dividend": 0.0,
            "stock_split": 0.0,
            "repaired_data": False,
            "quality_repaired": False,
            "collection_timestamp": "2026-01-01T00:00:00+00:00",
            "signal_open": close * 0.999,
            "signal_high": close * 1.01,
            "signal_low": close * 0.99,
            "signal_close": close,
            "market_close": market,
            "sector_close": market if fallback else sector,
            "market_benchmark": market_benchmark,
            "sector_benchmark": sector_benchmark,
            "daily_return": daily_return,
            "ema_20": close * (0.997 - strength * 0.0001),
            "ema_50": close * (0.990 - strength * 0.0001),
            "ema_100": close * (0.980 - strength * 0.0001),
            "ema_200": close * (0.960 - strength * 0.0001),
            "rsi_14": 50.0 + strength,
            "atr_14": close * (0.015 + (8 - strength) * 0.0005),
            "volatility_20": 0.30 - strength * 0.01,
            "volatility_60": 0.32 - strength * 0.01,
            "volume_average_20": 1_000_000.0,
            "volume_ratio_20": 1.0 + strength * 0.05,
            "relative_strength_market": close / market,
            "relative_strength_sector": close / (market if fallback else sector),
            "relative_momentum_market_20": 0.005 + strength * 0.003,
            "relative_momentum_market_60": 0.010 + strength * 0.004,
            "relative_momentum_market_120": 0.015 + strength * 0.005,
            "relative_momentum_sector_20": 0.004 + strength * 0.002,
            "relative_momentum_sector_60": 0.008 + strength * 0.003,
            "relative_momentum_sector_120": 0.012 + strength * 0.004,
            "is_indicator_warmup": False,
            "quality_valid_ohlc": True,
            "quality_positive_volume": True,
            "quality_market_missing": False,
            "quality_sector_missing": False,
            "quality_sector_fallback": fallback,
        }
    )


def _eight_symbol_panel() -> tuple[tuple[SymbolConfig, ...], dict[str, pd.DataFrame]]:
    configs = INITIAL_BIST_SYMBOLS[:8]
    frames = {
        config.symbol: _feature_frame(config, index, fallback=config.symbol == "ASELS.IS")
        for index, config in enumerate(configs, start=1)
    }
    return configs, frames


def _rule_frame(periods: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2025-01-02", periods=periods, freq="B")
    frame = pd.DataFrame(
        {
            "symbol": "TEST.IS",
            "date": dates,
            "_stock_session_index": np.arange(periods),
            "signal_open": 100.0,
            "signal_high": 101.0,
            "signal_low": 99.0,
            "signal_close": 100.0,
            "market_close": 1_000.0 + np.arange(periods),
            "sector_close": 700.0 + np.arange(periods),
            "sector_benchmark": "XUSIN.IS",
            "sector_score_available": True,
            "final_score": 60.0,
            "score_state": "NEUTRAL",
            "ema_20": 99.0,
            "ema_50": 98.0,
            "ema_100": 97.0,
            "ema_200": 96.0,
            "ema_20_slope_10": 0.01,
            "rsi_14": 50.0,
            "atr_14": 2.0,
            "volume_ratio_20": 1.30,
            "relative_momentum_market_60": 0.10,
            "quality_repaired": False,
            "repaired_data": False,
        }
    )
    return frame


def test_scores_stay_in_range_and_rank_same_date_cross_section() -> None:
    configs, frames = _eight_symbol_panel()
    panel = build_score_panel(frames, configs)
    latest = panel.loc[panel["date"].eq(panel["date"].max())]
    scores = latest["final_score"].dropna()

    assert len(scores) == 8
    assert scores.between(0, 100, inclusive="both").all()
    assert set(latest["cross_section_rank"].dropna().astype(int)) == set(range(1, 9))
    assert latest.loc[latest["final_score"].idxmax(), "cross_section_rank"] == 1


def test_minimum_eight_symbol_rule_marks_insufficient_cross_section() -> None:
    configs, frames = _eight_symbol_panel()
    one_symbol = configs[-1].symbol
    latest_date = frames[one_symbol]["date"].max()
    frames[one_symbol] = frames[one_symbol].loc[frames[one_symbol]["date"].ne(latest_date)]

    panel = build_score_panel(frames, configs)
    latest = panel.loc[panel["date"].eq(latest_date)]

    assert latest["insufficient_cross_section"].all()
    assert latest["cross_section_size"].eq(7).all()
    assert latest["final_score"].isna().all()


def test_sector_fallback_is_neutral_and_not_double_counted() -> None:
    configs, frames = _eight_symbol_panel()
    panel = build_score_panel(frames, configs)
    fallback = panel.loc[
        panel["symbol"].eq("ASELS.IS") & panel["final_score"].notna()
    ]

    assert not fallback["sector_score_available"].any()
    assert fallback["sector_fallback_neutralized"].all()
    assert fallback["sector_relative_strength_score"].eq(50.0).all()


def test_future_price_change_does_not_alter_earlier_scores() -> None:
    configs, frames = _eight_symbol_panel()
    changed = {symbol: frame.copy() for symbol, frame in frames.items()}
    symbol = configs[0].symbol
    changed[symbol].loc[changed[symbol].index[-1], ["signal_close", "signal_high"]] *= 10

    original_panel = build_score_panel(frames, configs)
    changed_panel = build_score_panel(changed, configs)
    cutoff = original_panel["date"].max()
    columns = ["symbol", "date", *COMPONENT_COLUMNS, "final_score"]

    pd.testing.assert_frame_equal(
        original_panel.loc[original_panel["date"].lt(cutoff), columns].reset_index(drop=True),
        changed_panel.loc[changed_panel["date"].lt(cutoff), columns].reset_index(drop=True),
    )


def test_future_benchmark_change_does_not_alter_earlier_scores() -> None:
    configs, frames = _eight_symbol_panel()
    changed = {symbol: frame.copy() for symbol, frame in frames.items()}
    symbol = configs[1].symbol
    last = changed[symbol].index[-1]
    changed[symbol].loc[last, "market_close"] *= 5
    changed[symbol].loc[last, "relative_strength_market"] *= 0.2
    changed[symbol].loc[last, "relative_momentum_market_60"] = -0.9

    original_panel = build_score_panel(frames, configs)
    changed_panel = build_score_panel(changed, configs)
    cutoff = original_panel["date"].max()

    pd.testing.assert_series_equal(
        original_panel.loc[original_panel["date"].lt(cutoff), "final_score"].reset_index(drop=True),
        changed_panel.loc[changed_panel["date"].lt(cutoff), "final_score"].reset_index(drop=True),
    )


def test_breakout_uses_previous_55_session_high_with_shift() -> None:
    frame = _rule_frame()
    frame.loc[55, ["final_score", "score_state", "signal_close", "signal_high"]] = [
        80.0,
        "STRONG",
        102.0,
        150.0,
    ]

    signals = add_signal_rules(frame)

    assert signals.loc[55, "prior_55_session_high"] == 101.0
    assert bool(signals.loc[55, "breakout_buy_condition"])
    assert signals.loc[55, "signal_type"] == "BREAKOUT_BUY"


def test_breakdown_uses_previous_20_session_low_with_shift() -> None:
    frame = _rule_frame()
    frame.loc[20, ["signal_close", "signal_low", "final_score", "score_state"]] = [
        98.0,
        50.0,
        70.0,
        "POSITIVE",
    ]

    signals = add_signal_rules(frame)

    assert signals.loc[20, "prior_20_session_low"] == 99.0
    assert bool(signals.loc[20, "is_breakdown_exit"])
    assert signals.loc[20, "primary_exit_reason"] == "BREAKDOWN_EXIT"


def test_pullback_entry_conditions_and_next_valid_session_execution() -> None:
    frame = _rule_frame(periods=4)
    frame.loc[1, ["final_score", "score_state", "ema_20", "ema_50"]] = [72.0, "POSITIVE", 99.5, 99.0]
    frame.loc[2:, "date"] = pd.to_datetime(["2025-01-10", "2025-01-14"])
    frame.loc[2, "signal_open"] = 101.25

    signals = add_signal_rules(frame)

    assert bool(signals.loc[1, "new_pullback_buy"])
    assert signals.loc[1, "decision_label"] == "BUY"
    assert signals.loc[1, "signal_type"] == "PULLBACK_BUY"
    assert signals.loc[1, "executable_entry_date"] == pd.Timestamp("2025-01-10")
    assert signals.loc[1, "executable_entry_signal_open"] == 101.25
    assert signals.loc[2, "decision_label"] == "HOLD"


def test_breakout_entry_conditions() -> None:
    frame = _rule_frame()
    frame.loc[55, ["final_score", "score_state", "signal_close", "signal_high"]] = [
        80.0,
        "STRONG",
        102.0,
        102.5,
    ]

    signals = add_signal_rules(frame)

    assert bool(signals.loc[55, "new_breakout_buy"])
    assert signals.loc[55, "decision_label"] == "BUY"


def test_score_exit_requires_two_consecutive_valid_sessions() -> None:
    frame = _rule_frame(periods=5)
    frame.loc[1:2, "final_score"] = [45.0, 44.0]
    frame.loc[1:2, "score_state"] = "WEAKENING"

    signals = add_signal_rules(frame)

    assert not bool(signals.loc[1, "is_score_exit"])
    assert signals.loc[1, "decision_label"] == "WEAKENING"
    assert bool(signals.loc[2, "is_score_exit"])
    assert signals.loc[2, "decision_label"] == "SELL"


def test_persistent_exit_condition_creates_one_sell_event() -> None:
    frame = _rule_frame(periods=5)
    frame.loc[1:3, "final_score"] = [45.0, 44.0, 43.0]
    frame.loc[1:3, "score_state"] = "WEAKENING"

    signals = add_signal_rules(frame)

    assert signals.loc[1:3, "has_confirmed_exit"].tolist() == [False, True, True]
    assert signals.loc[1:3, "is_sell_event"].tolist() == [False, True, False]
    assert signals.loc[2:3, "decision_label"].eq("SELL").all()


def test_cross_symbol_contamination_is_rejected() -> None:
    config = INITIAL_BIST_SYMBOLS[0]
    frame = _feature_frame(config, 1)
    frame.loc[10, "source_symbol"] = "OTHER.IS"

    with pytest.raises(ValueError, match="Cross-symbol source"):
        prepare_symbol_inputs(config, frame)


def test_missing_stock_dates_are_not_forward_filled() -> None:
    configs, frames = _eight_symbol_panel()
    symbol = configs[0].symbol
    missing_date = frames[symbol].loc[300, "date"]
    frames[symbol] = frames[symbol].drop(index=300)

    panel = build_score_panel(frames, configs)

    assert panel.loc[panel["symbol"].eq(symbol) & panel["date"].eq(missing_date)].empty


def test_repaired_row_cannot_create_new_buy() -> None:
    frame = _rule_frame(periods=4)
    frame.loc[1, ["final_score", "score_state", "ema_20", "ema_50"]] = [72.0, "POSITIVE", 99.5, 99.0]
    frame.loc[1, "quality_repaired"] = True

    signals = add_signal_rules(frame)

    assert not bool(signals.loc[1, "is_buy_event"])


def test_overlapping_buy_events_are_marked() -> None:
    frame = _rule_frame(periods=100)
    frame["is_buy_event"] = False
    frame["signal_type"] = ""
    frame.loc[[10, 20], "is_buy_event"] = True
    frame.loc[[10, 20], "signal_type"] = "PULLBACK_BUY"

    events, study = build_buy_event_study(frame)

    assert len(events) == 2
    assert events["overlapping_signal"].all()
    assert events["overlap_cluster_id"].nunique() == 1
    assert study["overlap_cluster_size"].eq(2).all()


def test_pipeline_output_schemas_and_duplicate_checks(tmp_path: Path) -> None:
    configs, frames = _eight_symbol_panel()
    feature_dir = tmp_path / "features"
    for config in configs:
        symbol_dir = feature_dir / config.symbol
        symbol_dir.mkdir(parents=True)
        frames[config.symbol].to_parquet(symbol_dir / "features.parquet", index=False)

    outputs = run_signal_pipeline(
        configs,
        feature_dir=feature_dir,
        output_dir=tmp_path / "signals",
        report_path=tmp_path / "reports" / "score_signal_research.md",
    )
    panel = pd.read_parquet(outputs.panel_parquet)
    current_signals = pd.read_csv(outputs.current_signals)
    buy_events = pd.read_csv(outputs.buy_events)
    sell_events = pd.read_csv(outputs.sell_events)
    event_study = pd.read_csv(outputs.event_study)
    quality_summary = pd.read_csv(outputs.quality_summary)

    assert not panel.duplicated(["symbol", "date"]).any()
    assert set(COMPONENT_COLUMNS).issubset(panel.columns)
    assert {"final_score", "score_state", "decision_label"}.issubset(panel.columns)
    assert set(CURRENT_SIGNAL_COLUMNS).issubset(current_signals.columns)
    assert {"symbol", "signal_date", "signal_type"}.issubset(buy_events.columns)
    assert {"symbol", "signal_date", "primary_exit_reason"}.issubset(sell_events.columns)
    assert {"forward_return_20", "xu100_excess_return_60"}.issubset(event_study.columns)
    assert {"analysis", "horizon_sessions", "median_forward_return"}.issubset(
        quality_summary.columns
    )
    assert outputs.report.exists()
