import numpy as np
import pandas as pd
import pytest

from bist_research.config import SymbolConfig
from bist_research.features import GENERIC_FEATURE_COLUMNS, build_signal_prices, build_symbol_features


def _stock_frame(symbol: str, periods: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    close = pd.Series(100.0 + np.arange(periods) * 0.1)
    return pd.DataFrame(
        {
            "Symbol": symbol,
            "Date": dates,
            "Open": close - 0.2,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000.0 + np.arange(periods),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "Repaired": False,
            "Source Symbol": symbol,
            "Collection Timestamp": "2024-01-01T00:00:00+00:00",
        }
    )


def _benchmark(periods: int = 240, base: float = 1_000.0) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=periods, freq="B")
    return pd.DataFrame({"Date": dates, "Close": base + np.arange(periods)})


def _features(stock: pd.DataFrame, symbol: str = "ASELS.IS") -> pd.DataFrame:
    config = SymbolConfig(symbol, "equity", symbol, "XUTEK.IS")
    return build_symbol_features(
        config,
        stock,
        _benchmark(len(stock)),
        _benchmark(len(stock), 500.0),
        "XUTEK.IS",
    )


def test_corporate_actions_are_applied_per_stock_and_raw_prices_stay_unchanged() -> None:
    stock = _stock_frame("ASELS.IS", periods=4)
    stock.loc[:, ["Open", "High", "Low", "Close", "Adj Close"]] = 100.0
    stock.loc[2, ["Open", "High", "Low", "Close", "Adj Close"]] = 95.0
    stock.loc[2, "Dividends"] = 5.0
    other = _stock_frame("AKBNK.IS", periods=4)

    asels = _features(stock)
    akbnk = _features(other, "AKBNK.IS")

    assert asels.loc[2, "daily_return"] == pytest.approx(0.0)
    assert asels.loc[2, "signal_close"] == pytest.approx(asels.loc[1, "signal_close"])
    assert asels.loc[2, "execution_close"] == 95.0
    assert akbnk["dividend"].eq(0).all()


def test_signal_prices_are_split_consistent() -> None:
    stock = _stock_frame("ASELS.IS", periods=3)
    stock.loc[:, ["Open", "High", "Low", "Close", "Adj Close"]] = 100.0
    stock.loc[1:, ["Open", "High", "Low", "Close", "Adj Close"]] = 50.0
    stock.loc[1, "Stock Splits"] = 2.0

    signal = build_signal_prices(stock)

    assert signal.loc[1, "signal_close"] == pytest.approx(100.0)
    assert signal.loc[2, "signal_close"] == pytest.approx(100.0)


def test_future_data_changes_do_not_change_earlier_signal_features() -> None:
    original_stock = _stock_frame("ASELS.IS")
    changed_stock = original_stock.copy()
    changed_stock.loc[len(changed_stock) - 1, "Dividends"] = 25.0
    market = _benchmark()
    changed_market = market.copy()
    changed_market.loc[len(changed_market) - 1, "Close"] *= 10
    sector = _benchmark(base=500.0)
    changed_sector = sector.copy()
    changed_sector.loc[len(changed_sector) - 1, "Close"] *= 5
    config = SymbolConfig("ASELS.IS", "equity", "Aselsan", "XUTEK.IS")

    original = build_symbol_features(config, original_stock, market, sector, "XUTEK.IS")
    changed = build_symbol_features(
        config,
        changed_stock,
        changed_market,
        changed_sector,
        "XUTEK.IS",
    )

    pd.testing.assert_frame_equal(original.iloc[:-1], changed.iloc[:-1])


def test_cross_symbol_rows_are_rejected() -> None:
    stock = _stock_frame("ASELS.IS")
    stock.loc[10, "Symbol"] = "AKBNK.IS"

    with pytest.raises(ValueError, match="Cross-symbol"):
        _features(stock)


def test_only_valid_tradable_sessions_enter_feature_calendar() -> None:
    stock = _stock_frame("ASELS.IS", periods=5)
    stock.loc[1, "Volume"] = 0
    stock.loc[2, "High"] = stock.loc[2, "Low"] - 1

    features = _features(stock)

    assert len(features) == 3
    assert pd.Timestamp(stock.loc[1, "Date"]) not in set(features["date"])
    assert pd.Timestamp(stock.loc[2, "Date"]) not in set(features["date"])
    assert features["quality_valid_ohlc"].all()
    assert features["quality_positive_volume"].all()


def test_generic_feature_schema_and_past_only_benchmark_fill() -> None:
    stock = _stock_frame("ASELS.IS")
    market = _benchmark().iloc[1::2].reset_index(drop=True)
    sector = _benchmark(base=500.0).iloc[1::2].reset_index(drop=True)
    config = SymbolConfig("ASELS.IS", "equity", "Aselsan", "XUTEK.IS")

    features = build_symbol_features(config, stock, market, sector, "XUTEK.IS")

    assert set(GENERIC_FEATURE_COLUMNS).issubset(features.columns)
    assert {"execution_open", "signal_open", "dividend", "stock_split"}.issubset(features.columns)
    assert pd.isna(features.loc[0, "market_close"])
    assert features.loc[2, "market_close"] == features.loc[1, "market_close"]
    assert features["symbol"].eq("ASELS.IS").all()
