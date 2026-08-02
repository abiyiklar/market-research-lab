from pathlib import Path

import pandas as pd

from bist_research.features import (
    EXTERNAL_MARKETS,
    MARKET_FILES,
    FEATURE_COLUMNS,
    add_features,
    build_feature_dataset,
    merge_processed_markets,
    save_feature_dataset,
)


def _market_frame(symbol: str, dates: pd.DatetimeIndex, close: list[float]) -> pd.DataFrame:
    close_series = pd.Series(close, dtype="float64")
    return pd.DataFrame(
        {
            "Symbol": symbol,
            "Date": dates,
            "Open": close_series - 0.5,
            "High": close_series + 1.0,
            "Low": close_series - 1.0,
            "Close": close_series,
            "Adj Close": close_series,
            "Volume": pd.Series(range(1_000, 1_000 + len(dates)), dtype="int64"),
        }
    )


def _write_market_inputs(processed_dir: Path, dates: pd.DatetimeIndex) -> None:
    symbols = {
        "tuprs": "TUPRS.IS",
        "xu100": "XU100.IS",
        "xu030": "XU030.IS",
        "xusin": "XUSIN.IS",
        "usdtry": "TRY=X",
        "brent": "BZ=F",
        "wti": "CL=F",
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    for offset, (market, filename) in enumerate(MARKET_FILES.items(), start=1):
        close = [float(offset * 100 + index) for index in range(len(dates))]
        _market_frame(symbols[market], dates, close).to_parquet(processed_dir / filename, index=False)


def _feature_input(row_count: int = 260) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=row_count, freq="B")
    index = pd.Series(range(row_count), dtype="float64")
    frame = pd.DataFrame(
        {
            "date": dates,
            "tuprs_close": 100.0 + index,
            "tuprs_high": 101.0 + index,
            "tuprs_low": 99.0 + index,
            "tuprs_volume": 1_000.0 + index,
            "xu100_close": 50.0 + index / 2,
            "xusin_close": 25.0 + index / 4,
            "brent_close": 80.0 + index,
            "usdtry_close": 20.0 + index / 10,
        }
    )
    return frame


def test_merge_uses_tuprs_calendar_and_only_forward_fills(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    tuprs_dates = pd.date_range("2024-01-02", periods=3, freq="D")
    _write_market_inputs(processed_dir, tuprs_dates)

    xu100 = _market_frame("XU100.IS", tuprs_dates[1:2], [200.0])
    xu100.to_parquet(processed_dir / MARKET_FILES["xu100"], index=False)

    merged, stats = merge_processed_markets(processed_dir)

    assert merged["date"].tolist() == tuprs_dates.tolist()
    assert pd.isna(merged.loc[0, "xu100_close"])
    assert merged.loc[1, "xu100_close"] == 200.0
    assert merged.loc[2, "xu100_close"] == 200.0
    assert stats["xu100"] == {
        "missing_close_before_fill": 2,
        "missing_close_after_fill": 1,
    }
    assert set(stats) == set(EXTERNAL_MARKETS)


def test_features_have_expected_windows_and_warmup_marker() -> None:
    features = add_features(_feature_input())

    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert features["is_indicator_warmup"].sum() == 200
    assert features.loc[199, "is_indicator_warmup"]
    assert not features.loc[200, "is_indicator_warmup"]
    assert features.loc[20, "relative_momentum_20d"] == 0.0
    assert features.loc[19, "volume_average_20d"] == 1_009.5
    assert features.loc[200, "ema_200"] > 0
    assert features.loc[20, "rsi_14"] == 100.0
    assert features.loc[20, "atr_14"] == 2.0


def test_changing_future_values_does_not_change_past_features() -> None:
    original_input = _feature_input()
    changed_input = original_input.copy()
    changed_input.loc[len(changed_input) - 1, "tuprs_close"] *= 10
    changed_input.loc[len(changed_input) - 1, "tuprs_high"] *= 10

    original = add_features(original_input)
    changed = add_features(changed_input)

    pd.testing.assert_frame_equal(original.iloc[:-1], changed.iloc[:-1])


def test_pipeline_saves_features_and_quality_summary(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "features"
    dates = pd.date_range("2023-01-02", periods=240, freq="B")
    _write_market_inputs(processed_dir, dates)

    features, summary = build_feature_dataset(processed_dir)
    paths = save_feature_dataset(features, summary, output_dir)

    assert len(features) == len(dates)
    assert paths.csv.exists()
    assert paths.parquet.exists()
    assert paths.summary.exists()
    summary_metrics = set(summary["metric"])
    assert {"row_count", "indicator_warmup_rows", "complete_post_warmup_rows"}.issubset(summary_metrics)
    assert int(summary.loc[summary["metric"].eq("indicator_warmup_rows"), "value"].iloc[0]) == 200
