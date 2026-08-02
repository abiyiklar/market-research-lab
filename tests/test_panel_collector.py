from pathlib import Path

import pandas as pd

from bist_research.collector import collect_panel, merge_incremental_frames
from bist_research.config import SymbolConfig


def _download_frame(symbol: str, periods: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=periods, freq="B")
    close = pd.Series(range(100, 100 + periods), index=dates, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000.0,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "Repaired": False,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def test_symbol_specific_outputs_are_isolated(tmp_path: Path) -> None:
    configs = (
        SymbolConfig("ASELS.IS", "equity", "Aselsan", "XUTEK.IS"),
        SymbolConfig("AKBNK.IS", "equity", "Akbank", "XBANK.IS"),
    )

    def downloader(**kwargs: object) -> pd.DataFrame:
        symbol = str(kwargs.get("tickers") or kwargs.get("symbol"))
        return _download_frame(symbol)

    results = collect_panel(
        configs,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        quality_dir=tmp_path / "quality",
        downloader=downloader,
        retries=1,
        minimum_history=1,
    )

    assert all(result.success for result in results)
    for config in configs:
        path = tmp_path / "processed" / "equities" / config.symbol / "prices.parquet"
        assert path.exists()
        frame = pd.read_parquet(path)
        assert set(frame["Symbol"]) == {config.symbol}
        assert set(frame["Source Symbol"]) == {config.symbol}
    assert len(list((tmp_path / "raw" / "equities" / "ASELS.IS").glob("prices_*.parquet"))) == 1


def test_failed_download_does_not_stop_other_symbols(tmp_path: Path) -> None:
    configs = (
        SymbolConfig("ASELS.IS", "equity", "Aselsan", "XUTEK.IS"),
        SymbolConfig("AKBNK.IS", "equity", "Akbank", "XBANK.IS"),
    )

    def downloader(**kwargs: object) -> pd.DataFrame:
        symbol = str(kwargs.get("tickers") or kwargs.get("symbol"))
        if symbol == "ASELS.IS":
            raise RuntimeError("simulated equity failure")
        return _download_frame(symbol)

    results = collect_panel(
        configs,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        quality_dir=tmp_path / "quality",
        downloader=downloader,
        retries=1,
        minimum_history=1,
    )

    by_symbol = {result.symbol: result for result in results}
    assert not by_symbol["ASELS.IS"].success
    assert by_symbol["AKBNK.IS"].success
    assert (tmp_path / "processed" / "equities" / "AKBNK.IS" / "prices.parquet").exists()
    summary = pd.read_csv(tmp_path / "quality" / "multi_stock_collection_summary.csv")
    assert set(summary["symbol"]) == {"ASELS.IS", "AKBNK.IS"}


def test_sector_index_fallback_is_explicitly_reported(tmp_path: Path) -> None:
    config = SymbolConfig("ASELS.IS", "equity", "Aselsan", "XUTEK.IS")

    def downloader(**kwargs: object) -> pd.DataFrame:
        symbol = str(kwargs.get("tickers") or kwargs.get("symbol"))
        if symbol == "XUTEK.IS":
            raise RuntimeError("sector index unavailable")
        return _download_frame(symbol)

    result = collect_panel(
        (config,),
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        quality_dir=tmp_path / "quality",
        downloader=downloader,
        retries=1,
        minimum_history=1,
    )[0]

    quality = pd.read_csv(tmp_path / "quality" / "ASELS.IS_collection_quality.csv").iloc[0]
    assert result.success
    assert result.sector_benchmark_fallback
    assert quality["sector_benchmark_used"] == "XU100.IS"
    assert bool(quality["sector_benchmark_fallback"])
    assert quality["failed_benchmark_downloads"] == "XUTEK.IS"
    assert "unavailable" in quality["benchmark_fallback_reason"]


def test_incremental_merge_removes_duplicate_dates_and_keeps_latest_row() -> None:
    previous = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Close": [100.0, 101.0],
        }
    )
    downloaded = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "Close": [101.5, 102.0],
        }
    )

    merged, duplicate_count = merge_incremental_frames(previous, downloaded)

    assert duplicate_count == 1
    assert not merged["Date"].duplicated().any()
    assert merged.loc[merged["Date"].eq(pd.Timestamp("2024-01-03")), "Close"].item() == 101.5
    assert len(merged) == 3
