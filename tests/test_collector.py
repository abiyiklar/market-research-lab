from pathlib import Path

import pandas as pd

from bist_research.collector import (
    clean_price_data,
    collect_corporate_action_history,
    download_symbol,
    find_zero_volume_records,
    safe_symbol_name,
    save_frame,
)
from bist_research.config import SymbolConfig


def test_safe_symbol_name_handles_market_symbols() -> None:
    assert safe_symbol_name("TRY=X") == "TRY_X"
    assert safe_symbol_name("BZ=F") == "BZ_F"
    assert safe_symbol_name("TUPRS.IS") == "TUPRS.IS"


def test_clean_price_data_drops_missing_close_records() -> None:
    frame = pd.DataFrame(
        {
            "Symbol": ["TUPRS.IS", "TUPRS.IS", "TUPRS.IS"],
            "Date": pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]),
            "Close": [10.5, None, 11.0],
            "Volume": [100, 200, 0],
        }
    )

    cleaned = clean_price_data(frame)

    assert cleaned["Close"].isna().sum() == 0
    assert cleaned["Date"].tolist() == sorted(cleaned["Date"].tolist())
    assert len(cleaned) == 2


def test_zero_volume_report_is_only_for_equities() -> None:
    frame = pd.DataFrame(
        {
            "Symbol": ["TUPRS.IS", "TUPRS.IS"],
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Open": [1.0, 2.0],
            "High": [1.2, 2.2],
            "Low": [0.9, 1.9],
            "Close": [1.1, 2.1],
            "Volume": [0, 100],
        }
    )

    equity_report = find_zero_volume_records(SymbolConfig("TUPRS.IS", "equity", "Tupras"), frame)
    index_report = find_zero_volume_records(SymbolConfig("XU100.IS", "index", "BIST 100"), frame)

    assert len(equity_report) == 1
    assert equity_report.iloc[0]["Volume"] == 0
    assert index_report.empty


def test_download_symbol_retries_and_normalizes_response() -> None:
    calls: list[str] = []

    def flaky_downloader(**kwargs: object) -> pd.DataFrame:
        calls.append(str(kwargs.get("tickers") or kwargs.get("symbol")))
        if len(calls) == 1:
            raise RuntimeError("temporary API error")
        return pd.DataFrame(
            {
                "Open": [1.0],
                "High": [1.2],
                "Low": [0.9],
                "Close": [1.1],
                "Adj Close": [1.1],
                "Volume": [100],
            },
            index=pd.DatetimeIndex(["2024-01-01"], name="Date"),
        )

    frame = download_symbol(
        "TUPRS.IS",
        start_date="2024-01-01",
        downloader=flaky_downloader,
        retries=2,
        backoff_seconds=0,
    )

    assert calls == ["TUPRS.IS", "TUPRS.IS"]
    assert frame.iloc[0]["Symbol"] == "TUPRS.IS"
    assert frame.iloc[0]["Close"] == 1.1


def test_download_requests_actions_and_repair_mode() -> None:
    captured: dict[str, object] = {}

    def downloader(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "Open": [1.0],
                "High": [1.2],
                "Low": [0.9],
                "Close": [1.1],
                "Adj Close": [1.0],
                "Volume": [100],
                "Dividends": [0.1],
                "Stock Splits": [0.0],
                "Repaired?": [False],
            },
            index=pd.DatetimeIndex(["2024-01-01"], name="Date"),
        )

    frame = download_symbol("TUPRS.IS", downloader=downloader)

    assert captured["actions"] is True
    assert captured["repair"] is True
    assert {"Dividends", "Stock Splits", "Repaired?"}.issubset(frame.columns)


def test_corporate_action_history_is_saved_in_separate_directory(tmp_path: Path) -> None:
    def downloader(**kwargs: object) -> pd.DataFrame:
        del kwargs
        return pd.DataFrame(
            {
                "Open": [100.0, 90.0],
                "High": [101.0, 91.0],
                "Low": [99.0, 89.0],
                "Close": [100.0, 90.0],
                "Adj Close": [90.0, 90.0],
                "Volume": [100, 100],
                "Dividends": [0.0, 10.0],
                "Stock Splits": [0.0, 0.0],
                "Repaired?": [False, True],
            },
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], name="Date"),
        )

    output = tmp_path / "processed" / "corporate_actions"
    history_paths, audit_paths, audit = collect_corporate_action_history(
        SymbolConfig("TUPRS.IS", "equity", "Tupras"),
        output_dir=output,
        downloader=downloader,
    )

    assert history_paths.parquet.exists()
    assert audit_paths.csv.exists()
    assert int(audit["corporate_action_flag"].sum()) == 1
    assert not (tmp_path / "raw").exists()


def test_save_frame_writes_csv_and_parquet(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame({"Symbol": ["TRY=X"], "Close": [30.0]})

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        Path(path).write_text("parquet placeholder", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    paths = save_frame(frame, tmp_path, "TRY_X")

    assert paths.csv.exists()
    assert paths.parquet.exists()
    assert "TRY=X" in paths.csv.read_text(encoding="utf-8")
