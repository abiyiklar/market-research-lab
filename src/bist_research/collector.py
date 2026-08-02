from __future__ import annotations

import argparse
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from .config import DEFAULT_SYMBOLS, EQUITY_ASSET_TYPES, START_DATE, SymbolConfig
from .backtest.corporate_actions import audit_corporate_actions, detect_price_basis

Downloader = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class SavedPaths:
    csv: Path
    parquet: Path


@dataclass(frozen=True)
class CollectionResult:
    symbol: str
    asset_type: str
    success: bool
    raw_rows: int = 0
    processed_rows: int = 0
    dropped_missing_close_rows: int = 0
    zero_volume_rows: int = 0
    error: str | None = None


class DownloadError(RuntimeError):
    """Raised when a symbol cannot be downloaded after retries."""


def safe_symbol_name(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("_")
    return cleaned or "symbol"


def configure_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bist_research.collector")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler = logging.FileHandler(log_dir / "collector.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _default_downloader(**kwargs: object) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(**kwargs)


def _call_downloader(
    downloader: Downloader,
    symbol: str,
    start_date: str,
    interval: str,
) -> pd.DataFrame:
    kwargs = {
        "tickers": symbol,
        "start": start_date,
        "interval": interval,
        "progress": False,
        "auto_adjust": False,
        "actions": True,
        "repair": True,
        "threads": False,
    }
    try:
        return downloader(**kwargs)
    except TypeError:
        fallback_kwargs = {
            "symbol": symbol,
            "start": start_date,
            "interval": interval,
        }
        return downloader(**fallback_kwargs)


def normalize_yfinance_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        ticker_level = _find_multiindex_level(normalized.columns, symbol)
        if ticker_level is not None:
            normalized = normalized.xs(symbol, axis=1, level=ticker_level, drop_level=True)
        else:
            normalized.columns = [
                "_".join(str(part) for part in column if str(part))
                for column in normalized.columns.to_flat_index()
            ]

    normalized = normalized.reset_index()
    if "Date" not in normalized.columns:
        normalized = normalized.rename(columns={normalized.columns[0]: "Date"})

    normalized.columns = [str(column).strip() for column in normalized.columns]
    normalized["Date"] = pd.to_datetime(normalized["Date"]).dt.date
    normalized.insert(0, "Symbol", symbol)

    ordered_columns = [
        "Symbol",
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Repaired?",
    ]
    existing_ordered = [column for column in ordered_columns if column in normalized.columns]
    remaining = [column for column in normalized.columns if column not in existing_ordered]
    return normalized[existing_ordered + remaining]


def _find_multiindex_level(columns: pd.MultiIndex, symbol: str) -> int | None:
    for level in range(columns.nlevels):
        if symbol in columns.get_level_values(level):
            return level
    return None


def download_symbol(
    symbol: str,
    start_date: str = START_DATE,
    interval: str = "1d",
    downloader: Downloader | None = None,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    active_downloader = downloader or _default_downloader
    active_logger = logger or logging.getLogger("bist_research.collector")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            frame = _call_downloader(active_downloader, symbol, start_date, interval)
            normalized = normalize_yfinance_frame(frame, symbol)
            if normalized.empty:
                raise DownloadError(f"{symbol} returned no rows")
            return normalized
        except Exception as exc:
            last_error = exc
            active_logger.warning(
                "Download failed for %s on attempt %s/%s: %s",
                symbol,
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(backoff_seconds * attempt)

    raise DownloadError(f"{symbol} failed after {retries} attempts") from last_error


def clean_price_data(frame: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in frame.columns:
        raise ValueError("Downloaded data must contain a Close column")

    cleaned = frame.dropna(subset=["Close"]).copy()
    if "Date" in cleaned.columns:
        cleaned = cleaned.sort_values("Date")
    return cleaned.reset_index(drop=True)


def find_zero_volume_records(config: SymbolConfig, frame: pd.DataFrame) -> pd.DataFrame:
    report_columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    if config.asset_type not in EQUITY_ASSET_TYPES or "Volume" not in frame.columns:
        return pd.DataFrame(columns=report_columns)

    zero_volume = frame.loc[frame["Volume"].fillna(-1).eq(0)].copy()
    available_columns = [column for column in report_columns if column in zero_volume.columns]
    return zero_volume[available_columns].reset_index(drop=True)


def save_frame(
    frame: pd.DataFrame,
    output_dir: Path,
    stem: str,
    *,
    overwrite: bool = True,
) -> SavedPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    parquet_path = output_dir / f"{stem}.parquet"
    if overwrite or not csv_path.exists():
        frame.to_csv(csv_path, index=False)
    if overwrite or not parquet_path.exists():
        frame.to_parquet(parquet_path, index=False)
    return SavedPaths(csv=csv_path, parquet=parquet_path)


def collect_corporate_action_history(
    config: SymbolConfig,
    output_dir: Path = Path("data/processed/corporate_actions"),
    start_date: str = START_DATE,
    downloader: Downloader | None = None,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    logger: logging.Logger | None = None,
) -> tuple[SavedPaths, SavedPaths, pd.DataFrame]:
    """Save action-enriched history and an explicit audit outside raw storage."""
    active_logger = logger or logging.getLogger("bist_research.collector")
    history = download_symbol(
        config.symbol,
        start_date=start_date,
        downloader=downloader,
        retries=retries,
        backoff_seconds=backoff_seconds,
        logger=active_logger,
    )
    for column, default in (
        ("Dividends", 0.0),
        ("Stock Splits", 0.0),
        ("Repaired?", False),
    ):
        if column not in history:
            history[column] = default
    history["price_basis"] = detect_price_basis(history)

    stem = f"{safe_symbol_name(config.symbol)}_actions"
    history_paths = save_frame(history, output_dir, stem)
    feature_like = history.rename(
        columns={
            "Date": "date",
            "Open": "tuprs_open",
            "Close": "tuprs_close",
            "Adj Close": "tuprs_adj_close",
            "Dividends": "dividend_per_share",
            "Stock Splits": "stock_split_factor",
            "Repaired?": "repaired_data",
        }
    )
    feature_like["corporate_action_flag"] = (
        feature_like["dividend_per_share"].fillna(0).ne(0)
        | feature_like["stock_split_factor"].fillna(0).ne(0)
    )
    audit = audit_corporate_actions(feature_like)
    audit_paths = save_frame(audit, output_dir, f"{stem}_audit")
    active_logger.info(
        "Saved %s corporate-action rows for %s (%s)",
        int(feature_like["corporate_action_flag"].sum()),
        config.symbol,
        history["price_basis"].iloc[0],
    )
    return history_paths, audit_paths, audit


def collect_symbol(
    config: SymbolConfig,
    raw_dir: Path,
    processed_dir: Path,
    start_date: str = START_DATE,
    downloader: Downloader | None = None,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    logger: logging.Logger | None = None,
    overwrite_raw: bool = False,
) -> tuple[CollectionResult, pd.DataFrame]:
    active_logger = logger or logging.getLogger("bist_research.collector")
    active_logger.info("Collecting %s from %s", config.symbol, start_date)

    raw = download_symbol(
        config.symbol,
        start_date=start_date,
        downloader=downloader,
        retries=retries,
        backoff_seconds=backoff_seconds,
        logger=active_logger,
    )
    cleaned = clean_price_data(raw)
    zero_volume = find_zero_volume_records(config, cleaned)
    safe_name = safe_symbol_name(config.symbol)

    save_frame(raw, raw_dir, safe_name, overwrite=overwrite_raw)
    save_frame(cleaned, processed_dir, safe_name)

    result = CollectionResult(
        symbol=config.symbol,
        asset_type=config.asset_type,
        success=True,
        raw_rows=len(raw),
        processed_rows=len(cleaned),
        dropped_missing_close_rows=len(raw) - len(cleaned),
        zero_volume_rows=len(zero_volume),
    )
    return result, zero_volume


def collect_all(
    symbols: Sequence[SymbolConfig] = DEFAULT_SYMBOLS,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    start_date: str = START_DATE,
    downloader: Downloader | None = None,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    logger: logging.Logger | None = None,
    overwrite_raw: bool = False,
) -> list[CollectionResult]:
    active_logger = logger or configure_logging()
    results: list[CollectionResult] = []
    zero_volume_reports: list[pd.DataFrame] = []

    for config in symbols:
        try:
            result, zero_volume = collect_symbol(
                config=config,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                start_date=start_date,
                downloader=downloader,
                retries=retries,
                backoff_seconds=backoff_seconds,
                logger=active_logger,
                overwrite_raw=overwrite_raw,
            )
            results.append(result)
            if not zero_volume.empty:
                zero_volume_reports.append(zero_volume)
        except Exception as exc:
            active_logger.exception("Collection failed for %s", config.symbol)
            results.append(
                CollectionResult(
                    symbol=config.symbol,
                    asset_type=config.asset_type,
                    success=False,
                    error=str(exc),
                )
            )

    zero_volume_report = (
        pd.concat(zero_volume_reports, ignore_index=True)
        if zero_volume_reports
        else pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    )
    summary = pd.DataFrame(asdict(result) for result in results)

    save_frame(zero_volume_report, processed_dir, "zero_volume_equity_records")
    save_frame(summary, processed_dir, "collection_summary")
    active_logger.info("Collection finished: %s/%s successful", sum(r.success for r in results), len(results))
    return results


def configs_for_symbols(symbols: Iterable[str]) -> tuple[SymbolConfig, ...]:
    known = {config.symbol: config for config in DEFAULT_SYMBOLS}
    return tuple(known.get(symbol, SymbolConfig(symbol, "equity", symbol)) for symbol in symbols)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and clean daily BIST research data.")
    parser.add_argument("--start-date", default=START_DATE, help="Download start date in YYYY-MM-DD format.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw output directory.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed output directory.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Log directory.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per symbol.")
    parser.add_argument("--backoff-seconds", type=float, default=2.0, help="Linear retry backoff base seconds.")
    parser.add_argument("--symbols", nargs="*", help="Optional symbol override. Defaults to starter universe.")
    parser.add_argument(
        "--actions-only",
        action="store_true",
        help="Save action-enriched equity histories separately without changing raw files.",
    )
    parser.add_argument(
        "--corporate-action-dir",
        type=Path,
        default=Path("data/processed/corporate_actions"),
    )
    parser.add_argument(
        "--overwrite-raw",
        action="store_true",
        help="Explicitly allow replacement of existing raw CSV and Parquet files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging(args.log_dir)
    symbols = configs_for_symbols(args.symbols) if args.symbols else DEFAULT_SYMBOLS
    if args.actions_only:
        action_symbols = [config for config in symbols if config.asset_type in EQUITY_ASSET_TYPES]
        try:
            for config in action_symbols:
                collect_corporate_action_history(
                    config,
                    output_dir=args.corporate_action_dir,
                    start_date=args.start_date,
                    retries=args.retries,
                    backoff_seconds=args.backoff_seconds,
                    logger=logger,
                )
        except Exception:
            logger.exception("Corporate-action collection failed")
            return 1
        return 0
    results = collect_all(
        symbols=symbols,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        start_date=args.start_date,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        logger=logger,
        overwrite_raw=args.overwrite_raw,
    )
    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
