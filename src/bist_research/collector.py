from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from .backtest.corporate_actions import audit_corporate_actions, detect_price_basis
from .config import (
    BENCHMARK_SYMBOLS,
    DEFAULT_SYMBOLS,
    EQUITY_ASSET_TYPES,
    INITIAL_BIST_PANEL,
    START_DATE,
    SymbolConfig,
)
from .data_quality import MINIMUM_FEATURE_HISTORY, collection_quality_record
from .symbols import panel_configs, safe_symbol_name

Downloader = Callable[..., pd.DataFrame]
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
ACTION_DEFAULTS: dict[str, object] = {
    "Dividends": 0.0,
    "Stock Splits": 0.0,
    "Repaired": False,
}


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
    excluded: bool = False
    exclusion_reason: str = ""
    sector_benchmark_used: str | None = None
    sector_benchmark_fallback: bool = False
    error: str | None = None


class DownloadError(RuntimeError):
    """Raised when a symbol cannot be downloaded after retries."""


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
        "keepna": True,
        "threads": False,
    }
    try:
        return downloader(**kwargs)
    except TypeError:
        return downloader(symbol=symbol, start=start_date, interval=interval)


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
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    normalized = normalized.dropna(subset=["Date"])
    normalized.insert(0, "Symbol", symbol)

    normalized = _canonicalize_repaired_column(normalized)

    for column in PRICE_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    for column, default in ACTION_DEFAULTS.items():
        if column not in normalized.columns:
            normalized[column] = default

    normalized["Source Symbol"] = symbol
    normalized["Collection Timestamp"] = pd.Timestamp.now(tz="UTC").isoformat()
    ordered_columns = [
        "Symbol",
        "Date",
        *PRICE_COLUMNS,
        *ACTION_DEFAULTS,
        "Source Symbol",
        "Collection Timestamp",
    ]
    remaining = [column for column in normalized.columns if column not in ordered_columns]
    return normalized[ordered_columns + remaining]


def _canonicalize_repaired_column(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy()
    if "Repaired?" not in canonical.columns:
        return canonical
    repaired = canonical.pop("Repaired?").fillna(False).astype(bool)
    if "Repaired" in canonical.columns:
        canonical["Repaired"] = canonical["Repaired"].fillna(False).astype(bool) | repaired
    else:
        canonical["Repaired"] = repaired
    return canonical


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

    zero_volume = frame.loc[pd.to_numeric(frame["Volume"], errors="coerce").fillna(-1).eq(0)].copy()
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
        ("Repaired", False),
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
            "Repaired": "repaired_data",
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
    return (
        CollectionResult(
            symbol=config.symbol,
            asset_type=config.asset_type,
            success=True,
            raw_rows=len(raw),
            processed_rows=len(cleaned),
            dropped_missing_close_rows=len(raw) - len(cleaned),
            zero_volume_rows=len(zero_volume),
        ),
        zero_volume,
    )


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
            results.append(CollectionResult(config.symbol, config.asset_type, False, error=str(exc)))

    zero_volume_report = (
        pd.concat(zero_volume_reports, ignore_index=True)
        if zero_volume_reports
        else pd.DataFrame(columns=["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"])
    )
    save_frame(zero_volume_report, processed_dir, "zero_volume_equity_records")
    save_frame(pd.DataFrame(asdict(result) for result in results), processed_dir, "collection_summary")
    active_logger.info("Collection finished: %s/%s successful", sum(r.success for r in results), len(results))
    return results


def merge_incremental_frames(
    previous: pd.DataFrame,
    downloaded: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    combined = pd.concat([previous, downloaded], ignore_index=True, sort=False)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    combined = combined.dropna(subset=["Date"]).sort_values("Date")
    duplicate_count = int(combined["Date"].duplicated(keep="last").sum())
    combined = combined.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    return combined, duplicate_count


def _read_existing_dataset(processed_symbol_dir: Path) -> pd.DataFrame:
    path = processed_symbol_dir / "prices.parquet"
    return _canonicalize_repaired_column(pd.read_parquet(path)) if path.exists() else pd.DataFrame()


def _incremental_start(previous: pd.DataFrame, requested_start: str) -> str:
    if previous.empty:
        return requested_start
    last_date = pd.to_datetime(previous["Date"], errors="raise").max()
    return (last_date + pd.Timedelta(days=1)).date().isoformat()


def _snapshot_stem() -> str:
    return f"prices_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S%fZ')}"


def _collect_panel_dataset(
    config: SymbolConfig,
    raw_symbol_dir: Path,
    processed_symbol_dir: Path,
    start_date: str,
    downloader: Downloader | None,
    retries: int,
    backoff_seconds: float,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    previous = _read_existing_dataset(processed_symbol_dir)
    incremental_start = _incremental_start(previous, start_date)
    logger.info("Collecting %s from %s (existing rows: %s)", config.symbol, incremental_start, len(previous))
    try:
        downloaded = download_symbol(
            config.symbol,
            start_date=incremental_start,
            downloader=downloader,
            retries=retries,
            backoff_seconds=backoff_seconds,
            logger=logger,
        )
    except DownloadError:
        if previous.empty:
            raise
        logger.info("No incremental rows returned for %s; preserving prior data", config.symbol)
        downloaded = pd.DataFrame(columns=previous.columns)

    merged, duplicate_count = merge_incremental_frames(previous, downloaded)
    if not downloaded.empty:
        save_frame(merged, raw_symbol_dir, _snapshot_stem())
    cleaned = clean_price_data(merged)
    save_frame(cleaned, processed_symbol_dir, "prices")
    return merged, cleaned, duplicate_count, len(downloaded)


def collect_panel(
    symbols: Sequence[SymbolConfig],
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    quality_dir: Path = Path("data/quality"),
    start_date: str = START_DATE,
    downloader: Downloader | None = None,
    retries: int = 3,
    backoff_seconds: float = 2.0,
    logger: logging.Logger | None = None,
    minimum_history: int = MINIMUM_FEATURE_HISTORY,
) -> list[CollectionResult]:
    active_logger = logger or configure_logging()
    quality_dir.mkdir(parents=True, exist_ok=True)
    benchmark_configs = {config.symbol: config for config in BENCHMARK_SYMBOLS}
    required_benchmarks = {config.market_benchmark for config in symbols}
    required_benchmarks.update(
        config.preferred_sector_benchmark for config in symbols if config.preferred_sector_benchmark
    )

    benchmark_failures: dict[str, str] = {}
    available_benchmarks: set[str] = set()
    for benchmark_symbol in sorted(required_benchmarks):
        benchmark_config = benchmark_configs.get(
            benchmark_symbol,
            SymbolConfig(benchmark_symbol, "index", benchmark_symbol),
        )
        try:
            _, frame, _, _ = _collect_panel_dataset(
                benchmark_config,
                raw_dir / "benchmarks" / safe_symbol_name(benchmark_symbol),
                processed_dir / "benchmarks" / safe_symbol_name(benchmark_symbol),
                start_date,
                downloader,
                retries,
                backoff_seconds,
                active_logger,
            )
            valid_close_count = int(
                pd.to_numeric(frame.get("Close"), errors="coerce").dropna().gt(0).sum()
            )
            if valid_close_count >= minimum_history:
                available_benchmarks.add(benchmark_symbol)
            else:
                benchmark_failures[benchmark_symbol] = (
                    f"insufficient_history:{valid_close_count}<{minimum_history}"
                )
                active_logger.warning(
                    "Benchmark %s is unusable: %s valid rows; minimum is %s",
                    benchmark_symbol,
                    valid_close_count,
                    minimum_history,
                )
        except Exception as exc:
            benchmark_failures[benchmark_symbol] = str(exc)
            active_logger.error("Benchmark collection failed for %s: %s", benchmark_symbol, exc)

    results: list[CollectionResult] = []
    quality_rows: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    for config in symbols:
        preferred = config.preferred_sector_benchmark or config.market_benchmark
        failed = []
        if config.market_benchmark not in available_benchmarks:
            failed.append(config.market_benchmark)
        if preferred not in available_benchmarks and preferred not in failed:
            failed.append(preferred)

        sector_used = preferred if preferred in available_benchmarks else None
        fallback = False
        fallback_reason = ""
        if sector_used is None and config.market_benchmark in available_benchmarks:
            sector_used = config.market_benchmark
            fallback = preferred != config.market_benchmark
            detail = benchmark_failures.get(preferred, "unavailable")
            fallback_reason = f"{preferred} unavailable ({detail}); using {config.market_benchmark}"
        benchmark_status = {
            "sector_benchmark_used": sector_used,
            "sector_benchmark_fallback": fallback,
            "failed_benchmark_downloads": ";".join(failed),
            "benchmark_fallback_reason": fallback_reason,
        }
        assignments.append({"symbol": config.symbol, **benchmark_status})

        try:
            raw_frame, frame, duplicate_count, downloaded_rows = _collect_panel_dataset(
                config,
                raw_dir / "equities" / safe_symbol_name(config.symbol),
                processed_dir / "equities" / safe_symbol_name(config.symbol),
                start_date,
                downloader,
                retries,
                backoff_seconds,
                active_logger,
            )
            zero_volume = find_zero_volume_records(config, frame)
            save_frame(
                zero_volume,
                processed_dir / "equities" / safe_symbol_name(config.symbol),
                "zero_volume_records",
            )
            quality = collection_quality_record(
                config,
                raw_frame,
                benchmark_status,
                duplicate_date_count=duplicate_count,
                minimum_history=minimum_history,
            )
            quality["downloaded_row_count"] = downloaded_rows
            quality["collection_error"] = ""
            result = CollectionResult(
                symbol=config.symbol,
                asset_type=config.asset_type,
                success=True,
                raw_rows=int(quality["raw_row_count"]),
                processed_rows=len(frame),
                dropped_missing_close_rows=len(raw_frame) - len(frame),
                zero_volume_rows=len(zero_volume),
                excluded=bool(quality["excluded"]),
                exclusion_reason=str(quality["exclusion_reason"]),
                sector_benchmark_used=sector_used,
                sector_benchmark_fallback=fallback,
            )
        except Exception as exc:
            active_logger.exception("Collection failed for %s", config.symbol)
            empty = pd.DataFrame(columns=["Date", *PRICE_COLUMNS, *ACTION_DEFAULTS])
            quality = collection_quality_record(
                config,
                empty,
                benchmark_status,
                minimum_history=minimum_history,
            )
            quality["downloaded_row_count"] = 0
            quality["collection_error"] = str(exc)
            result = CollectionResult(
                config.symbol,
                config.asset_type,
                False,
                excluded=True,
                exclusion_reason=str(quality["exclusion_reason"]),
                sector_benchmark_used=sector_used,
                sector_benchmark_fallback=fallback,
                error=str(exc),
            )

        quality_rows.append(quality)
        pd.DataFrame([quality]).to_csv(
            quality_dir / f"{safe_symbol_name(config.symbol)}_collection_quality.csv",
            index=False,
        )
        results.append(result)

    summary = pd.DataFrame(quality_rows)
    summary = summary.sort_values(
        ["excluded", "data_completeness", "tradable_session_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    summary.insert(0, "completeness_rank", range(1, len(summary) + 1))
    summary.to_csv(quality_dir / "multi_stock_collection_summary.csv", index=False)
    pd.DataFrame(assignments).to_csv(quality_dir / "benchmark_assignments.csv", index=False)
    active_logger.info("Panel collection finished: %s/%s downloads successful", sum(r.success for r in results), len(results))
    return results


def configs_for_symbols(symbols: Iterable[str]) -> tuple[SymbolConfig, ...]:
    known = {config.symbol: config for config in DEFAULT_SYMBOLS}
    return tuple(known.get(symbol, SymbolConfig(symbol, "equity", symbol)) for symbol in symbols)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and clean daily BIST research data.")
    parser.add_argument("--start-date", default=START_DATE, help="Download start date in YYYY-MM-DD format.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Raw output directory.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"), help="Processed output directory.")
    parser.add_argument("--quality-dir", type=Path, default=Path("data/quality"), help="Quality report directory.")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Log directory.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per symbol.")
    parser.add_argument("--backoff-seconds", type=float, default=2.0, help="Linear retry backoff base seconds.")
    parser.add_argument("--universe", choices=[INITIAL_BIST_PANEL], help="Configured research universe.")
    parser.add_argument("--symbols", nargs="+", help="Optional panel symbol override.")
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
    if args.actions_only:
        action_configs = configs_for_symbols(args.symbols) if args.symbols else DEFAULT_SYMBOLS
        action_symbols = [
            config for config in action_configs if config.asset_type in EQUITY_ASSET_TYPES
        ]
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

    if args.universe or args.symbols:
        results = collect_panel(
            symbols=panel_configs(args.universe, args.symbols),
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            quality_dir=args.quality_dir,
            start_date=args.start_date,
            retries=args.retries,
            backoff_seconds=args.backoff_seconds,
            logger=logger,
        )
    else:
        results = collect_all(
            symbols=DEFAULT_SYMBOLS,
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
