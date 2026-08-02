from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .collector import panel_configs, safe_symbol_name
from .config import INITIAL_BIST_PANEL, SymbolConfig
from .data_quality import OHLC_COLUMNS, tradable_session_mask


MARKET_FILES: Mapping[str, str] = {
    "tuprs": "TUPRS.IS.parquet",
    "xu100": "XU100.IS.parquet",
    "xu030": "XU030.IS.parquet",
    "xusin": "XUSIN.IS.parquet",
    "usdtry": "TRY_X.parquet",
    "brent": "BZ_F.parquet",
    "wti": "CL_F.parquet",
}

MARKET_VALUE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
EXTERNAL_MARKETS = tuple(market for market in MARKET_FILES if market != "tuprs")
WARMUP_DAYS = 200
TRADING_DAYS_PER_YEAR = 252

FEATURE_COLUMNS = (
    "tuprs_return_1d",
    "xu100_return_1d",
    "xusin_return_1d",
    "brent_return_1d",
    "usdtry_return_1d",
    "relative_strength_xu100",
    "relative_strength_xusin",
    "relative_momentum_20d",
    "relative_momentum_60d",
    "relative_momentum_120d",
    "volatility_20d",
    "volatility_60d",
    "volume_average_20d",
    "volume_ratio_20d",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",
    "rsi_14",
    "atr_14",
)


@dataclass(frozen=True)
class FeatureOutputPaths:
    csv: Path
    parquet: Path
    summary: Path


@dataclass(frozen=True)
class PanelFeatureResult:
    symbol: str
    success: bool
    rows: int = 0
    post_warmup_missing_values: int = 0
    post_warmup_infinite_values: int = 0
    duplicate_dates: int = 0
    sector_benchmark_used: str | None = None
    sector_benchmark_fallback: bool = False
    excluded: bool = False
    error: str | None = None


GENERIC_FEATURE_COLUMNS = (
    "daily_return",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",
    "rsi_14",
    "atr_14",
    "volatility_20",
    "volatility_60",
    "volume_average_20",
    "volume_ratio_20",
    "relative_strength_market",
    "relative_momentum_market_20",
    "relative_momentum_market_60",
    "relative_momentum_market_120",
    "relative_strength_sector",
    "relative_momentum_sector_20",
    "relative_momentum_sector_60",
    "relative_momentum_sector_120",
)


def _load_market_frame(processed_dir: Path, market: str) -> pd.DataFrame:
    path = processed_dir / MARKET_FILES[market]
    if not path.exists():
        raise FileNotFoundError(f"Processed market file not found: {path}")

    frame = pd.read_parquet(path)
    required_columns = {"Date", "Close"}
    if market == "tuprs":
        required_columns.update({"High", "Low", "Volume"})

    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    prepared = frame.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="raise").dt.normalize()
    prepared = prepared.sort_values("Date").reset_index(drop=True)
    if prepared["Date"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate dates")
    return prepared


def _prefix_market_columns(frame: pd.DataFrame, market: str) -> pd.DataFrame:
    selected_columns = ["Date"]
    if "Symbol" in frame.columns:
        selected_columns.append("Symbol")
    selected_columns.extend(column for column in MARKET_VALUE_COLUMNS if column in frame.columns)

    rename_map = {"Date": "date", "Symbol": f"{market}_symbol"}
    rename_map.update(
        {column: f"{market}_{column.lower().replace(' ', '_')}" for column in MARKET_VALUE_COLUMNS}
    )
    return frame[selected_columns].rename(columns=rename_map)


def merge_processed_markets(
    processed_dir: Path = Path("data/processed"),
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    tuprs = _prefix_market_columns(_load_market_frame(processed_dir, "tuprs"), "tuprs")
    merged = tuprs.sort_values("date").reset_index(drop=True)
    fill_stats: dict[str, dict[str, int]] = {}

    for market in EXTERNAL_MARKETS:
        external = _prefix_market_columns(_load_market_frame(processed_dir, market), market)
        value_columns = [column for column in external.columns if column != "date"]
        merged = merged.merge(
            external,
            on="date",
            how="left",
            sort=False,
            validate="one_to_one",
        )

        close_column = f"{market}_close"
        missing_before = int(merged[close_column].isna().sum())
        merged[value_columns] = merged[value_columns].ffill()
        fill_stats[market] = {
            "missing_close_before_fill": missing_before,
            "missing_close_after_fill": int(merged[close_column].isna().sum()),
        }

    return merged, fill_stats


def _daily_return(series: pd.Series) -> pd.Series:
    return series.pct_change(fill_method=None)


def _relative_momentum(relative_strength: pd.Series, periods: int) -> pd.Series:
    return relative_strength.pct_change(periods=periods, fill_method=None)


def _rsi(close: pd.Series, periods: int = 14) -> pd.Series:
    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)
    average_gain = gains.ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()
    average_loss = losses.ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()
    relative_strength = average_gain / average_loss
    result = 100 - (100 / (1 + relative_strength))
    result = result.mask(average_loss.eq(0) & average_gain.gt(0), 100.0)
    return result.mask(average_loss.eq(0) & average_gain.eq(0), 50.0)


def _atr(frame: pd.DataFrame, periods: int = 14) -> pd.Series:
    previous_close = frame["tuprs_close"].shift(1)
    true_range = pd.concat(
        [
            frame["tuprs_high"] - frame["tuprs_low"],
            (frame["tuprs_high"] - previous_close).abs(),
            (frame["tuprs_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()


def add_features(market_data: pd.DataFrame) -> pd.DataFrame:
    frame = market_data.sort_values("date").reset_index(drop=True).copy()

    frame["tuprs_return_1d"] = _daily_return(frame["tuprs_close"])
    frame["xu100_return_1d"] = _daily_return(frame["xu100_close"])
    frame["xusin_return_1d"] = _daily_return(frame["xusin_close"])
    frame["brent_return_1d"] = _daily_return(frame["brent_close"])
    frame["usdtry_return_1d"] = _daily_return(frame["usdtry_close"])

    frame["relative_strength_xu100"] = frame["tuprs_close"] / frame["xu100_close"]
    frame["relative_strength_xusin"] = frame["tuprs_close"] / frame["xusin_close"]
    for periods in (20, 60, 120):
        frame[f"relative_momentum_{periods}d"] = _relative_momentum(
            frame["relative_strength_xu100"], periods
        )

    for periods in (20, 60):
        frame[f"volatility_{periods}d"] = (
            frame["tuprs_return_1d"].rolling(periods, min_periods=periods).std()
            * (TRADING_DAYS_PER_YEAR**0.5)
        )

    frame["volume_average_20d"] = frame["tuprs_volume"].rolling(20, min_periods=20).mean()
    frame["volume_ratio_20d"] = frame["tuprs_volume"] / frame["volume_average_20d"].replace(0, float("nan"))

    for periods in (20, 50, 100, 200):
        frame[f"ema_{periods}"] = frame["tuprs_close"].ewm(
            span=periods,
            adjust=False,
            min_periods=periods,
        ).mean()

    frame["rsi_14"] = _rsi(frame["tuprs_close"])
    frame["atr_14"] = _atr(frame)
    frame["is_indicator_warmup"] = pd.Series(range(len(frame)), index=frame.index).lt(WARMUP_DAYS)
    return frame


def build_quality_summary(
    features: pd.DataFrame,
    fill_stats: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"category": "dataset", "metric": "row_count", "value": len(features)},
        {"category": "dataset", "metric": "start_date", "value": features["date"].min().date()},
        {"category": "dataset", "metric": "end_date", "value": features["date"].max().date()},
        {
            "category": "dataset",
            "metric": "duplicate_date_count",
            "value": int(features["date"].duplicated().sum()),
        },
        {
            "category": "dataset",
            "metric": "indicator_warmup_rows",
            "value": int(features["is_indicator_warmup"].sum()),
        },
    ]

    for market, stats in fill_stats.items():
        for metric, value in stats.items():
            rows.append({"category": market, "metric": metric, "value": value})

    for feature in FEATURE_COLUMNS:
        rows.append(
            {
                "category": "feature_missing_values",
                "metric": feature,
                "value": int(features[feature].isna().sum()),
            }
        )

    post_warmup = features.loc[~features["is_indicator_warmup"], FEATURE_COLUMNS]
    rows.append(
        {
            "category": "dataset",
            "metric": "complete_post_warmup_rows",
            "value": int(post_warmup.notna().all(axis=1).sum()),
        }
    )
    return pd.DataFrame(rows, columns=["category", "metric", "value"])


def build_feature_dataset(
    processed_dir: Path = Path("data/processed"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged, fill_stats = merge_processed_markets(processed_dir)
    features = add_features(merged)
    summary = build_quality_summary(features, fill_stats)
    return features, summary


def save_feature_dataset(
    features: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path = Path("data/features"),
) -> FeatureOutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "tuprs_features.csv"
    parquet_path = output_dir / "tuprs_features.parquet"
    summary_path = output_dir / "tuprs_feature_summary.csv"

    features.to_csv(csv_path, index=False)
    features.to_parquet(parquet_path, index=False)
    summary.to_csv(summary_path, index=False)
    return FeatureOutputPaths(csv=csv_path, parquet=parquet_path, summary=summary_path)


def run_feature_pipeline(
    processed_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/features"),
) -> FeatureOutputPaths:
    features, summary = build_feature_dataset(processed_dir)
    return save_feature_dataset(features, summary, output_dir)


def build_signal_prices(stock: pd.DataFrame) -> pd.DataFrame:
    """Build a causal total-return series while preserving raw intraday geometry."""

    ordered = stock.sort_values("Date").reset_index(drop=True).copy()
    raw = ordered[list(OHLC_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    dividends = pd.to_numeric(ordered.get("Dividends", 0.0), errors="coerce").fillna(0.0)
    splits = pd.to_numeric(ordered.get("Stock Splits", 0.0), errors="coerce").fillna(0.0)

    signal_close = pd.Series(np.nan, index=ordered.index, dtype="float64")
    if ordered.empty:
        return pd.DataFrame(index=ordered.index, columns=[f"signal_{name.lower()}" for name in OHLC_COLUMNS])
    signal_close.iloc[0] = raw["Close"].iloc[0]

    for index in range(1, len(ordered)):
        previous_raw_close = raw["Close"].iloc[index - 1]
        current_raw_close = raw["Close"].iloc[index]
        split_factor = 1.0
        split = float(splits.iloc[index])
        if split > 0 and np.isfinite(previous_raw_close) and previous_raw_close > 0:
            observed_open_ratio = raw["Open"].iloc[index] / previous_raw_close
            if abs(observed_open_ratio - (1.0 / split)) < abs(observed_open_ratio - 1.0):
                split_factor = split

        if not np.isfinite([previous_raw_close, current_raw_close]).all() or previous_raw_close <= 0:
            continue
        total_return_ratio = ((current_raw_close + dividends.iloc[index]) * split_factor) / previous_raw_close
        signal_close.iloc[index] = signal_close.iloc[index - 1] * total_return_ratio

    scale = signal_close / raw["Close"].replace(0, np.nan)
    return pd.DataFrame(
        {
            "signal_open": raw["Open"] * scale,
            "signal_high": raw["High"] * scale,
            "signal_low": raw["Low"] * scale,
            "signal_close": signal_close,
        }
    )


def _prepare_benchmark(frame: pd.DataFrame, output_column: str) -> pd.DataFrame:
    required = {"Date", "Close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Benchmark data is missing columns: {', '.join(sorted(missing))}")
    prepared = frame[["Date", "Close"]].copy()
    prepared["date"] = pd.to_datetime(prepared.pop("Date"), errors="coerce").dt.tz_localize(None).dt.normalize()
    prepared[output_column] = pd.to_numeric(prepared.pop("Close"), errors="coerce")
    return prepared.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")


def _generic_atr(frame: pd.DataFrame, periods: int = 14) -> pd.Series:
    previous_close = frame["signal_close"].shift(1)
    true_range = pd.concat(
        [
            frame["signal_high"] - frame["signal_low"],
            (frame["signal_high"] - previous_close).abs(),
            (frame["signal_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / periods, adjust=False, min_periods=periods).mean()


def build_symbol_features(
    config: SymbolConfig,
    stock: pd.DataFrame,
    market_benchmark: pd.DataFrame,
    sector_benchmark: pd.DataFrame,
    sector_benchmark_used: str,
    sector_benchmark_fallback: bool = False,
) -> pd.DataFrame:
    if "Symbol" in stock.columns:
        observed_symbols = set(stock["Symbol"].dropna().astype(str).unique())
        if observed_symbols.difference({config.symbol}):
            raise ValueError(f"Cross-symbol rows found for {config.symbol}: {sorted(observed_symbols)}")

    prepared = stock.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    prepared = prepared.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    prepared = prepared.loc[tradable_session_mask(prepared)].reset_index(drop=True)
    if prepared.empty:
        raise ValueError(f"{config.symbol} has no valid tradable sessions")

    features = pd.DataFrame(
        {
            "symbol": config.symbol,
            "date": prepared["Date"],
            "execution_open": pd.to_numeric(prepared["Open"], errors="coerce"),
            "execution_high": pd.to_numeric(prepared["High"], errors="coerce"),
            "execution_low": pd.to_numeric(prepared["Low"], errors="coerce"),
            "execution_close": pd.to_numeric(prepared["Close"], errors="coerce"),
            "adjusted_close": pd.to_numeric(prepared.get("Adj Close"), errors="coerce"),
            "volume": pd.to_numeric(prepared["Volume"], errors="coerce"),
            "dividend": pd.to_numeric(prepared.get("Dividends", 0.0), errors="coerce").fillna(0.0),
            "stock_split": pd.to_numeric(prepared.get("Stock Splits", 0.0), errors="coerce").fillna(0.0),
            "repaired_data": prepared.get("Repaired", pd.Series(False, index=prepared.index)).fillna(False).astype(bool),
            "source_symbol": prepared.get("Source Symbol", prepared.get("Symbol", config.symbol)),
            "collection_timestamp": prepared.get("Collection Timestamp", pd.NA),
        }
    )
    signal = build_signal_prices(prepared)
    features = pd.concat([features, signal], axis=1)

    features = features.merge(
        _prepare_benchmark(market_benchmark, "market_close"),
        on="date",
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        _prepare_benchmark(sector_benchmark, "sector_close"),
        on="date",
        how="left",
        validate="one_to_one",
    )
    features[["market_close", "sector_close"]] = features[["market_close", "sector_close"]].ffill()
    features["market_benchmark"] = config.market_benchmark
    features["sector_benchmark"] = sector_benchmark_used

    features["daily_return"] = _daily_return(features["signal_close"])
    for periods in (20, 50, 100, 200):
        features[f"ema_{periods}"] = features["signal_close"].ewm(
            span=periods,
            adjust=False,
            min_periods=periods,
        ).mean()
    features["rsi_14"] = _rsi(features["signal_close"])
    features["atr_14"] = _generic_atr(features)
    for periods in (20, 60):
        features[f"volatility_{periods}"] = (
            features["daily_return"].rolling(periods, min_periods=periods).std()
            * (TRADING_DAYS_PER_YEAR**0.5)
        )
    features["volume_average_20"] = features["volume"].rolling(20, min_periods=20).mean()
    features["volume_ratio_20"] = features["volume"] / features["volume_average_20"].replace(0, np.nan)

    features["relative_strength_market"] = features["signal_close"] / features["market_close"]
    features["relative_strength_sector"] = features["signal_close"] / features["sector_close"]
    for periods in (20, 60, 120):
        features[f"relative_momentum_market_{periods}"] = _relative_momentum(
            features["relative_strength_market"],
            periods,
        )
        features[f"relative_momentum_sector_{periods}"] = _relative_momentum(
            features["relative_strength_sector"],
            periods,
        )

    features["is_indicator_warmup"] = pd.Series(range(len(features)), index=features.index).lt(WARMUP_DAYS)
    features["quality_valid_ohlc"] = True
    features["quality_positive_volume"] = True
    features["quality_repaired"] = features["repaired_data"]
    features["quality_market_missing"] = features["market_close"].isna()
    features["quality_sector_missing"] = features["sector_close"].isna()
    features["quality_sector_fallback"] = sector_benchmark_fallback
    return features


def _load_panel_prices(processed_dir: Path, category: str, symbol: str) -> pd.DataFrame:
    path = processed_dir / category / safe_symbol_name(symbol) / "prices.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed price file not found: {path}")
    return pd.read_parquet(path)


def _benchmark_assignment(quality_dir: Path, config: SymbolConfig) -> tuple[str, bool]:
    path = quality_dir / "benchmark_assignments.csv"
    if path.exists():
        assignments = pd.read_csv(path)
        matching = assignments.loc[assignments["symbol"].eq(config.symbol)]
        if not matching.empty:
            row = matching.iloc[-1]
            used = row.get("sector_benchmark_used")
            if pd.notna(used):
                return str(used), bool(row.get("sector_benchmark_fallback", False))
    return config.preferred_sector_benchmark or config.market_benchmark, False


def _collection_exclusion(quality_dir: Path, symbol: str) -> tuple[bool, str]:
    path = quality_dir / f"{safe_symbol_name(symbol)}_collection_quality.csv"
    if not path.exists():
        return False, ""
    quality = pd.read_csv(path)
    if quality.empty:
        return False, ""
    excluded = str(quality.iloc[-1].get("excluded", False)).lower() == "true"
    return excluded, str(quality.iloc[-1].get("exclusion_reason", ""))


def build_panel_features(
    symbols: Sequence[SymbolConfig],
    processed_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/features"),
    quality_dir: Path = Path("data/quality"),
) -> list[PanelFeatureResult]:
    quality_dir.mkdir(parents=True, exist_ok=True)
    results: list[PanelFeatureResult] = []
    for config in symbols:
        excluded, exclusion_reason = _collection_exclusion(quality_dir, config.symbol)
        if excluded:
            results.append(
                PanelFeatureResult(
                    symbol=config.symbol,
                    success=False,
                    excluded=True,
                    error=exclusion_reason,
                )
            )
            continue
        try:
            stock = _load_panel_prices(processed_dir, "equities", config.symbol)
            market = _load_panel_prices(processed_dir, "benchmarks", config.market_benchmark)
            sector_used, fallback = _benchmark_assignment(quality_dir, config)
            sector = _load_panel_prices(processed_dir, "benchmarks", sector_used)
            features = build_symbol_features(config, stock, market, sector, sector_used, fallback)
            symbol_output = output_dir / "equities" / safe_symbol_name(config.symbol)
            save_frame = FeatureOutputPaths(
                csv=symbol_output / "features.csv",
                parquet=symbol_output / "features.parquet",
                summary=symbol_output / "feature_quality.csv",
            )
            symbol_output.mkdir(parents=True, exist_ok=True)
            features.to_csv(save_frame.csv, index=False)
            features.to_parquet(save_frame.parquet, index=False)

            post_warmup = features.loc[~features["is_indicator_warmup"], GENERIC_FEATURE_COLUMNS]
            numeric = post_warmup.apply(pd.to_numeric, errors="coerce")
            missing = int(numeric.isna().sum().sum())
            infinite = int(np.isinf(numeric.to_numpy()).sum())
            result = PanelFeatureResult(
                symbol=config.symbol,
                success=True,
                rows=len(features),
                post_warmup_missing_values=missing,
                post_warmup_infinite_values=infinite,
                duplicate_dates=int(features["date"].duplicated().sum()),
                sector_benchmark_used=sector_used,
                sector_benchmark_fallback=fallback,
            )
            pd.DataFrame([result.__dict__]).to_csv(save_frame.summary, index=False)
        except Exception as exc:
            result = PanelFeatureResult(config.symbol, False, error=str(exc))
        results.append(result)

    summary = pd.DataFrame(result.__dict__ for result in results)
    summary.to_csv(quality_dir / "multi_stock_feature_summary.csv", index=False)
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BIST market research feature datasets.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing cleaned Parquet inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Feature dataset output directory.",
    )
    parser.add_argument(
        "--quality-dir",
        type=Path,
        default=Path("data/quality"),
        help="Collection and feature quality report directory.",
    )
    parser.add_argument("--universe", choices=[INITIAL_BIST_PANEL], help="Configured research universe.")
    parser.add_argument("--symbols", nargs="+", help="Optional panel symbol override.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.universe or args.symbols:
        results = build_panel_features(
            panel_configs(args.universe, args.symbols),
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            quality_dir=args.quality_dir,
        )
        for result in results:
            status = "ok" if result.success else "excluded" if result.excluded else "failed"
            print(f"{result.symbol}: {status} ({result.rows} rows)")
        return 0 if all(result.success or result.excluded for result in results) else 1

    paths = run_feature_pipeline(args.processed_dir, args.output_dir)
    print(f"Feature CSV: {paths.csv}")
    print(f"Feature Parquet: {paths.parquet}")
    print(f"Quality summary: {paths.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
