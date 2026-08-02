from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .trading_calendar import only_tradable_tuprs_sessions


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
CORPORATE_ACTION_FILE = Path("corporate_actions/TUPRS.IS_actions.parquet")
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


def _load_market_frame(processed_dir: Path, market: str) -> pd.DataFrame:
    path = processed_dir / MARKET_FILES[market]
    if not path.exists():
        raise FileNotFoundError(f"Processed market file not found: {path}")

    frame = pd.read_parquet(path)
    required_columns = {"Date", "Close"}
    if market == "tuprs":
        required_columns.update({"Open", "High", "Low", "Volume"})

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


def _load_corporate_actions(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / CORPORATE_ACTION_FILE
    columns = [
        "date",
        "dividend_per_share",
        "stock_split_factor",
        "repaired_data",
        "corporate_action_flag",
        "price_basis",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)

    actions = pd.read_parquet(path).copy()
    if "Date" not in actions:
        raise ValueError(f"{path.name} is missing required column: Date")
    actions["date"] = pd.to_datetime(actions["Date"], errors="raise").dt.normalize()
    dividends = pd.to_numeric(
        actions.get("Dividends", pd.Series(0.0, index=actions.index)),
        errors="coerce",
    ).fillna(0.0)
    splits = pd.to_numeric(
        actions.get("Stock Splits", pd.Series(0.0, index=actions.index)),
        errors="coerce",
    ).fillna(0.0)
    repaired = actions.get("Repaired?", pd.Series(False, index=actions.index)).fillna(False)
    price_basis = actions.get(
        "price_basis",
        pd.Series("raw_ohlc_split_basis_unknown", index=actions.index),
    )
    prepared = pd.DataFrame(
        {
            "date": actions["date"],
            "dividend_per_share": dividends,
            "stock_split_factor": splits,
            "repaired_data": repaired.astype(bool),
            "corporate_action_flag": dividends.ne(0) | splits.ne(0),
            "price_basis": price_basis.astype(str),
        }
    )
    if prepared["date"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate dates")
    return prepared.sort_values("date").reset_index(drop=True)


def merge_processed_markets(
    processed_dir: Path = Path("data/processed"),
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    tuprs = _prefix_market_columns(_load_market_frame(processed_dir, "tuprs"), "tuprs")
    merged = only_tradable_tuprs_sessions(tuprs)
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

    actions = _load_corporate_actions(processed_dir)
    merged = merged.merge(actions, on="date", how="left", sort=False, validate="one_to_one")
    merged["dividend_per_share"] = merged["dividend_per_share"].fillna(0.0)
    merged["stock_split_factor"] = merged["stock_split_factor"].fillna(0.0)
    merged["repaired_data"] = merged["repaired_data"].fillna(False).astype(bool)
    merged["corporate_action_flag"] = (
        merged["corporate_action_flag"].fillna(False).astype(bool)
    )
    observed_basis = actions["price_basis"].dropna()
    default_basis = (
        str(observed_basis.iloc[0])
        if not observed_basis.empty
        else "raw_ohlc_split_basis_unknown"
    )
    merged["price_basis"] = merged["price_basis"].fillna(default_basis)

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
    frame = only_tradable_tuprs_sessions(market_data)
    for column, default in (
        ("dividend_per_share", 0.0),
        ("stock_split_factor", 0.0),
        ("repaired_data", False),
        ("corporate_action_flag", False),
        ("price_basis", "raw_ohlc_split_basis_unknown"),
    ):
        if column not in frame:
            frame[column] = default

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
        {
            "category": "dataset",
            "metric": "corporate_action_rows",
            "value": int(features["corporate_action_flag"].sum()),
        },
        {
            "category": "dataset",
            "metric": "repaired_rows",
            "value": int(features["repaired_data"].sum()),
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the TUPRS market research feature dataset.")
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = run_feature_pipeline(args.processed_dir, args.output_dir)
    print(f"Feature CSV: {paths.csv}")
    print(f"Feature Parquet: {paths.parquet}")
    print(f"Quality summary: {paths.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
