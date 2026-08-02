from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import SymbolConfig


OHLC_COLUMNS = ("Open", "High", "Low", "Close")
MINIMUM_FEATURE_HISTORY = 200


def tradable_session_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = frame.reindex(columns=[*OHLC_COLUMNS, "Volume"]).apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite = pd.Series(
        np.isfinite(numeric.to_numpy()).all(axis=1),
        index=frame.index,
    )
    positive_prices = numeric[list(OHLC_COLUMNS)].gt(0).all(axis=1)
    ordered_prices = (
        numeric["High"].ge(numeric[["Open", "Close"]].max(axis=1))
        & numeric["Low"].le(numeric[["Open", "Close"]].min(axis=1))
        & numeric["High"].ge(numeric["Low"])
    )
    return finite & positive_prices & ordered_prices & numeric["Volume"].gt(0)


def detect_raw_price_basis(frame: pd.DataFrame) -> str:
    if "Stock Splits" not in frame.columns or len(frame) < 2:
        return "no-split-observation"

    ordered = frame.sort_values("Date").reset_index(drop=True)
    splits = pd.to_numeric(ordered["Stock Splits"], errors="coerce").fillna(0.0)
    observations: list[str] = []
    for index in ordered.index[splits.gt(0)]:
        if index == 0:
            continue
        split = float(splits.iloc[index])
        previous_close = float(ordered.loc[index - 1, "Close"])
        current_open = float(ordered.loc[index, "Open"])
        if not np.isfinite([split, previous_close, current_open]).all() or previous_close <= 0:
            continue
        observed_ratio = current_open / previous_close
        adjusted_distance = abs(observed_ratio - 1.0)
        unadjusted_distance = abs(observed_ratio - (1.0 / split))
        observations.append("split-adjusted" if adjusted_distance <= unadjusted_distance else "unadjusted")

    if not observations:
        return "no-split-observation"
    if len(set(observations)) == 1:
        return observations[0]
    return "mixed"


def collection_quality_record(
    config: SymbolConfig,
    frame: pd.DataFrame,
    benchmark_status: Mapping[str, object] | None = None,
    duplicate_date_count: int = 0,
    minimum_history: int = MINIMUM_FEATURE_HISTORY,
) -> dict[str, object]:
    prepared = frame.copy()
    prepared["Date"] = pd.to_datetime(prepared.get("Date"), errors="coerce")
    tradable = tradable_session_mask(prepared) if not prepared.empty else pd.Series(dtype=bool)
    ohlc = prepared.reindex(columns=OHLC_COLUMNS).apply(pd.to_numeric, errors="coerce")
    invalid_ohlc = (
        ohlc.isna().any(axis=1)
        | ohlc.le(0).any(axis=1)
        | ohlc["High"].lt(ohlc[["Open", "Close"]].max(axis=1))
        | ohlc["Low"].gt(ohlc[["Open", "Close"]].min(axis=1))
        | ohlc["High"].lt(ohlc["Low"])
    )
    status = dict(benchmark_status or {})
    tradable_count = int(tradable.sum())
    raw_count = len(prepared)
    excluded_reasons: list[str] = []
    if tradable_count < minimum_history:
        excluded_reasons.append(f"insufficient_history:{tradable_count}<{minimum_history}")
    if raw_count and bool(invalid_ohlc.all()):
        excluded_reasons.append("invalid_ohlc")
    if raw_count == 0:
        excluded_reasons.append("no_data")

    return {
        "symbol": config.symbol,
        "short_name": config.short_name,
        "asset_type": config.asset_type,
        "first_date": prepared["Date"].min(),
        "last_date": prepared["Date"].max(),
        "raw_row_count": raw_count,
        "tradable_session_count": tradable_count,
        "missing_ohlc_count": int(ohlc.isna().any(axis=1).sum()),
        "invalid_ohlc_count": int(invalid_ohlc.sum()),
        "zero_volume_count": int(
            pd.to_numeric(prepared.get("Volume"), errors="coerce").fillna(-1).eq(0).sum()
        ),
        "duplicate_date_count": int(duplicate_date_count),
        "dividend_event_count": int(
            pd.to_numeric(prepared.get("Dividends"), errors="coerce").fillna(0).gt(0).sum()
        ),
        "stock_split_count": int(
            pd.to_numeric(prepared.get("Stock Splits"), errors="coerce").fillna(0).gt(0).sum()
        ),
        "repaired_row_count": int(
            prepared.get("Repaired", pd.Series(False, index=prepared.index)).fillna(False).astype(bool).sum()
        ),
        "detected_raw_price_basis": detect_raw_price_basis(prepared),
        "market_benchmark": config.market_benchmark,
        "preferred_sector_benchmark": config.preferred_sector_benchmark,
        "sector_benchmark_used": status.get("sector_benchmark_used"),
        "sector_benchmark_fallback": bool(status.get("sector_benchmark_fallback", False)),
        "failed_benchmark_downloads": status.get("failed_benchmark_downloads", ""),
        "benchmark_fallback_reason": status.get("benchmark_fallback_reason", ""),
        "data_completeness": (tradable_count / raw_count) if raw_count else 0.0,
        "excluded": bool(excluded_reasons),
        "exclusion_reason": ";".join(excluded_reasons),
    }
