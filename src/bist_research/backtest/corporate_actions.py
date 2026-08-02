from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_RATIO_CHANGE_THRESHOLD = 0.05
PRICE_BASIS_SPLIT_ADJUSTED = "raw_ohlc_split_adjusted"
PRICE_BASIS_UNADJUSTED = "raw_ohlc_unadjusted"
PRICE_BASIS_UNKNOWN = "raw_ohlc_split_basis_unknown"


def detect_price_basis(frame: pd.DataFrame) -> str:
    """Infer whether raw OHLC is already normalized across reported splits."""
    close_column = "tuprs_close" if "tuprs_close" in frame else "Close"
    open_column = "tuprs_open" if "tuprs_open" in frame else "Open"
    split_column = "stock_split_factor" if "stock_split_factor" in frame else "Stock Splits"
    date_column = "date" if "date" in frame else "Date"
    required = {close_column, open_column, split_column, date_column}
    if not required.issubset(frame.columns):
        return PRICE_BASIS_UNKNOWN

    ordered = frame.loc[:, [date_column, open_column, close_column, split_column]].copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], errors="coerce")
    ordered = ordered.sort_values(date_column).reset_index(drop=True)
    splits = pd.to_numeric(ordered[split_column], errors="coerce").fillna(0.0)
    event_indices = ordered.index[splits.gt(0)]
    decisions: list[bool] = []
    for index in event_indices:
        if index == 0:
            continue
        previous_close = float(ordered.loc[index - 1, close_column])
        event_open = float(ordered.loc[index, open_column])
        factor = float(splits.loc[index])
        if min(previous_close, event_open, factor) <= 0:
            continue
        observed_ratio = previous_close / event_open
        continuous_distance = abs(np.log(observed_ratio))
        unadjusted_distance = abs(np.log(observed_ratio / factor))
        decisions.append(continuous_distance <= unadjusted_distance)

    if decisions and all(decisions):
        return PRICE_BASIS_SPLIT_ADJUSTED
    if decisions and not any(decisions):
        return PRICE_BASIS_UNADJUSTED
    return PRICE_BASIS_UNKNOWN


def audit_corporate_actions(
    frame: pd.DataFrame,
    ratio_change_threshold: float = DEFAULT_RATIO_CHANGE_THRESHOLD,
) -> pd.DataFrame:
    """Flag discontinuities between raw Close and dividend/split-adjusted Close."""
    required = {"date", "tuprs_close", "tuprs_adj_close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Corporate-action audit is missing columns: {', '.join(missing)}")
    if ratio_change_threshold <= 0:
        raise ValueError("ratio_change_threshold must be positive")

    optional = [
        column
        for column in (
            "tuprs_open",
            "dividend_per_share",
            "stock_split_factor",
            "repaired_data",
            "corporate_action_flag",
            "price_basis",
        )
        if column in frame
    ]
    audit = frame.loc[:, ["date", "tuprs_close", "tuprs_adj_close", *optional]].copy()
    audit["date"] = pd.to_datetime(audit["date"]).dt.normalize()
    close = pd.to_numeric(audit["tuprs_close"], errors="coerce")
    adjusted = pd.to_numeric(audit["tuprs_adj_close"], errors="coerce")
    audit["close_adjusted_ratio"] = close / adjusted.replace(0, np.nan)
    audit["ratio_change"] = audit["close_adjusted_ratio"].pct_change(fill_method=None)
    audit["corporate_action_candidate"] = (
        audit["ratio_change"].abs().ge(ratio_change_threshold).fillna(False)
    )
    audit["raw_close_return"] = close.pct_change(fill_method=None)
    audit["adjusted_close_return"] = adjusted.pct_change(fill_method=None)
    audit["raw_price_continuity_warning"] = (
        audit["corporate_action_candidate"]
        & audit["raw_close_return"].abs().ge(ratio_change_threshold)
        & audit["adjusted_close_return"].abs().lt(ratio_change_threshold / 2)
    )
    if "dividend_per_share" not in audit:
        audit["dividend_per_share"] = 0.0
    if "stock_split_factor" not in audit:
        audit["stock_split_factor"] = 0.0
    if "repaired_data" not in audit:
        audit["repaired_data"] = False
    if "corporate_action_flag" not in audit:
        audit["corporate_action_flag"] = (
            audit["dividend_per_share"].ne(0) | audit["stock_split_factor"].ne(0)
        )
    detected_basis = detect_price_basis(frame)
    if "price_basis" not in audit:
        audit["price_basis"] = detected_basis
    audit["price_basis"] = audit["price_basis"].replace(
        {"raw_ohlc_split_basis_unknown": detected_basis}
    )
    audit["strategy_price_basis"] = audit["price_basis"]
    return audit


def raw_strategy_prices_are_consistent(frame: pd.DataFrame) -> bool:
    """Validate that strategy execution uses a complete, internally valid raw OHLC set."""
    columns = ["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]
    if any(column not in frame for column in columns):
        return False
    prices = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return bool(
        np.isfinite(prices.to_numpy(dtype="float64")).all()
        and (prices > 0).all().all()
        and (prices["tuprs_high"] >= prices[["tuprs_open", "tuprs_close", "tuprs_low"]].max(axis=1)).all()
        and (prices["tuprs_low"] <= prices[["tuprs_open", "tuprs_close", "tuprs_high"]].min(axis=1)).all()
    )
