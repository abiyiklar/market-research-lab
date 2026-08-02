from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_RATIO_CHANGE_THRESHOLD = 0.05


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

    audit = frame.loc[:, ["date", "tuprs_close", "tuprs_adj_close"]].copy()
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
    audit["strategy_price_basis"] = "raw_ohlc"
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
