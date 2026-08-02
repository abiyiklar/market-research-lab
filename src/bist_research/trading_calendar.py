from __future__ import annotations

import numpy as np
import pandas as pd


TUPRS_OHLC_COLUMNS = (
    "tuprs_open",
    "tuprs_high",
    "tuprs_low",
    "tuprs_close",
)


def tradable_tuprs_mask(frame: pd.DataFrame) -> pd.Series:
    """Return the canonical mask for executable TUPRS sessions."""
    required = {*TUPRS_OHLC_COLUMNS, "tuprs_volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"TUPRS calendar is missing columns: {', '.join(missing)}")

    values = frame.loc[:, [*TUPRS_OHLC_COLUMNS, "tuprs_volume"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    finite = pd.Series(
        np.isfinite(values.to_numpy(dtype="float64")).all(axis=1),
        index=frame.index,
    )
    positive_prices = values.loc[:, TUPRS_OHLC_COLUMNS].gt(0).all(axis=1)
    high_valid = values["tuprs_high"].ge(
        values.loc[:, ["tuprs_open", "tuprs_close", "tuprs_low"]].max(axis=1)
    )
    low_valid = values["tuprs_low"].le(
        values.loc[:, ["tuprs_open", "tuprs_close", "tuprs_high"]].min(axis=1)
    )
    return finite & positive_prices & values["tuprs_volume"].gt(0) & high_valid & low_valid


def only_tradable_tuprs_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy and chronologically filter a frame to executable TUPRS sessions."""
    filtered = frame.loc[tradable_tuprs_mask(frame)].copy()
    if "date" in filtered:
        filtered = filtered.sort_values("date")
    return filtered.reset_index(drop=True)
