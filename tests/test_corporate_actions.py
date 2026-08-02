from __future__ import annotations

import pandas as pd

from bist_research.backtest.corporate_actions import (
    audit_corporate_actions,
    raw_strategy_prices_are_consistent,
)


def test_large_close_adjusted_ratio_change_is_corporate_action_candidate() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "tuprs_close": [100.0, 102.0, 120.0],
            "tuprs_adj_close": [100.0, 102.0, 100.0],
        }
    )

    audit = audit_corporate_actions(frame, ratio_change_threshold=0.10)

    assert audit["corporate_action_candidate"].tolist() == [False, False, True]
    assert audit.loc[2, "ratio_change"] > 0.10


def test_raw_strategy_price_consistency_uses_raw_ohlc() -> None:
    frame = pd.DataFrame(
        {
            "tuprs_open": [100.0],
            "tuprs_high": [105.0],
            "tuprs_low": [98.0],
            "tuprs_close": [103.0],
            "tuprs_adj_close": [90.0],
        }
    )

    assert raw_strategy_prices_are_consistent(frame)
    assert not raw_strategy_prices_are_consistent(frame.drop(columns="tuprs_open"))
