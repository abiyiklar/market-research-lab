from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from bist_research.backtest.corporate_actions import PRICE_BASIS_UNADJUSTED
from bist_research.config import SymbolConfig

from .engine import causal_split_price_basis
from .models import PortfolioConfig


def _finalize_curve(frame: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    curve = frame.copy()
    curve["daily_return"] = curve["total_equity"].pct_change(fill_method=None).fillna(
        curve["total_equity"].iloc[0] / starting_capital - 1.0
    )
    curve["cumulative_return"] = curve["total_equity"] / starting_capital - 1.0
    peak = curve["total_equity"].cummax()
    curve["drawdown"] = curve["total_equity"] / peak - 1.0
    return curve


def build_xu100_benchmark(
    portfolio_dates: pd.Series,
    panel: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.DataFrame:
    source = panel.loc[:, ["date", "market_close"]].copy()
    source["date"] = pd.to_datetime(source["date"])
    consistency = source.groupby("date")["market_close"].agg(["min", "max"])
    if ((consistency["max"] - consistency["min"]).abs() > 1e-8).any():
        raise ValueError("XU100 values are inconsistent across symbol rows")
    market = source.drop_duplicates("date", keep="last").sort_values("date")
    dates = pd.DataFrame({"date": pd.to_datetime(portfolio_dates)})
    aligned = pd.merge_asof(dates, market, on="date", direction="backward")
    if aligned["market_close"].isna().any():
        raise ValueError("XU100 benchmark cannot be aligned without future data")
    first_price = float(aligned["market_close"].iloc[0])
    gross_budget = config.starting_capital / (1.0 + config.transaction_cost_rate)
    transaction_cost = gross_budget * config.transaction_cost_rate
    units = gross_budget / first_price
    aligned["cash"] = 0.0
    aligned["total_equity"] = units * aligned["market_close"]
    aligned["gross_exposure"] = 1.0
    aligned["number_open_positions"] = 1
    aligned["cumulative_transaction_costs"] = transaction_cost
    aligned["cumulative_dividends"] = 0.0
    aligned["price_index_only"] = True
    return _finalize_curve(aligned, config.starting_capital)


def build_equal_weight_benchmark(
    portfolio_dates: pd.Series,
    feature_frames: Mapping[str, pd.DataFrame],
    configs: Sequence[SymbolConfig],
    config: PortfolioConfig,
) -> pd.DataFrame:
    dates = [pd.Timestamp(date) for date in pd.to_datetime(portfolio_dates)]
    if not dates:
        raise ValueError("Equal-weight benchmark requires portfolio dates")
    start = dates[0]
    available: dict[str, pd.Series] = {}
    lookups: dict[str, pd.DataFrame] = {}
    for symbol_config in configs:
        frame = feature_frames[symbol_config.symbol].set_index("date", drop=False)
        lookups[symbol_config.symbol] = frame
        if start in frame.index:
            row = frame.loc[start]
            if bool(row["is_valid_stock_session"]):
                available[symbol_config.symbol] = row
    if not available:
        raise ValueError("No stocks are available for equal-weight benchmark inception")

    cash = config.starting_capital
    target = config.starting_capital / len(available)
    holdings: dict[str, dict[str, object]] = {}
    cumulative_costs = 0.0
    cumulative_dividends = 0.0
    for symbol in sorted(available):
        row = available[symbol]
        price = float(row["execution_close"])
        quantity = math.floor(target / (price * (1.0 + config.transaction_cost_rate)))
        gross = quantity * price
        cost = gross * config.transaction_cost_rate
        cash -= gross + cost
        cumulative_costs += cost
        holdings[symbol] = {
            "quantity": quantity,
            "last_close": price,
            "price_basis": str(row.get("price_basis", "raw_ohlc_split_basis_unknown")),
            "dividend_dates": set(),
            "split_dates": set(),
        }

    rows: list[dict[str, object]] = []
    for date_index, date in enumerate(dates):
        if date_index > 0:
            for symbol, holding in holdings.items():
                frame = lookups[symbol]
                if date not in frame.index:
                    continue
                row = frame.loc[date]
                if not bool(row["is_valid_stock_session"]):
                    continue
                split = float(row.get("stock_split", 0.0) or 0.0)
                if split > 0 and date not in holding["split_dates"]:
                    holding["price_basis"] = causal_split_price_basis(
                        float(holding["last_close"]),
                        float(row["execution_open"]),
                        split,
                    )
                if split > 0 and date not in holding["split_dates"] and holding[
                    "price_basis"
                ] == PRICE_BASIS_UNADJUSTED:
                    exact = int(holding["quantity"]) * split
                    whole = math.floor(exact + 1e-12)
                    fractional = exact - whole
                    cash += fractional * float(row["execution_open"])
                    holding["quantity"] = whole
                if split > 0:
                    holding["split_dates"].add(date)
                dividend = float(row.get("dividend", 0.0) or 0.0)
                if dividend > 0 and date not in holding["dividend_dates"]:
                    dividend_cash = int(holding["quantity"]) * dividend
                    cash += dividend_cash
                    cumulative_dividends += dividend_cash
                    holding["dividend_dates"].add(date)

        market_value = 0.0
        stale = 0
        for symbol, holding in holdings.items():
            frame = lookups[symbol]
            if date in frame.index and bool(frame.loc[date, "is_valid_stock_session"]):
                holding["last_close"] = float(frame.loc[date, "execution_close"])
            else:
                stale += 1
            market_value += int(holding["quantity"]) * float(holding["last_close"])
        equity = cash + market_value
        rows.append(
            {
                "date": date,
                "cash": cash,
                "total_equity": equity,
                "gross_exposure": market_value / equity if equity > 0 else 0.0,
                "number_open_positions": len(holdings),
                "cumulative_transaction_costs": cumulative_costs,
                "cumulative_dividends": cumulative_dividends,
                "stale_valuation_count": stale,
            }
        )
    return _finalize_curve(pd.DataFrame(rows), config.starting_capital)
