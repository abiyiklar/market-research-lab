from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .models import PortfolioBacktestResult, PortfolioConfig


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-15:
        return 0.0
    return float(numerator / denominator)


def _equity_column(frame: pd.DataFrame) -> str:
    if "total_portfolio_equity" in frame:
        return "total_portfolio_equity"
    if "total_equity" in frame:
        return "total_equity"
    raise ValueError("Daily frame has no recognized equity column")


def _maximum_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def calculate_performance_metrics(
    name: str,
    daily: pd.DataFrame,
    config: PortfolioConfig,
    trades: pd.DataFrame | None = None,
    open_positions: pd.DataFrame | None = None,
    gross_traded_value: float | None = None,
    total_transaction_costs: float | None = None,
    total_dividends: float | None = None,
) -> dict[str, object]:
    if daily.empty:
        raise ValueError(f"Cannot calculate performance metrics for empty {name} data")
    equity_column = _equity_column(daily)
    equity = pd.to_numeric(daily[equity_column], errors="raise")
    dates = pd.to_datetime(daily["date"], errors="raise")
    returns = equity.pct_change(fill_method=None)
    if "daily_return" in daily:
        returns = pd.to_numeric(daily["daily_return"], errors="coerce")
    returns = returns.fillna(0.0)
    ending = float(equity.iloc[-1])
    total_return = ending / config.starting_capital - 1.0
    elapsed_years = max((dates.iloc[-1] - dates.iloc[0]).days / 365.25, 1 / 365.25)
    cagr = (ending / config.starting_capital) ** (1.0 / elapsed_years) - 1.0
    volatility = float(returns.iloc[1:].std(ddof=1)) * math.sqrt(config.trading_days_per_year)
    return_std = float(returns.iloc[1:].std(ddof=1))
    sharpe = _safe_ratio(
        float(returns.iloc[1:].mean()) * math.sqrt(config.trading_days_per_year),
        return_std,
    )
    downside = returns.iloc[1:].clip(upper=0.0)
    downside_deviation = math.sqrt(float((downside**2).mean())) if len(downside) else 0.0
    sortino = _safe_ratio(
        float(returns.iloc[1:].mean()) * math.sqrt(config.trading_days_per_year),
        downside_deviation,
    )
    maximum_drawdown = _maximum_drawdown(equity)
    calmar = _safe_ratio(cagr, abs(maximum_drawdown))

    active_trades = trades if trades is not None else pd.DataFrame()
    active_open = open_positions if open_positions is not None else pd.DataFrame()
    trade_returns = (
        pd.to_numeric(active_trades["net_return"], errors="coerce")
        if not active_trades.empty
        else pd.Series(dtype="float64")
    )
    net_pnl = (
        pd.to_numeric(active_trades["net_pnl"], errors="coerce")
        if not active_trades.empty
        else pd.Series(dtype="float64")
    )
    wins = trade_returns.gt(0)
    losses = trade_returns.lt(0)
    winning = trade_returns.loc[wins]
    losing = trade_returns.loc[losses]
    gross_profit = float(net_pnl.loc[net_pnl.gt(0)].sum())
    gross_loss = abs(float(net_pnl.loc[net_pnl.lt(0)].sum()))

    if gross_traded_value is None:
        gross_traded_value = (
            float(
                (
                    active_trades["quantity"] * active_trades["entry_price"]
                    + active_trades["exit_quantity"] * active_trades["exit_price"]
                ).sum()
            )
            if not active_trades.empty
            else 0.0
        )
    if total_transaction_costs is None:
        total_transaction_costs = (
            float(daily["cumulative_transaction_costs"].iloc[-1])
            if "cumulative_transaction_costs" in daily
            else 0.0
        )
    if total_dividends is None:
        total_dividends = (
            float(daily["cumulative_dividends"].iloc[-1])
            if "cumulative_dividends" in daily
            else 0.0
        )
    average_equity = float(equity.mean())
    annual_turnover = _safe_ratio(float(gross_traded_value), average_equity * elapsed_years)

    cash = (
        pd.to_numeric(daily["cash"], errors="coerce")
        if "cash" in daily
        else pd.Series(0.0, index=daily.index)
    )
    exposure = (
        pd.to_numeric(daily["gross_exposure"], errors="coerce")
        if "gross_exposure" in daily
        else 1.0 - cash / equity
    )
    open_count = (
        pd.to_numeric(daily["number_open_positions"], errors="coerce")
        if "number_open_positions" in daily
        else pd.Series(0.0, index=daily.index)
    )

    return {
        "portfolio": name,
        "start_date": dates.iloc[0].date().isoformat(),
        "end_date": dates.iloc[-1].date().isoformat(),
        "starting_capital": config.starting_capital,
        "ending_value": ending,
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": calmar,
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "average_gross_exposure": float(exposure.mean()),
        "average_open_positions": float(open_count.mean()),
        "percentage_days_fully_invested": float(open_count.ge(config.max_positions).mean()),
        "percentage_days_holding_cash": float(cash.gt(1e-8).mean()),
        "average_cash_weight": float((cash / equity).mean()),
        "annual_turnover": annual_turnover,
        "total_transaction_costs": float(total_transaction_costs),
        "total_dividends_received": float(total_dividends),
        "completed_trades": int(len(active_trades)),
        "open_positions_at_end": int(len(active_open)),
        "winning_trades": int(wins.sum()),
        "losing_trades": int(losses.sum()),
        "win_rate": float(wins.mean()) if len(wins) else 0.0,
        "average_trade_return": float(trade_returns.mean()) if len(trade_returns) else 0.0,
        "median_trade_return": float(trade_returns.median()) if len(trade_returns) else 0.0,
        "average_winning_trade": float(winning.mean()) if len(winning) else 0.0,
        "average_losing_trade": float(losing.mean()) if len(losing) else 0.0,
        "payoff_ratio": _safe_ratio(
            float(winning.mean()) if len(winning) else 0.0,
            abs(float(losing.mean())) if len(losing) else 0.0,
        ),
        "profit_factor": _safe_ratio(gross_profit, gross_loss),
        "average_holding_sessions": (
            float(active_trades["holding_sessions"].mean())
            if not active_trades.empty
            else 0.0
        ),
        "median_holding_sessions": (
            float(active_trades["holding_sessions"].median())
            if not active_trades.empty
            else 0.0
        ),
        "maximum_holding_sessions": (
            int(active_trades["holding_sessions"].max())
            if not active_trades.empty
            else 0
        ),
        "average_holding_calendar_days": (
            float(active_trades["holding_calendar_days"].mean())
            if not active_trades.empty
            else 0.0
        ),
        "maximum_favorable_excursion": (
            float(active_trades["mfe"].max()) if not active_trades.empty else 0.0
        ),
        "maximum_adverse_excursion": (
            float(active_trades["mae"].min()) if not active_trades.empty else 0.0
        ),
    }


def build_period_tables(
    portfolio_daily: pd.DataFrame,
    benchmark_curves: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    combined = portfolio_daily.loc[:, ["date", "total_portfolio_equity"]].copy()
    combined["date"] = pd.to_datetime(combined["date"])
    for name, curve in benchmark_curves.items():
        benchmark = curve.loc[:, ["date", "total_equity"]].rename(
            columns={"total_equity": name}
        )
        combined = combined.merge(benchmark, on="date", how="left", validate="one_to_one")
    combined = combined.sort_values("date").reset_index(drop=True)
    value_columns = ["total_portfolio_equity", *benchmark_curves.keys()]
    returns = combined[value_columns].pct_change(fill_method=None).fillna(0.0)
    returns["date"] = combined["date"]

    yearly_rows: list[dict[str, object]] = []
    for year, group in returns.groupby(returns["date"].dt.year, sort=True):
        row: dict[str, object] = {"year": int(year)}
        for column in value_columns:
            row[f"{column}_return"] = float((1.0 + group[column]).prod() - 1.0)
        row["portfolio_xu100_excess"] = (
            row["total_portfolio_equity_return"] - row.get("xu100_return", np.nan)
        )
        yearly_rows.append(row)
    yearly = pd.DataFrame(yearly_rows)

    monthly_rows: list[dict[str, object]] = []
    periods = returns["date"].dt.to_period("M")
    for period, group in returns.groupby(periods, sort=True):
        row = {"month": str(period)}
        for column in value_columns:
            row[f"{column}_return"] = float((1.0 + group[column]).prod() - 1.0)
        row["portfolio_xu100_excess"] = (
            row["total_portfolio_equity_return"] - row.get("xu100_return", np.nan)
        )
        monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)

    rolling = pd.DataFrame({"date": combined["date"]})
    portfolio = combined["total_portfolio_equity"].astype(float)
    xu100 = combined["xu100"].astype(float)
    rolling["rolling_3m_return"] = portfolio / portfolio.shift(63) - 1.0
    rolling["rolling_6m_return"] = portfolio / portfolio.shift(126) - 1.0
    rolling["rolling_12m_return"] = portfolio / portfolio.shift(252) - 1.0
    rolling["rolling_12m_xu100_return"] = xu100 / xu100.shift(252) - 1.0
    rolling["rolling_12m_xu100_excess"] = (
        rolling["rolling_12m_return"] - rolling["rolling_12m_xu100_return"]
    )
    rolling["rolling_24m_cagr"] = (portfolio / portfolio.shift(504)) ** 0.5 - 1.0
    portfolio_returns = portfolio.pct_change(fill_method=None)
    rolling_mean = portfolio_returns.rolling(504, min_periods=504).mean()
    rolling_std = portfolio_returns.rolling(504, min_periods=504).std()
    rolling["rolling_24m_sharpe"] = rolling_mean / rolling_std * math.sqrt(252)
    rolling["rolling_24m_max_drawdown"] = portfolio.rolling(
        504, min_periods=504
    ).apply(lambda values: np.min(values / np.maximum.accumulate(values) - 1.0), raw=True)

    rolling_12 = rolling["rolling_12m_return"].dropna()
    rolling_excess = rolling["rolling_12m_xu100_excess"].dropna()
    yearly_portfolio = yearly.get(
        "total_portfolio_equity_return", pd.Series(dtype="float64")
    )
    yearly_excess = yearly.get("portfolio_xu100_excess", pd.Series(dtype="float64"))
    summary = {
        "positive_rolling_12m_pct": float(rolling_12.gt(0).mean()) if len(rolling_12) else 0.0,
        "outperform_xu100_rolling_12m_pct": (
            float(rolling_excess.gt(0).mean()) if len(rolling_excess) else 0.0
        ),
        "outperform_xu100_calendar_year_pct": (
            float(yearly_excess.gt(0).mean()) if len(yearly_excess) else 0.0
        ),
        "positive_calendar_years": int(yearly_portfolio.gt(0).sum()),
        "negative_calendar_years": int(yearly_portfolio.lt(0).sum()),
        "worst_rolling_12m_return": float(rolling_12.min()) if len(rolling_12) else 0.0,
        "worst_rolling_24m_drawdown": float(
            rolling["rolling_24m_max_drawdown"].min()
        ),
    }
    return yearly, monthly, rolling, summary


def build_calendar_year_contribution(
    result: PortfolioBacktestResult,
) -> pd.DataFrame:
    daily = result.daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    rows: list[dict[str, object]] = []
    prior_equity = float(result.config.starting_capital)
    prior_unrealized = 0.0
    for year, group in daily.groupby("year", sort=True):
        ending_equity = float(group["total_portfolio_equity"].iloc[-1])
        ending_unrealized = float(group["unrealized_pnl"].iloc[-1])
        realized = float(group["realized_pnl_day"].sum())
        dividends = float(group["dividends_received_day"].sum())
        costs = float(group["transaction_costs_paid_day"].sum())
        unrealized_change = ending_unrealized - prior_unrealized
        net_contribution = realized + dividends - costs + unrealized_change
        rows.append(
            {
                "year": int(year),
                "starting_equity": prior_equity,
                "ending_equity": ending_equity,
                "realized_pnl": realized,
                "dividend_income": dividends,
                "transaction_costs": costs,
                "change_in_unrealized_pnl": unrealized_change,
                "net_contribution": net_contribution,
                "equity_change": ending_equity - prior_equity,
                "reconciliation_error": ending_equity
                - prior_equity
                - net_contribution,
            }
        )
        prior_equity = ending_equity
        prior_unrealized = ending_unrealized
    return pd.DataFrame(rows)


def build_contribution_tables(
    result: PortfolioBacktestResult,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    trades = result.trades.copy()
    open_positions = result.open_positions.copy()
    audit = result.position_daily_audit.copy()
    all_symbols = sorted(
        set(trades.get("symbol", pd.Series(dtype="string")).astype(str))
        | set(open_positions.get("symbol", pd.Series(dtype="string")).astype(str))
    )
    symbol_rows: list[dict[str, object]] = []
    total_profit = (
        float(result.daily["total_portfolio_equity"].iloc[-1])
        - result.config.starting_capital
    )
    for symbol in all_symbols:
        completed = trades.loc[trades["symbol"].eq(symbol)] if not trades.empty else trades
        opened = (
            open_positions.loc[open_positions["symbol"].eq(symbol)]
            if not open_positions.empty
            else open_positions
        )
        completed_net = float(completed["net_pnl"].sum()) if not completed.empty else 0.0
        open_net = (
            float(
                (
                    opened["market_value"]
                    + opened["split_cash_in_lieu"]
                    - opened["entry_price"] * opened["entry_quantity"]
                    + opened["dividends_received"]
                    - opened["entry_costs"]
                ).sum()
            )
            if not opened.empty
            else 0.0
        )
        symbol_audit = audit.loc[audit["symbol"].eq(symbol)] if not audit.empty else audit
        max_allocation = 0.0
        if not symbol_audit.empty:
            equity = result.daily.set_index("date")["total_portfolio_equity"]
            allocations = symbol_audit.apply(
                lambda row: row["market_value"] / equity.loc[pd.Timestamp(row["date"])],
                axis=1,
            )
            max_allocation = float(allocations.max())
        entries = len(completed) + len(opened)
        symbol_rows.append(
            {
                "symbol": symbol,
                "entries": entries,
                "completed_trades": len(completed),
                "realized_pnl": (
                    float(completed["gross_pnl"].sum()) if not completed.empty else 0.0
                ),
                "dividend_income": (
                    float(completed["dividends_received"].sum()) if not completed.empty else 0.0
                )
                + (
                    float(opened["dividends_received"].sum()) if not opened.empty else 0.0
                ),
                "transaction_costs": (
                    float((completed["entry_costs"] + completed["exit_costs"]).sum())
                    if not completed.empty
                    else 0.0
                )
                + (float(opened["entry_costs"].sum()) if not opened.empty else 0.0),
                "net_contribution": completed_net + open_net,
                "win_rate": (
                    float(completed["net_pnl"].gt(0).mean()) if not completed.empty else 0.0
                ),
                "average_holding_sessions": (
                    float(completed["holding_sessions"].mean())
                    if not completed.empty
                    else 0.0
                ),
                "maximum_capital_allocation": max_allocation,
                "percentage_total_portfolio_profit": _safe_ratio(
                    completed_net + open_net, total_profit
                ),
            }
        )
    symbol_contribution = pd.DataFrame(symbol_rows)

    def grouped_contribution(column: str, output_column: str) -> pd.DataFrame:
        completed = (
            trades.groupby(column, dropna=False, observed=True)
            .agg(
                completed_trades=("position_id", "count"),
                gross_pnl=("gross_pnl", "sum"),
                dividends_received=("dividends_received", "sum"),
                entry_costs=("entry_costs", "sum"),
                exit_costs=("exit_costs", "sum"),
                net_contribution=("net_pnl", "sum"),
            )
            .reset_index()
            if not trades.empty
            else pd.DataFrame(
                columns=[
                    column,
                    "completed_trades",
                    "gross_pnl",
                    "dividends_received",
                    "entry_costs",
                    "exit_costs",
                    "net_contribution",
                ]
            )
        )
        return completed.rename(columns={column: output_column})

    signal_type = grouped_contribution("entry_signal_type", "signal_type")
    exit_reason = grouped_contribution("exit_reason", "exit_reason")
    score_bucket = grouped_contribution("entry_score_state", "score_bucket")
    sector_fallback = grouped_contribution(
        "sector_fallback_at_entry", "sector_fallback_at_entry"
    )

    completed_trade_profit = float(trades["net_pnl"].sum()) if not trades.empty else 0.0
    top_trade_profit = (
        trades["net_pnl"].nlargest(3)
        if not trades.empty
        else pd.Series(dtype="float64")
    )
    best_trade = float(top_trade_profit.iloc[0]) if len(top_trade_profit) else 0.0
    best_three = float(top_trade_profit.sum())
    best_symbol = (
        float(symbol_contribution["net_contribution"].max())
        if not symbol_contribution.empty
        else 0.0
    )
    concentration = {
        "best_completed_trade_concentration": _safe_ratio(
            best_trade, completed_trade_profit
        ),
        "best_three_trade_concentration": _safe_ratio(
            best_three, completed_trade_profit
        ),
        "best_symbol_concentration": _safe_ratio(best_symbol, total_profit),
        "ending_value_without_best_three_trades": float(
            result.daily["total_portfolio_equity"].iloc[-1]
        )
        - best_three,
        "ending_value_without_best_symbol": float(
            result.daily["total_portfolio_equity"].iloc[-1]
        )
        - best_symbol,
    }
    return (
        symbol_contribution,
        signal_type,
        exit_reason,
        score_bucket,
        sector_fallback,
        concentration,
    )


def build_research_gates(
    primary_metrics: Mapping[str, object],
    xu100_metrics: Mapping[str, object],
    rolling_summary: Mapping[str, float],
    concentration: Mapping[str, float],
    stress_summary: pd.DataFrame,
    yearly_returns: pd.DataFrame,
) -> pd.DataFrame:
    cost_stress = stress_summary.loc[stress_summary["scenario"].eq("cost_0.0030")]
    profitable_cost_stress = (
        float(cost_stress["total_return"].iloc[0]) > 0 if not cost_stress.empty else False
    )
    yearly_column = "total_portfolio_equity_return"
    positive_years = yearly_returns.loc[yearly_returns[yearly_column].gt(0), yearly_column]
    largest_year_share = _safe_ratio(
        float(positive_years.max()) if len(positive_years) else 0.0,
        float(positive_years.sum()) if len(positive_years) else 0.0,
    )
    not_one_year = len(positive_years) >= 2 and largest_year_share < 0.60
    rows = [
        ("primary_cagr_exceeds_xu100", primary_metrics["cagr"], xu100_metrics["cagr"], float(primary_metrics["cagr"]) > float(xu100_metrics["cagr"])),
        ("primary_sharpe_at_least_0.70", primary_metrics["sharpe_ratio"], 0.70, float(primary_metrics["sharpe_ratio"]) >= 0.70),
        ("primary_drawdown_better_than_minus_30pct", primary_metrics["maximum_drawdown"], -0.30, float(primary_metrics["maximum_drawdown"]) > -0.30),
        ("profitable_at_0.30pct_cost", float(cost_stress["total_return"].iloc[0]) if not cost_stress.empty else np.nan, 0.0, profitable_cost_stress),
        ("positive_rolling_12m_at_least_60pct", rolling_summary["positive_rolling_12m_pct"], 0.60, rolling_summary["positive_rolling_12m_pct"] >= 0.60),
        ("rolling_12m_xu100_outperformance_at_least_55pct", rolling_summary["outperform_xu100_rolling_12m_pct"], 0.55, rolling_summary["outperform_xu100_rolling_12m_pct"] >= 0.55),
        ("best_three_trades_below_35pct", concentration["best_three_trade_concentration"], 0.35, concentration["best_three_trade_concentration"] < 0.35),
        ("best_symbol_below_35pct", concentration["best_symbol_concentration"], 0.35, concentration["best_symbol_concentration"] < 0.35),
        ("not_dependent_on_one_calendar_year", largest_year_share, 0.60, not_one_year),
    ]
    return pd.DataFrame(
        [
            {"gate": gate, "actual_value": actual, "threshold": threshold, "result": "PASS" if passed else "FAIL"}
            for gate, actual, threshold, passed in rows
        ]
    )
