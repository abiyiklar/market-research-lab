from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _format(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return str(value).replace("|", "\\|")


def _table(frame: pd.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if frame.empty:
        return "_No observations._"
    selected = frame.reindex(columns=columns)
    if limit is not None:
        selected = selected.head(limit)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_format(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_portfolio_report(
    performance: pd.DataFrame,
    yearly: pd.DataFrame,
    rolling_summary: dict[str, float],
    symbol_contribution: pd.DataFrame,
    stress: pd.DataFrame,
    gates: pd.DataFrame,
    concentration: dict[str, float],
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    open_positions: pd.DataFrame,
    primary_max_positions: int,
    primary_cost_rate: float,
) -> str:
    primary = performance.loc[performance["portfolio"].eq("primary_portfolio")]
    benchmarks = performance.loc[performance["portfolio"].ne("primary_portfolio")]
    rejected = orders.loc[orders["order_status"].ne("filled")]
    starting_capital = float(primary["starting_capital"].iloc[0])
    return f"""# Realistic Portfolio Backtest Research

This report evaluates the existing frozen score and BUY/SELL rules in a causal, long-only portfolio. It is a historical research diagnostic, not investment advice, not a guaranteed-profit claim, and not deployment-ready.

## Frozen Primary Policy

- Starting capital: TRY {starting_capital:,.2f}
- Maximum positions: {primary_max_positions}
- Target weight per new position: 25%
- All-in transaction cost per side: {primary_cost_rate:.4%}
- Integer quantities, no leverage, no short selling, no automatic rebalancing
- Signals form at date T close and can execute only at a later valid raw stock-session open
- Confirmed SELL orders execute before BUY orders on the same open

## Primary Performance

{_table(primary, ["start_date", "end_date", "ending_value", "total_return", "cagr", "annualized_volatility", "sharpe_ratio", "sortino_ratio", "maximum_drawdown", "calmar_ratio", "annual_turnover", "total_transaction_costs", "total_dividends_received"])}

## Trade Statistics

{_table(primary, ["completed_trades", "open_positions_at_end", "winning_trades", "losing_trades", "win_rate", "average_trade_return", "median_trade_return", "payoff_ratio", "profit_factor", "average_holding_sessions", "median_holding_sessions", "maximum_holding_sessions", "maximum_favorable_excursion", "maximum_adverse_excursion"])}

Orders: **{len(orders)}** total, **{len(rejected)}** rejected/open-at-end. Completed trades: **{len(trades)}**. Open positions at end: **{len(open_positions)}**.

## Benchmark Comparison

{_table(benchmarks, ["portfolio", "ending_value", "total_return", "cagr", "annualized_volatility", "sharpe_ratio", "sortino_ratio", "maximum_drawdown", "total_transaction_costs", "total_dividends_received"])}

The XU100 input is treated as a price index, not a dividend-inclusive total-return index. The equal-weight benchmark uses raw stock closes, explicit dividends and split-basis-aware quantity accounting. This mismatch is reported rather than silently treating the benchmarks as economically identical.

## Rolling and Calendar Diagnostics

- Positive rolling 12-month windows: **{rolling_summary['positive_rolling_12m_pct']:.2%}**
- Rolling 12-month windows outperforming XU100: **{rolling_summary['outperform_xu100_rolling_12m_pct']:.2%}**
- Calendar years outperforming XU100: **{rolling_summary['outperform_xu100_calendar_year_pct']:.2%}**
- Positive / negative calendar years: **{rolling_summary['positive_calendar_years']} / {rolling_summary['negative_calendar_years']}**
- Worst rolling 12-month return: **{rolling_summary['worst_rolling_12m_return']:.2%}**
- Worst rolling 24-month drawdown: **{rolling_summary['worst_rolling_24m_drawdown']:.2%}**

{_table(yearly, ["year", "total_portfolio_equity_return", "xu100_return", "equal_weight_12_stock_return", "portfolio_xu100_excess"])}

## Contribution and Concentration

{_table(symbol_contribution.sort_values("net_contribution", ascending=False), ["symbol", "entries", "completed_trades", "realized_pnl", "dividend_income", "transaction_costs", "net_contribution", "win_rate", "average_holding_sessions", "maximum_capital_allocation", "percentage_total_portfolio_profit"])}

- Best completed trade share of total net completed-trade profit: **{concentration['best_completed_trade_concentration']:.2%}**
- Best three completed trades share of total net completed-trade profit: **{concentration['best_three_trade_concentration']:.2%}**
- Best symbol share of total portfolio profit: **{concentration['best_symbol_concentration']:.2%}**
- Approximate ending value without best three completed trades: **{concentration['ending_value_without_best_three_trades']:,.2f} TRY**
- Approximate ending value without best-performing symbol: **{concentration['ending_value_without_best_symbol']:,.2f} TRY**

Removal diagnostics subtract audited historical net contribution from ending equity. They are approximate attribution replays, not counterfactual order-book reruns; freed capital is left unallocated.

## Fixed Robustness Cases

{_table(stress, ["scenario", "max_positions", "transaction_cost_rate", "execution_delay_sessions", "ending_value", "total_return", "cagr", "sharpe_ratio", "maximum_drawdown", "completed_trades", "total_transaction_costs"])}

The signals and score engine are held fixed in every case. These rows are shown side by side and are not used to select or optimize a preferred configuration.

## Research Continuation Gates

{_table(gates, ["gate", "actual_value", "threshold", "result"])}

These PASS/FAIL labels are research-continuation criteria only. They are not investment guarantees.

## Accounting and Execution Notes

- Cash accounting uses unchanged raw execution opens and raw closes; total-return signal prices are never used as cash trade prices.
- Dividends are credited once only to positions held before that day's trading. They are not also credited through signal-price returns.
- Reported splits adjust quantity only when the previous observed raw close, event-date raw open, and current split factor causally identify an unadjusted discontinuity. Split-adjusted Yahoo raw history is not adjusted twice.
- For a genuinely unadjusted split, fractional resulting shares are rounded down and valued at the event-date raw open as deterministic cash in lieu; allocated cost basis is realized proportionally.
- Missing stock sessions cannot execute orders. A pending order waits for a real valid open and records both session and calendar delay.
- A missing held-stock session uses the last genuinely observed prior raw close only for stale-marked valuation, never as an execution price.
- The daily portfolio output includes an explicit P&L reconciliation error and the engine fails if it exceeds floating-point tolerance.

## Limitations

- This is an inspected historical sample, not unseen out-of-sample evidence.
- The fixed current 12-stock universe does not reconstruct historical index membership.
- XU100 is a price-index comparison and omits index dividends; the stock portfolio and equal-weight stock benchmark include reported cash dividends.
- Transaction cost rates are all-in research assumptions, not exact broker tariffs.
- No market impact, order-book depth, auction mechanics, taxes by investor type, borrow, interest on cash, or capacity constraints are modeled beyond the fixed all-in cost.
- Corporate-action handling is limited by source accuracy and event-date raw-price-basis inference.
- No machine learning, parameter search, portfolio optimization, live trading, broker integration, UI, scraping, or notifications are included.
"""


def write_portfolio_report(report: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
