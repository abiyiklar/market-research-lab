from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


ORDER_COLUMNS = (
    "order_id",
    "position_id",
    "symbol",
    "side",
    "order_status",
    "signal_date",
    "scheduled_execution_date",
    "actual_execution_date",
    "signal_type",
    "final_score",
    "cross_section_rank",
    "risk_quality_score",
    "score_state",
    "primary_exit_reason",
    "all_exit_reasons",
    "requested_target_value",
    "quantity",
    "raw_execution_price",
    "gross_transaction_value",
    "transaction_cost",
    "cash_before",
    "cash_after",
    "rejection_reason",
    "execution_delay_sessions",
    "execution_delay_calendar_days",
)

TRADE_COLUMNS = (
    "position_id",
    "symbol",
    "entry_signal_date",
    "entry_execution_date",
    "exit_signal_date",
    "exit_execution_date",
    "entry_signal_type",
    "exit_reason",
    "all_exit_reasons",
    "entry_score",
    "entry_rank",
    "entry_score_state",
    "sector_fallback_at_entry",
    "entry_price",
    "exit_price",
    "quantity",
    "exit_quantity",
    "dividends_received",
    "split_cash_in_lieu",
    "entry_costs",
    "exit_costs",
    "gross_pnl",
    "net_pnl",
    "net_return",
    "holding_sessions",
    "holding_calendar_days",
    "mfe",
    "mae",
)

DAILY_COLUMNS = (
    "date",
    "cash",
    "holdings_market_value",
    "total_portfolio_equity",
    "daily_return",
    "cumulative_return",
    "gross_exposure",
    "number_open_positions",
    "transaction_costs_paid_day",
    "cumulative_transaction_costs",
    "dividends_received_day",
    "cumulative_dividends",
    "realized_pnl_day",
    "cumulative_realized_pnl",
    "unrealized_pnl",
    "drawdown",
    "benchmark_value",
    "benchmark_relative_equity",
    "stale_valuation_count",
    "stale_valuation",
    "reconciliation_error",
)

POSITION_AUDIT_COLUMNS = (
    "date",
    "position_id",
    "symbol",
    "quantity",
    "raw_close_used",
    "market_value",
    "remaining_cost_basis",
    "unrealized_pnl",
    "stale_valuation",
    "valuation_age_sessions",
    "dividend_cash_day",
    "cumulative_position_dividends",
    "split_factor_day",
    "split_cash_in_lieu",
    "price_basis",
)

OPEN_POSITION_COLUMNS = (
    "position_id",
    "symbol",
    "entry_signal_date",
    "entry_execution_date",
    "entry_signal_type",
    "entry_score",
    "entry_rank",
    "entry_price",
    "entry_quantity",
    "current_quantity",
    "remaining_cost_basis",
    "dividends_received",
    "split_cash_in_lieu",
    "entry_costs",
    "last_observed_close",
    "market_value",
    "unrealized_pnl",
    "mfe",
    "mae",
    "pending_exit_order_id",
    "open_at_end",
)


@dataclass(frozen=True)
class PortfolioConfig:
    starting_capital: float = 1_000_000.0
    max_positions: int = 4
    target_weight: float = 0.25
    transaction_cost_rate: float = 0.0015
    execution_delay_sessions: int = 0
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        if self.starting_capital <= 0:
            raise ValueError("starting_capital must be positive")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if not 0 < self.target_weight <= 1:
            raise ValueError("target_weight must be greater than zero and at most one")
        if not 0 <= self.transaction_cost_rate < 1:
            raise ValueError("transaction_cost_rate must be between zero and one")
        if self.execution_delay_sessions < 0:
            raise ValueError("execution_delay_sessions cannot be negative")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive")


@dataclass
class Position:
    position_id: str
    symbol: str
    entry_signal_date: pd.Timestamp
    entry_execution_date: pd.Timestamp
    entry_signal_type: str
    entry_score: float
    entry_rank: float
    entry_score_state: str
    sector_fallback_at_entry: bool
    entry_price: float
    entry_quantity: int
    quantity: int
    original_entry_value: float
    remaining_cost_basis: float
    entry_costs: float
    entry_session_index: int
    price_basis: str
    last_observed_close: float
    last_observed_global_index: int
    dividends_received: float = 0.0
    split_cash_in_lieu: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    pending_exit_order_id: str = ""
    dividend_dates: set[pd.Timestamp] = field(default_factory=set)
    split_dates: set[pd.Timestamp] = field(default_factory=set)


@dataclass
class PendingOrder:
    record: dict[str, object]
    remaining_valid_opens: int


@dataclass(frozen=True)
class PortfolioBacktestResult:
    config: PortfolioConfig
    daily: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame
    open_positions: pd.DataFrame
    position_daily_audit: pd.DataFrame
    start_date: pd.Timestamp
    end_date: pd.Timestamp
