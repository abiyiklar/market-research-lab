from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


TRADE_COLUMNS = (
    "trade_id",
    "signal_date",
    "entry_date",
    "entry_price_raw",
    "entry_price_effective",
    "quantity",
    "entry_commission",
    "initial_stop",
    "exit_date",
    "exit_price_raw",
    "exit_price_effective",
    "exit_reason",
    "exit_commission",
    "slippage_cost",
    "dividend_cash",
    "cumulative_dividend_cash",
    "stock_split_factor",
    "corporate_action_flag",
    "price_basis",
    "gross_pnl",
    "net_pnl",
    "return_pct",
    "holding_days",
    "maximum_favorable_excursion",
    "maximum_adverse_excursion",
)

EQUITY_COLUMNS = (
    "date",
    "cash",
    "position_quantity",
    "market_value",
    "total_equity",
    "daily_return",
    "cumulative_return",
    "running_peak",
    "drawdown",
    "position_open",
    "current_stop",
    "dividend_per_share",
    "dividend_cash",
    "cumulative_dividend_cash",
    "stock_split_factor",
    "corporate_action_flag",
    "price_basis",
)


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    position_size: float = 1.0
    atr_stop_multiplier: float = 2.5
    atr_trailing_multiplier: float = 3.0
    maximum_holding_days: int = 60
    minimum_volume_ratio: float = 1.10
    minimum_rsi: float = 50.0
    maximum_entry_rsi: float = 72.0
    maximum_exit_rsi: float = 78.0
    minimum_market_return: float = -0.03
    risk_free_rate: float = 0.0
    trading_days_per_year: int = 252
    entry_delay_days: int = 1
    dividend_withholding_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 <= self.commission_rate < 1:
            raise ValueError("commission_rate must be between 0 and 1")
        if not 0 <= self.slippage_rate < 1:
            raise ValueError("slippage_rate must be between 0 and 1")
        if not 0 < self.position_size <= 1:
            raise ValueError("position_size must be greater than 0 and at most 1")
        if self.atr_stop_multiplier <= 0 or self.atr_trailing_multiplier <= 0:
            raise ValueError("ATR multipliers must be positive")
        if self.maximum_holding_days <= 0:
            raise ValueError("maximum_holding_days must be positive")
        if self.minimum_rsi > self.maximum_entry_rsi:
            raise ValueError("minimum_rsi cannot exceed maximum_entry_rsi")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive")
        if self.entry_delay_days <= 0:
            raise ValueError("entry_delay_days must be positive")
        if not 0 <= self.dividend_withholding_rate < 1:
            raise ValueError("dividend_withholding_rate must be between 0 and 1")


@dataclass(frozen=True)
class PendingEntry:
    signal_date: pd.Timestamp
    atr_at_signal: float
    sessions_until_entry: int = 1


@dataclass(frozen=True)
class PendingExit:
    signal_date: pd.Timestamp
    reason: str


@dataclass
class Position:
    trade_id: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price_raw: float
    entry_price_effective: float
    quantity: int
    entry_commission: float
    initial_stop: float
    current_stop: float
    highest_close: float
    maximum_price: float
    minimum_price: float
    holding_days: int = 0
    cumulative_dividend_cash: float = 0.0
    cumulative_stock_split_factor: float = 1.0
    corporate_action_flag: bool = False
    price_basis: str = "raw_ohlc_split_basis_unknown"


@dataclass(frozen=True)
class Trade:
    trade_id: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price_raw: float
    entry_price_effective: float
    quantity: int
    entry_commission: float
    initial_stop: float
    exit_date: pd.Timestamp
    exit_price_raw: float
    exit_price_effective: float
    exit_reason: str
    exit_commission: float
    slippage_cost: float
    dividend_cash: float
    cumulative_dividend_cash: float
    stock_split_factor: float
    corporate_action_flag: bool
    price_basis: str
    gross_pnl: float
    net_pnl: float
    return_pct: float
    holding_days: int
    maximum_favorable_excursion: float
    maximum_adverse_excursion: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    trades: pd.DataFrame
    daily_equity: pd.DataFrame
    config: BacktestConfig


@dataclass(frozen=True)
class PeriodDefinition:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None


PERIOD_DEFINITIONS = (
    PeriodDefinition("train", pd.Timestamp("2013-01-01"), pd.Timestamp("2019-12-31")),
    PeriodDefinition("validation", pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31")),
    PeriodDefinition("test", pd.Timestamp("2023-01-01"), None),
)
