from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from bist_research.trading_calendar import only_tradable_tuprs_sessions

from .corporate_actions import PRICE_BASIS_UNADJUSTED
from .models import (
    EQUITY_COLUMNS,
    TRADE_COLUMNS,
    BacktestConfig,
    BacktestResult,
    PendingEntry,
    PendingExit,
    Position,
    Trade,
)
from .strategy import baseline_entry_signal, close_exit_reason, prepare_strategy_data


EntrySignal = Callable[[pd.Series, BacktestConfig], bool]
ExitSignal = Callable[[pd.Series, int, BacktestConfig], str | None]

REQUIRED_COLUMNS = (
    "date",
    "tuprs_open",
    "tuprs_high",
    "tuprs_low",
    "tuprs_close",
    "tuprs_volume",
    "xu100_close",
    "xu100_return_1d",
    "relative_strength_xu100",
    "relative_momentum_20d",
    "relative_momentum_60d",
    "volume_ratio_20d",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",
    "rsi_14",
    "atr_14",
    "is_indicator_warmup",
)

NUMERIC_REQUIRED_COLUMNS = tuple(
    column for column in REQUIRED_COLUMNS if column not in {"date", "is_indicator_warmup"}
)


def load_feature_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Backtest input file not found: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError("Backtest input must be a Parquet file")
    return validate_feature_data(pd.read_parquet(path))


def validate_feature_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {', '.join(missing)}")

    validated = frame.copy()
    optional_defaults: dict[str, object] = {
        "dividend_per_share": 0.0,
        "stock_split_factor": 0.0,
        "repaired_data": False,
        "corporate_action_flag": False,
        "price_basis": "raw_ohlc_split_basis_unknown",
    }
    for column, default in optional_defaults.items():
        if column not in validated:
            validated[column] = default
    validated["date"] = pd.to_datetime(validated["date"], errors="raise").dt.normalize()
    validated = validated.sort_values("date").reset_index(drop=True)
    if validated["date"].duplicated().any():
        raise ValueError("Feature dataset contains duplicate dates")
    if validated["is_indicator_warmup"].isna().any():
        raise ValueError("is_indicator_warmup contains missing values")

    validated = only_tradable_tuprs_sessions(validated)
    validated["is_tradable_tuprs_session"] = True
    active = validated.loc[~validated["is_indicator_warmup"].astype(bool)]
    if active.empty:
        raise ValueError("Feature dataset has no rows outside the indicator warm-up period")
    numeric = active.loc[:, NUMERIC_REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise ValueError("Feature dataset contains NaN or infinite values outside warm-up rows")
    if (active[["tuprs_open", "tuprs_high", "tuprs_low", "tuprs_close"]] <= 0).any().any():
        raise ValueError("TUPRS price columns must be positive outside warm-up rows")
    if (active["tuprs_high"] < active["tuprs_low"]).any():
        raise ValueError("TUPRS high price cannot be below low price")
    return validated


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig | None = None,
        entry_signal: EntrySignal = baseline_entry_signal,
        exit_signal: ExitSignal = close_exit_reason,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.entry_signal = entry_signal
        self.exit_signal = exit_signal
        self.logger = logger or logging.getLogger("bist_research.backtest")

    def run(self, feature_data: pd.DataFrame) -> BacktestResult:
        validated = validate_feature_data(feature_data)
        prepared = prepare_strategy_data(validated)
        rows = prepared.loc[~prepared["is_indicator_warmup"].astype(bool)].reset_index(drop=True)

        cash = self.config.initial_capital
        position: Position | None = None
        pending_entry: PendingEntry | None = None
        pending_exit: PendingExit | None = None
        trades: list[Trade] = []
        equity_rows: list[dict[str, object]] = []
        cumulative_dividend_cash = 0.0

        for row_number, row in rows.iterrows():
            date = pd.Timestamp(row["date"])
            is_last_row = row_number == len(rows) - 1
            dividend_cash = 0.0

            if position is not None:
                self._apply_stock_split(position, row)
                dividend_cash = self._credit_dividend(position, row)
                cash += dividend_cash
                cumulative_dividend_cash += dividend_cash

            if position is None and pending_entry is not None:
                if pending_entry.sessions_until_entry <= 1:
                    position, cash = self._open_position(
                        row, pending_entry, len(trades) + 1, cash
                    )
                    pending_entry = None
                else:
                    pending_entry = PendingEntry(
                        signal_date=pending_entry.signal_date,
                        atr_at_signal=pending_entry.atr_at_signal,
                        sessions_until_entry=pending_entry.sessions_until_entry - 1,
                    )

            if position is not None:
                position.holding_days += 1
                stop_price = position.current_stop
                stop_reason = self._stop_reason(position)

                # A gap through the known stop happens before a pending open exit.
                if float(row["tuprs_open"]) <= stop_price:
                    self._update_exit_excursions(position, float(row["tuprs_open"]))
                    trade, cash = self._close_position(
                        position,
                        date,
                        float(row["tuprs_open"]),
                        stop_reason,
                        cash,
                    )
                    trades.append(trade)
                    position = None
                    pending_exit = None
                elif pending_exit is not None:
                    self._update_exit_excursions(position, float(row["tuprs_open"]))
                    trade, cash = self._close_position(
                        position,
                        date,
                        float(row["tuprs_open"]),
                        pending_exit.reason,
                        cash,
                    )
                    trades.append(trade)
                    position = None
                    pending_exit = None
                elif float(row["tuprs_low"]) <= stop_price:
                    self._update_exit_excursions(position, stop_price)
                    trade, cash = self._close_position(
                        position,
                        date,
                        stop_price,
                        stop_reason,
                        cash,
                    )
                    trades.append(trade)
                    position = None
                else:
                    self._update_excursions(position, row)
                    position.highest_close = max(position.highest_close, float(row["tuprs_close"]))
                    if is_last_row:
                        trade, cash = self._close_position(
                            position,
                            date,
                            float(row["tuprs_close"]),
                            "end_of_period",
                            cash,
                        )
                        trades.append(trade)
                        position = None
                    else:
                        reason = self.exit_signal(row, position.holding_days, self.config)
                        if reason is not None:
                            pending_exit = PendingExit(signal_date=date, reason=reason)
                        trailing_stop = (
                            position.highest_close
                            - self.config.atr_trailing_multiplier * float(row["atr_14"])
                        )
                        position.current_stop = max(position.initial_stop, trailing_stop)

            if (
                position is None
                and pending_entry is None
                and pending_exit is None
                and not is_last_row
                and self._can_schedule_entry(row)
            ):
                pending_entry = PendingEntry(
                    date,
                    float(row["atr_14"]),
                    self.config.entry_delay_days,
                )

            equity_rows.append(
                self._equity_row(
                    row,
                    cash,
                    position,
                    dividend_cash,
                    cumulative_dividend_cash,
                )
            )

        trade_frame = pd.DataFrame((trade.to_dict() for trade in trades), columns=TRADE_COLUMNS)
        equity_frame = self._finalize_equity(pd.DataFrame(equity_rows))
        return BacktestResult(trades=trade_frame, daily_equity=equity_frame, config=self.config)

    def _can_schedule_entry(self, row: pd.Series) -> bool:
        return (
            float(row["tuprs_volume"]) > 0
            and math.isfinite(float(row["atr_14"]))
            and float(row["atr_14"]) > 0
            and self.entry_signal(row, self.config)
        )

    def _open_position(
        self,
        row: pd.Series,
        pending: PendingEntry,
        trade_id: int,
        cash: float,
    ) -> tuple[Position | None, float]:
        if float(row["tuprs_volume"]) <= 0:
            self.logger.info("Skipped entry on %s because volume is not positive", row["date"])
            return None, cash

        raw_price = float(row["tuprs_open"])
        effective_price = raw_price * (1 + self.config.slippage_rate)
        budget = cash * self.config.position_size
        per_share_cost = effective_price * (1 + self.config.commission_rate)
        quantity = math.floor(budget / per_share_cost)
        if quantity <= 0:
            self.logger.info("Skipped entry on %s because available cash is insufficient", row["date"])
            return None, cash

        entry_commission = quantity * effective_price * self.config.commission_rate
        cash_after_entry = cash - quantity * effective_price - entry_commission
        initial_stop = effective_price - self.config.atr_stop_multiplier * pending.atr_at_signal
        position = Position(
            trade_id=trade_id,
            signal_date=pending.signal_date,
            entry_date=pd.Timestamp(row["date"]),
            entry_price_raw=raw_price,
            entry_price_effective=effective_price,
            quantity=quantity,
            entry_commission=entry_commission,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            highest_close=raw_price,
            maximum_price=raw_price,
            minimum_price=raw_price,
            price_basis=str(row.get("price_basis", "raw_ohlc_split_basis_unknown")),
        )
        self.logger.debug(
            "Opened trade %s on %s at %.4f for %s shares",
            trade_id,
            row["date"],
            effective_price,
            quantity,
        )
        return position, cash_after_entry

    def _close_position(
        self,
        position: Position,
        exit_date: pd.Timestamp,
        raw_price: float,
        reason: str,
        cash: float,
    ) -> tuple[Trade, float]:
        effective_price = raw_price * (1 - self.config.slippage_rate)
        exit_commission = position.quantity * effective_price * self.config.commission_rate
        cash_after_exit = cash + position.quantity * effective_price - exit_commission

        entry_slippage = position.quantity * (
            position.entry_price_effective - position.entry_price_raw
        )
        exit_slippage = position.quantity * (raw_price - effective_price)
        slippage_cost = entry_slippage + exit_slippage
        gross_pnl = (
            position.quantity * (raw_price - position.entry_price_raw)
            + position.cumulative_dividend_cash
        )
        net_pnl = (
            position.quantity * (effective_price - position.entry_price_effective)
            - position.entry_commission
            - exit_commission
            + position.cumulative_dividend_cash
        )
        invested_capital = (
            position.quantity * position.entry_price_effective + position.entry_commission
        )
        trade_return = net_pnl / invested_capital if invested_capital else 0.0
        favorable = (position.maximum_price - position.entry_price_raw) / position.entry_price_raw
        adverse = (position.minimum_price - position.entry_price_raw) / position.entry_price_raw

        trade = Trade(
            trade_id=position.trade_id,
            signal_date=position.signal_date,
            entry_date=position.entry_date,
            entry_price_raw=position.entry_price_raw,
            entry_price_effective=position.entry_price_effective,
            quantity=position.quantity,
            entry_commission=position.entry_commission,
            initial_stop=position.initial_stop,
            exit_date=exit_date,
            exit_price_raw=raw_price,
            exit_price_effective=effective_price,
            exit_reason=reason,
            exit_commission=exit_commission,
            slippage_cost=slippage_cost,
            dividend_cash=position.cumulative_dividend_cash,
            cumulative_dividend_cash=position.cumulative_dividend_cash,
            stock_split_factor=position.cumulative_stock_split_factor,
            corporate_action_flag=position.corporate_action_flag,
            price_basis=position.price_basis,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_pct=trade_return,
            holding_days=position.holding_days,
            maximum_favorable_excursion=favorable,
            maximum_adverse_excursion=adverse,
        )
        self.logger.debug(
            "Closed trade %s on %s at %.4f (%s)",
            position.trade_id,
            exit_date.date(),
            effective_price,
            reason,
        )
        return trade, cash_after_exit

    @staticmethod
    def _update_excursions(position: Position, row: pd.Series) -> None:
        position.maximum_price = max(position.maximum_price, float(row["tuprs_high"]))
        position.minimum_price = min(position.minimum_price, float(row["tuprs_low"]))

    @staticmethod
    def _update_exit_excursions(position: Position, exit_price: float) -> None:
        position.maximum_price = max(position.maximum_price, exit_price)
        position.minimum_price = min(position.minimum_price, exit_price)

    @staticmethod
    def _stop_reason(position: Position) -> str:
        return (
            "trailing_stop"
            if position.current_stop > position.initial_stop + 1e-12
            else "atr_stop"
        )

    def _credit_dividend(self, position: Position, row: pd.Series) -> float:
        per_share = float(row.get("dividend_per_share", 0.0) or 0.0)
        if not math.isfinite(per_share) or per_share <= 0:
            return 0.0
        cash = (
            position.quantity
            * per_share
            * (1 - self.config.dividend_withholding_rate)
        )
        position.cumulative_dividend_cash += cash
        position.corporate_action_flag = True
        return cash

    @staticmethod
    def _apply_stock_split(position: Position, row: pd.Series) -> None:
        factor = float(row.get("stock_split_factor", 0.0) or 0.0)
        if not math.isfinite(factor) or factor <= 0:
            return
        position.corporate_action_flag = True
        if str(row.get("price_basis", position.price_basis)) != PRICE_BASIS_UNADJUSTED:
            return
        position.quantity = int(round(position.quantity * factor))
        position.entry_price_raw /= factor
        position.entry_price_effective /= factor
        position.initial_stop /= factor
        position.current_stop /= factor
        position.highest_close /= factor
        position.maximum_price /= factor
        position.minimum_price /= factor
        position.cumulative_stock_split_factor *= factor
        position.price_basis = PRICE_BASIS_UNADJUSTED

    @staticmethod
    def _equity_row(
        row: pd.Series,
        cash: float,
        position: Position | None,
        dividend_cash: float,
        cumulative_dividend_cash: float,
    ) -> dict[str, object]:
        quantity = position.quantity if position is not None else 0
        market_value = quantity * float(row["tuprs_close"])
        return {
            "date": pd.Timestamp(row["date"]),
            "cash": cash,
            "position_quantity": quantity,
            "market_value": market_value,
            "total_equity": cash + market_value,
            "position_open": position is not None,
            "current_stop": position.current_stop if position is not None else 0.0,
            "dividend_per_share": float(row.get("dividend_per_share", 0.0) or 0.0),
            "dividend_cash": dividend_cash,
            "cumulative_dividend_cash": cumulative_dividend_cash,
            "stock_split_factor": float(row.get("stock_split_factor", 0.0) or 0.0),
            "corporate_action_flag": bool(row.get("corporate_action_flag", False)),
            "price_basis": str(row.get("price_basis", "raw_ohlc_split_basis_unknown")),
        }

    def _finalize_equity(self, equity: pd.DataFrame) -> pd.DataFrame:
        if equity.empty:
            return pd.DataFrame(columns=EQUITY_COLUMNS)
        equity = equity.copy()
        equity["daily_return"] = equity["total_equity"].pct_change(fill_method=None).fillna(0.0)
        equity["cumulative_return"] = equity["total_equity"] / self.config.initial_capital - 1
        equity["running_peak"] = equity["total_equity"].cummax()
        equity["drawdown"] = equity["total_equity"] / equity["running_peak"] - 1
        return equity.loc[:, EQUITY_COLUMNS]
