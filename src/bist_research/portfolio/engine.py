from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bist_research.backtest.corporate_actions import (
    PRICE_BASIS_SPLIT_ADJUSTED,
    PRICE_BASIS_UNADJUSTED,
    PRICE_BASIS_UNKNOWN,
)
from bist_research.config import SymbolConfig

from .models import (
    DAILY_COLUMNS,
    OPEN_POSITION_COLUMNS,
    ORDER_COLUMNS,
    POSITION_AUDIT_COLUMNS,
    TRADE_COLUMNS,
    PendingOrder,
    PortfolioBacktestResult,
    PortfolioConfig,
    Position,
)


PANEL_REQUIRED_COLUMNS = {
    "symbol",
    "date",
    "decision_label",
    "is_buy_event",
    "final_score",
    "cross_section_rank",
    "risk_quality_score",
    "score_state",
    "signal_type",
    "primary_exit_reason",
    "exit_reasons",
    "insufficient_cross_section",
    "valid_score_input",
    "risk_data_quality_problem",
    "quality_repaired",
    "quality_valid_ohlc",
    "quality_positive_volume",
    "executable_entry_date",
    "executable_entry_signal_open",
    "sector_fallback_neutralized",
}

FEATURE_REQUIRED_COLUMNS = {
    "symbol",
    "date",
    "execution_open",
    "execution_high",
    "execution_low",
    "execution_close",
    "volume",
    "dividend",
    "stock_split",
    "source_symbol",
}


def _finite_positive(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0


def _normalize_date(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None).normalize()


def causal_split_price_basis(
    previous_close: float,
    event_open: float,
    split_factor: float,
) -> str:
    values = (previous_close, event_open, split_factor)
    if not all(math.isfinite(value) and value > 0 for value in values):
        return PRICE_BASIS_UNKNOWN
    observed_ratio = previous_close / event_open
    continuous_distance = abs(math.log(observed_ratio))
    unadjusted_distance = abs(math.log(observed_ratio / split_factor))
    if continuous_distance <= unadjusted_distance:
        return PRICE_BASIS_SPLIT_ADJUSTED
    return PRICE_BASIS_UNADJUSTED


def prepare_portfolio_inputs(
    panel: pd.DataFrame,
    feature_frames: Mapping[str, pd.DataFrame],
    configs: Sequence[SymbolConfig],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    missing_panel = sorted(PANEL_REQUIRED_COLUMNS.difference(panel.columns))
    if missing_panel:
        raise ValueError(f"Signal panel is missing columns: {', '.join(missing_panel)}")

    configured = {config.symbol for config in configs}
    prepared_panel = panel.loc[panel["symbol"].isin(configured)].copy()
    prepared_panel["date"] = pd.to_datetime(
        prepared_panel["date"], errors="raise"
    ).dt.tz_localize(None).dt.normalize()
    prepared_panel = prepared_panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    if prepared_panel.duplicated(["symbol", "date"]).any():
        raise ValueError("Signal panel contains duplicate symbol/date rows")
    observed_panel_symbols = set(prepared_panel["symbol"].dropna().astype(str).unique())
    unexpected = observed_panel_symbols.difference(configured)
    if unexpected:
        raise ValueError(f"Signal panel contains unexpected symbols: {sorted(unexpected)}")

    prepared_features: dict[str, pd.DataFrame] = {}
    for config in configs:
        if config.symbol not in feature_frames:
            raise FileNotFoundError(f"Missing feature data for {config.symbol}")
        frame = feature_frames[config.symbol].copy()
        missing = sorted(FEATURE_REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(
                f"{config.symbol} feature data is missing columns: {', '.join(missing)}"
            )
        observed = set(frame["symbol"].dropna().astype(str).unique())
        source = set(frame["source_symbol"].dropna().astype(str).unique())
        if observed.difference({config.symbol}) or source.difference({config.symbol}):
            raise ValueError(f"Cross-symbol contamination found in {config.symbol} features")
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.tz_localize(
            None
        ).dt.normalize()
        frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(
            drop=True
        )
        numeric = frame[
            [
                "execution_open",
                "execution_high",
                "execution_low",
                "execution_close",
                "volume",
                "dividend",
                "stock_split",
            ]
        ].apply(pd.to_numeric, errors="coerce")
        frame[numeric.columns] = numeric
        frame["is_valid_stock_session"] = (
            np.isfinite(
                frame[
                    ["execution_open", "execution_high", "execution_low", "execution_close"]
                ].to_numpy(dtype="float64")
            ).all(axis=1)
            & frame[
                ["execution_open", "execution_high", "execution_low", "execution_close"]
            ].gt(0).all(axis=1)
            & frame["volume"].gt(0)
            & frame["execution_high"].ge(frame["execution_low"])
        )
        if "price_basis" not in frame:
            frame["price_basis"] = PRICE_BASIS_UNKNOWN
        prepared_features[config.symbol] = frame

    return prepared_panel, prepared_features


@dataclass(frozen=True)
class PortfolioInputContext:
    panel: pd.DataFrame
    features: dict[str, pd.DataFrame]
    start_date: pd.Timestamp
    all_dates: tuple[pd.Timestamp, ...]
    date_to_index: dict[pd.Timestamp, int]
    feature_lookup: dict[str, dict[pd.Timestamp, dict[str, object]]]
    session_indices: dict[str, dict[pd.Timestamp, int]]
    signals_by_date: dict[pd.Timestamp, list[object]]


def build_portfolio_context(
    panel: pd.DataFrame,
    feature_frames: Mapping[str, pd.DataFrame],
    configs: Sequence[SymbolConfig],
) -> PortfolioInputContext:
    scored, features = prepare_portfolio_inputs(panel, feature_frames, configs)
    scored_rows = scored.loc[scored["final_score"].notna()]
    if scored_rows.empty:
        raise ValueError("Signal panel has no valid scored cross-section")
    start_date = pd.Timestamp(scored_rows["date"].min())
    all_dates = tuple(
        sorted(
            {
                pd.Timestamp(date)
                for frame in features.values()
                for date in frame.loc[frame["date"].ge(start_date), "date"]
            }
            | set(scored.loc[scored["date"].ge(start_date), "date"])
        )
    )
    if not all_dates:
        raise ValueError("No portfolio dates are available")
    return PortfolioInputContext(
        panel=scored,
        features=features,
        start_date=start_date,
        all_dates=all_dates,
        date_to_index={date: index for index, date in enumerate(all_dates)},
        feature_lookup={
            symbol: frame.set_index("date", drop=False).to_dict(orient="index")
            for symbol, frame in features.items()
        },
        session_indices={
            symbol: {
                pd.Timestamp(date): index
                for index, date in enumerate(
                    frame.loc[frame["is_valid_stock_session"], "date"].tolist()
                )
            }
            for symbol, frame in features.items()
        },
        signals_by_date={
            pd.Timestamp(date): list(group.sort_values("symbol").itertuples(index=False))
            for date, group in scored.loc[scored["date"].isin(all_dates)].groupby(
                "date", sort=True
            )
        },
    )


class PortfolioBacktestEngine:
    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()

    def run(
        self,
        panel: pd.DataFrame,
        feature_frames: Mapping[str, pd.DataFrame],
        configs: Sequence[SymbolConfig],
    ) -> PortfolioBacktestResult:
        context = build_portfolio_context(panel, feature_frames, configs)
        return self.run_prepared(context)

    def run_prepared(self, context: PortfolioInputContext) -> PortfolioBacktestResult:
        scored = context.panel
        features = context.features
        start_date = context.start_date
        all_dates = context.all_dates
        date_to_index = context.date_to_index
        feature_lookup = context.feature_lookup
        session_indices = context.session_indices
        signals_by_date = context.signals_by_date

        cash = float(self.config.starting_capital)
        positions: dict[str, Position] = {}
        pending_buys: dict[str, PendingOrder] = {}
        pending_sells: dict[str, PendingOrder] = {}
        orders: list[dict[str, object]] = []
        trades: list[dict[str, object]] = []
        daily_rows: list[dict[str, object]] = []
        audit_rows: list[dict[str, object]] = []
        cumulative_costs = 0.0
        cumulative_dividends = 0.0
        cumulative_realized = 0.0
        order_number = 0
        position_number = 0

        for global_index, date in enumerate(all_dates):
            costs_day = 0.0
            dividends_day = 0.0
            realized_day = 0.0
            dividend_by_symbol: dict[str, float] = {}
            split_by_symbol: dict[str, float] = {}

            for symbol, position in list(positions.items()):
                row = self._feature_row(feature_lookup, symbol, date)
                if row is None or not bool(row["is_valid_stock_session"]):
                    continue
                split_cash, split_realized, split_factor = self._apply_split(position, row)
                if split_cash:
                    cash += split_cash
                    cumulative_realized += split_realized
                    realized_day += split_realized
                split_by_symbol[symbol] = split_factor
                dividend_cash = self._credit_dividend(position, row, date)
                if dividend_cash:
                    cash += dividend_cash
                    cumulative_dividends += dividend_cash
                    dividends_day += dividend_cash
                    dividend_by_symbol[symbol] = dividend_cash

            for symbol in sorted(list(pending_sells)):
                pending = pending_sells[symbol]
                if pd.Timestamp(pending.record["scheduled_execution_date"]) > date:
                    continue
                row = self._feature_row(feature_lookup, symbol, date)
                if row is None or not bool(row["is_valid_stock_session"]):
                    continue
                pending.remaining_valid_opens -= 1
                if pending.remaining_valid_opens > 0:
                    continue
                if symbol not in positions:
                    self._reject_order(pending.record, "position_not_open")
                    del pending_sells[symbol]
                    continue
                position = positions[symbol]
                cash_before = cash
                price = float(row["execution_open"])
                quantity = int(position.quantity)
                gross = quantity * price
                cost = gross * self.config.transaction_cost_rate
                cash = cash + gross - cost
                cumulative_costs += cost
                costs_day += cost
                exit_realized = gross - position.remaining_cost_basis
                cumulative_realized += exit_realized
                realized_day += exit_realized
                self._update_excursion(position, price, price)
                trade = self._completed_trade(
                    position,
                    pending.record,
                    date,
                    price,
                    quantity,
                    cost,
                    session_indices[symbol].get(date, position.entry_session_index),
                )
                trades.append(trade)
                self._fill_order(
                    pending.record,
                    date,
                    price,
                    quantity,
                    gross,
                    cost,
                    cash_before,
                    cash,
                    date_to_index,
                )
                del positions[symbol]
                del pending_sells[symbol]

            executable_buys: list[PendingOrder] = []
            for symbol, pending in list(pending_buys.items()):
                if pd.Timestamp(pending.record["scheduled_execution_date"]) > date:
                    continue
                row = self._feature_row(feature_lookup, symbol, date)
                if row is None or not bool(row["is_valid_stock_session"]):
                    continue
                pending.remaining_valid_opens -= 1
                if pending.remaining_valid_opens <= 0:
                    executable_buys.append(pending)
            executable_buys.sort(key=self._buy_sort_key)

            for pending in executable_buys:
                symbol = str(pending.record["symbol"])
                if symbol in positions:
                    self._reject_order(pending.record, "already_held")
                    pending_buys.pop(symbol, None)
                    continue
                if len(positions) >= self.config.max_positions:
                    self._reject_order(pending.record, "max_positions_reached")
                    pending_buys.pop(symbol, None)
                    continue
                row = self._feature_row(feature_lookup, symbol, date)
                if row is None:
                    continue
                equity_before = cash + self._holdings_value_at_open(
                    positions, feature_lookup, date
                )
                requested_target = equity_before * self.config.target_weight
                available_target = min(requested_target, cash)
                price = float(row["execution_open"])
                quantity = math.floor(
                    available_target
                    / (price * (1.0 + self.config.transaction_cost_rate))
                )
                pending.record["requested_target_value"] = requested_target
                if quantity <= 0:
                    self._reject_order(pending.record, "zero_quantity")
                    pending_buys.pop(symbol, None)
                    continue
                gross = quantity * price
                cost = gross * self.config.transaction_cost_rate
                total = gross + cost
                if total > cash + 1e-8:
                    self._reject_order(pending.record, "insufficient_cash")
                    pending_buys.pop(symbol, None)
                    continue
                cash_before = cash
                cash -= total
                if cash < -1e-8:
                    raise AssertionError("Portfolio cash became negative")
                cash = max(cash, 0.0)
                cumulative_costs += cost
                costs_day += cost
                position_number += 1
                position_id = f"P{position_number:06d}"
                price_basis = str(row.get("price_basis", "raw_ohlc_split_basis_unknown"))
                session_index = session_indices[symbol][date]
                positions[symbol] = Position(
                    position_id=position_id,
                    symbol=symbol,
                    entry_signal_date=pd.Timestamp(pending.record["signal_date"]),
                    entry_execution_date=date,
                    entry_signal_type=str(pending.record["signal_type"]),
                    entry_score=float(pending.record["final_score"]),
                    entry_rank=float(pending.record["cross_section_rank"]),
                    entry_score_state=str(pending.record["score_state"]),
                    sector_fallback_at_entry=bool(
                        pending.record.get("sector_fallback_at_entry", False)
                    ),
                    entry_price=price,
                    entry_quantity=quantity,
                    quantity=quantity,
                    original_entry_value=gross,
                    remaining_cost_basis=gross,
                    entry_costs=cost,
                    entry_session_index=session_index,
                    price_basis=price_basis,
                    last_observed_close=price,
                    last_observed_global_index=global_index,
                )
                pending.record["position_id"] = position_id
                self._fill_order(
                    pending.record,
                    date,
                    price,
                    quantity,
                    gross,
                    cost,
                    cash_before,
                    cash,
                    date_to_index,
                )
                pending_buys.pop(symbol, None)

            holdings_value = 0.0
            unrealized = 0.0
            stale_count = 0
            for symbol, position in positions.items():
                row = self._feature_row(feature_lookup, symbol, date)
                stale = row is None or not bool(row["is_valid_stock_session"])
                if not stale:
                    close = float(row["execution_close"])
                    position.last_observed_close = close
                    position.last_observed_global_index = global_index
                    self._update_excursion(
                        position,
                        float(row["execution_high"]),
                        float(row["execution_low"]),
                    )
                else:
                    close = position.last_observed_close
                    stale_count += 1
                market_value = position.quantity * close
                position_unrealized = market_value - position.remaining_cost_basis
                holdings_value += market_value
                unrealized += position_unrealized
                audit_rows.append(
                    {
                        "date": date,
                        "position_id": position.position_id,
                        "symbol": symbol,
                        "quantity": position.quantity,
                        "raw_close_used": close,
                        "market_value": market_value,
                        "remaining_cost_basis": position.remaining_cost_basis,
                        "unrealized_pnl": position_unrealized,
                        "stale_valuation": stale,
                        "valuation_age_sessions": global_index
                        - position.last_observed_global_index,
                        "dividend_cash_day": dividend_by_symbol.get(symbol, 0.0),
                        "cumulative_position_dividends": position.dividends_received,
                        "split_factor_day": split_by_symbol.get(symbol, 0.0),
                        "split_cash_in_lieu": position.split_cash_in_lieu,
                        "price_basis": position.price_basis,
                    }
                )

            equity = cash + holdings_value
            reconciliation = (
                self.config.starting_capital
                + cumulative_realized
                + unrealized
                + cumulative_dividends
                - cumulative_costs
            )
            reconciliation_error = equity - reconciliation
            if abs(reconciliation_error) > max(1e-6, equity * 1e-10):
                raise AssertionError(
                    f"Portfolio accounting failed to reconcile on {date.date()}: "
                    f"{reconciliation_error}"
                )
            daily_rows.append(
                {
                    "date": date,
                    "cash": cash,
                    "holdings_market_value": holdings_value,
                    "total_portfolio_equity": equity,
                    "gross_exposure": holdings_value / equity if equity > 0 else 0.0,
                    "number_open_positions": len(positions),
                    "transaction_costs_paid_day": costs_day,
                    "cumulative_transaction_costs": cumulative_costs,
                    "dividends_received_day": dividends_day,
                    "cumulative_dividends": cumulative_dividends,
                    "realized_pnl_day": realized_day,
                    "cumulative_realized_pnl": cumulative_realized,
                    "unrealized_pnl": unrealized,
                    "stale_valuation_count": stale_count,
                    "stale_valuation": stale_count > 0,
                    "reconciliation_error": reconciliation_error,
                }
            )

            signal_rows = signals_by_date.get(date)
            if signal_rows is not None:
                next_date = all_dates[global_index + 1] if global_index + 1 < len(all_dates) else None
                for row in signal_rows:
                    symbol = str(row.symbol)
                    if symbol in positions and str(row.decision_label) == "SELL":
                        if symbol not in pending_sells:
                            order_number += 1
                            record = self._new_order_record(
                                order_number, symbol, "SELL", row, next_date
                            )
                            record["position_id"] = positions[symbol].position_id
                            orders.append(record)
                            if next_date is None:
                                record["order_status"] = "open_at_end"
                                record["rejection_reason"] = "no_future_valid_session"
                            else:
                                pending_sells[symbol] = PendingOrder(
                                    record,
                                    1 + self.config.execution_delay_sessions,
                                )
                                positions[symbol].pending_exit_order_id = str(
                                    record["order_id"]
                                )

                for row in signal_rows:
                    if str(row.decision_label) != "BUY" or not bool(row.is_buy_event):
                        continue
                    symbol = str(row.symbol)
                    order_number += 1
                    record = self._new_order_record(order_number, symbol, "BUY", row, next_date)
                    record["sector_fallback_at_entry"] = bool(
                        row.sector_fallback_neutralized
                    )
                    orders.append(record)
                    rejection = self._entry_rejection_reason(row)
                    if symbol in positions:
                        rejection = "already_held"
                    elif symbol in pending_buys:
                        rejection = "entry_already_pending"
                    elif next_date is None:
                        rejection = "no_future_valid_session"
                    if rejection:
                        self._reject_order(record, rejection)
                    else:
                        pending_buys[symbol] = PendingOrder(
                            record,
                            1 + self.config.execution_delay_sessions,
                        )

        for pending in pending_buys.values():
            self._reject_order(pending.record, "no_future_valid_session", "open_at_end")
        for symbol, pending in pending_sells.items():
            pending.record["order_status"] = "open_at_end"
            pending.record["rejection_reason"] = "no_future_valid_session"
            if symbol in positions:
                positions[symbol].pending_exit_order_id = str(pending.record["order_id"])

        daily = self._finalize_daily(pd.DataFrame(daily_rows))
        order_frame = pd.DataFrame(orders).reindex(columns=ORDER_COLUMNS)
        trade_frame = pd.DataFrame(trades).reindex(columns=TRADE_COLUMNS)
        audit_frame = pd.DataFrame(audit_rows).reindex(columns=POSITION_AUDIT_COLUMNS)
        open_frame = self._open_positions_frame(positions)
        self._validate_result(daily, order_frame, trade_frame, open_frame, audit_frame)
        return PortfolioBacktestResult(
            config=self.config,
            daily=daily,
            orders=order_frame,
            trades=trade_frame,
            open_positions=open_frame,
            position_daily_audit=audit_frame,
            start_date=start_date,
            end_date=pd.Timestamp(all_dates[-1]),
        )

    @staticmethod
    def _feature_row(
        lookup: Mapping[str, Mapping[pd.Timestamp, dict[str, object]]],
        symbol: str,
        date: pd.Timestamp,
    ) -> dict[str, object] | None:
        return lookup[symbol].get(date)

    @staticmethod
    def _new_order_record(
        number: int,
        symbol: str,
        side: str,
        row: object,
        scheduled_date: pd.Timestamp | None,
    ) -> dict[str, object]:
        return {
            "order_id": f"O{number:07d}",
            "position_id": "",
            "symbol": symbol,
            "side": side,
            "order_status": "pending" if scheduled_date is not None else "rejected",
            "signal_date": pd.Timestamp(getattr(row, "date")),
            "scheduled_execution_date": scheduled_date,
            "actual_execution_date": pd.NaT,
            "signal_type": str(getattr(row, "signal_type", "")),
            "final_score": float(getattr(row, "final_score", np.nan)),
            "cross_section_rank": float(getattr(row, "cross_section_rank", np.nan)),
            "risk_quality_score": float(getattr(row, "risk_quality_score", np.nan)),
            "score_state": str(getattr(row, "score_state", "")),
            "primary_exit_reason": str(getattr(row, "primary_exit_reason", "")),
            "all_exit_reasons": str(getattr(row, "exit_reasons", "")),
            "requested_target_value": 0.0,
            "quantity": 0,
            "raw_execution_price": np.nan,
            "gross_transaction_value": 0.0,
            "transaction_cost": 0.0,
            "cash_before": np.nan,
            "cash_after": np.nan,
            "rejection_reason": "" if scheduled_date is not None else "no_future_session",
            "execution_delay_sessions": 0,
            "execution_delay_calendar_days": 0,
        }

    @staticmethod
    def _entry_rejection_reason(row: object) -> str:
        if not math.isfinite(float(getattr(row, "final_score", np.nan))):
            return "invalid_score"
        if bool(getattr(row, "insufficient_cross_section", True)):
            return "insufficient_cross_section"
        if not bool(getattr(row, "valid_score_input", False)):
            return "invalid_score_input"
        if bool(getattr(row, "risk_data_quality_problem", True)):
            return "risk_data_quality_problem"
        if bool(getattr(row, "quality_repaired", True)):
            return "repaired_data"
        if not bool(getattr(row, "quality_valid_ohlc", False)):
            return "invalid_ohlc"
        if not bool(getattr(row, "quality_positive_volume", False)):
            return "invalid_volume"
        if pd.isna(getattr(row, "executable_entry_date", pd.NaT)):
            return "missing_executable_entry_date"
        if not _finite_positive(getattr(row, "executable_entry_signal_open", np.nan)):
            return "invalid_executable_entry_open"
        return ""

    @staticmethod
    def _buy_sort_key(pending: PendingOrder) -> tuple[float, float, float, str]:
        record = pending.record
        return (
            -float(record["final_score"]),
            float(record["cross_section_rank"]),
            -float(record.get("risk_quality_score", 0.0)),
            str(record["symbol"]),
        )

    @staticmethod
    def _reject_order(
        record: dict[str, object], reason: str, status: str = "rejected"
    ) -> None:
        record["order_status"] = status
        record["rejection_reason"] = reason

    @staticmethod
    def _fill_order(
        record: dict[str, object],
        actual_date: pd.Timestamp,
        price: float,
        quantity: int,
        gross: float,
        cost: float,
        cash_before: float,
        cash_after: float,
        date_to_index: Mapping[pd.Timestamp, int],
    ) -> None:
        scheduled = pd.Timestamp(record["scheduled_execution_date"])
        record.update(
            {
                "order_status": "filled",
                "actual_execution_date": actual_date,
                "quantity": int(quantity),
                "raw_execution_price": price,
                "gross_transaction_value": gross,
                "transaction_cost": cost,
                "cash_before": cash_before,
                "cash_after": cash_after,
                "rejection_reason": "",
                "execution_delay_sessions": date_to_index[actual_date]
                - date_to_index[scheduled],
                "execution_delay_calendar_days": (actual_date - scheduled).days,
            }
        )

    def _apply_split(
        self, position: Position, row: Mapping[str, object]
    ) -> tuple[float, float, float]:
        factor = float(row.get("stock_split", 0.0) or 0.0)
        date = _normalize_date(row["date"])
        if not math.isfinite(factor) or factor <= 0 or date in position.split_dates:
            return 0.0, 0.0, 0.0
        position.split_dates.add(date)
        position.price_basis = causal_split_price_basis(
            position.last_observed_close,
            float(row["execution_open"]),
            factor,
        )
        if position.price_basis != PRICE_BASIS_UNADJUSTED:
            return 0.0, 0.0, factor

        exact_quantity = position.quantity * factor
        whole_quantity = math.floor(exact_quantity + 1e-12)
        fractional_quantity = exact_quantity - whole_quantity
        allocated_basis = (
            position.remaining_cost_basis * fractional_quantity / exact_quantity
            if exact_quantity > 0
            else 0.0
        )
        cash_in_lieu = fractional_quantity * float(row["execution_open"])
        realized = cash_in_lieu - allocated_basis
        position.quantity = int(whole_quantity)
        position.remaining_cost_basis -= allocated_basis
        position.split_cash_in_lieu += cash_in_lieu
        return cash_in_lieu, realized, factor

    @staticmethod
    def _credit_dividend(
        position: Position, row: Mapping[str, object], date: pd.Timestamp
    ) -> float:
        per_share = float(row.get("dividend", 0.0) or 0.0)
        if (
            not math.isfinite(per_share)
            or per_share <= 0
            or date in position.dividend_dates
        ):
            return 0.0
        cash = position.quantity * per_share
        position.dividend_dates.add(date)
        position.dividends_received += cash
        return cash

    @staticmethod
    def _update_excursion(position: Position, high: float, low: float) -> None:
        if position.original_entry_value <= 0:
            return
        high_return = (
            position.quantity * high + position.split_cash_in_lieu
        ) / position.original_entry_value - 1.0
        low_return = (
            position.quantity * low + position.split_cash_in_lieu
        ) / position.original_entry_value - 1.0
        position.mfe = max(position.mfe, high_return)
        position.mae = min(position.mae, low_return)

    @staticmethod
    def _completed_trade(
        position: Position,
        exit_order: Mapping[str, object],
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_quantity: int,
        exit_cost: float,
        exit_session_index: int,
    ) -> dict[str, object]:
        exit_value = exit_quantity * exit_price
        gross_pnl = (
            exit_value + position.split_cash_in_lieu - position.original_entry_value
        )
        net_pnl = (
            gross_pnl
            + position.dividends_received
            - position.entry_costs
            - exit_cost
        )
        invested = position.original_entry_value + position.entry_costs
        return {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "entry_signal_date": position.entry_signal_date,
            "entry_execution_date": position.entry_execution_date,
            "exit_signal_date": pd.Timestamp(exit_order["signal_date"]),
            "exit_execution_date": exit_date,
            "entry_signal_type": position.entry_signal_type,
            "exit_reason": str(exit_order["primary_exit_reason"]),
            "all_exit_reasons": str(exit_order["all_exit_reasons"]),
            "entry_score": position.entry_score,
            "entry_rank": position.entry_rank,
            "entry_score_state": position.entry_score_state,
            "sector_fallback_at_entry": position.sector_fallback_at_entry,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "quantity": position.entry_quantity,
            "exit_quantity": exit_quantity,
            "dividends_received": position.dividends_received,
            "split_cash_in_lieu": position.split_cash_in_lieu,
            "entry_costs": position.entry_costs,
            "exit_costs": exit_cost,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "net_return": net_pnl / invested if invested else 0.0,
            "holding_sessions": max(exit_session_index - position.entry_session_index, 0),
            "holding_calendar_days": (exit_date - position.entry_execution_date).days,
            "mfe": position.mfe,
            "mae": position.mae,
        }

    @staticmethod
    def _holdings_value_at_open(
        positions: Mapping[str, Position],
        feature_lookup: Mapping[
            str, Mapping[pd.Timestamp, dict[str, object]]
        ],
        date: pd.Timestamp,
    ) -> float:
        value = 0.0
        for symbol, position in positions.items():
            row = PortfolioBacktestEngine._feature_row(feature_lookup, symbol, date)
            price = (
                float(row["execution_open"])
                if row is not None and bool(row["is_valid_stock_session"])
                else position.last_observed_close
            )
            value += position.quantity * price
        return value

    def _finalize_daily(self, daily: pd.DataFrame) -> pd.DataFrame:
        if daily.empty:
            return pd.DataFrame(columns=DAILY_COLUMNS)
        finalized = daily.copy()
        finalized["daily_return"] = finalized["total_portfolio_equity"].pct_change(
            fill_method=None
        ).fillna(
            finalized["total_portfolio_equity"].iloc[0]
            / self.config.starting_capital
            - 1.0
        )
        finalized["cumulative_return"] = (
            finalized["total_portfolio_equity"] / self.config.starting_capital - 1.0
        )
        peak = finalized["total_portfolio_equity"].cummax()
        finalized["drawdown"] = finalized["total_portfolio_equity"] / peak - 1.0
        finalized["benchmark_value"] = np.nan
        finalized["benchmark_relative_equity"] = np.nan
        return finalized.reindex(columns=DAILY_COLUMNS)

    @staticmethod
    def _open_positions_frame(positions: Mapping[str, Position]) -> pd.DataFrame:
        rows = []
        for position in positions.values():
            market_value = position.quantity * position.last_observed_close
            rows.append(
                {
                    "position_id": position.position_id,
                    "symbol": position.symbol,
                    "entry_signal_date": position.entry_signal_date,
                    "entry_execution_date": position.entry_execution_date,
                    "entry_signal_type": position.entry_signal_type,
                    "entry_score": position.entry_score,
                    "entry_rank": position.entry_rank,
                    "entry_price": position.entry_price,
                    "entry_quantity": position.entry_quantity,
                    "current_quantity": position.quantity,
                    "remaining_cost_basis": position.remaining_cost_basis,
                    "dividends_received": position.dividends_received,
                    "split_cash_in_lieu": position.split_cash_in_lieu,
                    "entry_costs": position.entry_costs,
                    "last_observed_close": position.last_observed_close,
                    "market_value": market_value,
                    "unrealized_pnl": market_value - position.remaining_cost_basis,
                    "mfe": position.mfe,
                    "mae": position.mae,
                    "pending_exit_order_id": position.pending_exit_order_id,
                    "open_at_end": True,
                }
            )
        return pd.DataFrame(rows).reindex(columns=OPEN_POSITION_COLUMNS)

    def _validate_result(
        self,
        daily: pd.DataFrame,
        orders: pd.DataFrame,
        trades: pd.DataFrame,
        open_positions: pd.DataFrame,
        audit: pd.DataFrame,
    ) -> None:
        if (daily["cash"] < -1e-8).any():
            raise AssertionError("Negative cash found in portfolio output")
        if daily["number_open_positions"].gt(self.config.max_positions).any():
            raise AssertionError("Maximum position count exceeded")
        if daily["gross_exposure"].gt(1.0 + 1e-8).any():
            raise AssertionError("Leverage found in portfolio output")
        if not np.isfinite(daily["total_portfolio_equity"].to_numpy(dtype="float64")).all():
            raise AssertionError("Non-finite portfolio equity found")
        if daily["reconciliation_error"].abs().max() > 1e-5:
            raise AssertionError("Portfolio accounting reconciliation failed")
        if orders["order_id"].duplicated().any():
            raise AssertionError("Duplicate order IDs found")
        filled = orders.loc[orders["order_status"].eq("filled")]
        if filled["actual_execution_date"].le(filled["signal_date"]).any():
            raise AssertionError("Same-day or pre-signal execution found")
        if not filled["quantity"].map(lambda value: int(value) == value and value >= 0).all():
            raise AssertionError("Non-integer or negative order quantity found")
        if not open_positions.empty and open_positions["symbol"].duplicated().any():
            raise AssertionError("Duplicate open positions found")
        if not audit.empty and audit.duplicated(["symbol", "date"]).any():
            raise AssertionError("Duplicate position audit rows found")
        if not trades.empty and trades["position_id"].duplicated().any():
            raise AssertionError("Duplicate completed positions found")
