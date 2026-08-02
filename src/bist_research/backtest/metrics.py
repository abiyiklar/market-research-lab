from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .models import BacktestConfig


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    daily_equity: pd.DataFrame
    total_commission: float = 0.0
    total_slippage: float = 0.0


def calculate_drawdowns(daily_equity: pd.DataFrame) -> pd.DataFrame:
    drawdowns = daily_equity.loc[:, ["date", "total_equity"]].copy()
    drawdowns["running_peak"] = drawdowns["total_equity"].cummax()
    drawdowns["drawdown"] = drawdowns["total_equity"] / drawdowns["running_peak"] - 1

    peak_date: pd.Timestamp | None = None
    peak_value = -math.inf
    peak_dates: list[pd.Timestamp] = []
    for date, equity in zip(drawdowns["date"], drawdowns["total_equity"], strict=True):
        if float(equity) >= peak_value:
            peak_value = float(equity)
            peak_date = pd.Timestamp(date)
        peak_dates.append(pd.Timestamp(peak_date))
    drawdowns["drawdown_start"] = peak_dates
    return drawdowns


def _maximum_drawdown_details(daily_equity: pd.DataFrame) -> tuple[float, str, str]:
    drawdowns = calculate_drawdowns(daily_equity)
    trough_index = drawdowns["drawdown"].idxmin()
    trough = drawdowns.loc[trough_index]
    return (
        float(trough["drawdown"]),
        pd.Timestamp(trough["drawdown_start"]).date().isoformat(),
        pd.Timestamp(trough["date"]).date().isoformat(),
    )


def _longest_streak(outcomes: pd.Series, target: bool) -> int:
    longest = 0
    current = 0
    for outcome in outcomes:
        if bool(outcome) is target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-15:
        return 0.0
    value = numerator / denominator
    return float(value) if math.isfinite(value) else 0.0


def calculate_metrics(
    daily_equity: pd.DataFrame,
    trades: pd.DataFrame,
    config: BacktestConfig,
) -> dict[str, object]:
    if daily_equity.empty:
        raise ValueError("Cannot calculate metrics from an empty equity curve")

    start_capital = config.initial_capital
    end_capital = float(daily_equity["total_equity"].iloc[-1])
    total_return = end_capital / start_capital - 1
    start_date = pd.Timestamp(daily_equity["date"].iloc[0])
    end_date = pd.Timestamp(daily_equity["date"].iloc[-1])
    elapsed_years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    cagr = (end_capital / start_capital) ** (1 / elapsed_years) - 1

    returns = daily_equity["daily_return"].astype(float).iloc[1:]
    daily_risk_free = (1 + config.risk_free_rate) ** (1 / config.trading_days_per_year) - 1
    excess_returns = returns - daily_risk_free
    return_std = float(excess_returns.std(ddof=1)) if len(excess_returns) > 1 else 0.0
    sharpe = _safe_ratio(
        float(excess_returns.mean()) * math.sqrt(config.trading_days_per_year),
        return_std,
    )
    downside = excess_returns.loc[excess_returns < 0]
    downside_deviation = (
        math.sqrt(float((downside**2).mean())) if not downside.empty else 0.0
    )
    sortino = _safe_ratio(
        float(excess_returns.mean()) * math.sqrt(config.trading_days_per_year),
        downside_deviation,
    )
    annual_volatility = (
        float(returns.std(ddof=1)) * math.sqrt(config.trading_days_per_year)
        if len(returns) > 1
        else 0.0
    )
    max_drawdown, drawdown_start, drawdown_end = _maximum_drawdown_details(daily_equity)
    calmar = _safe_ratio(cagr, abs(max_drawdown))

    trade_returns = (
        trades["return_pct"].astype(float) if not trades.empty else pd.Series(dtype="float64")
    )
    net_pnl = trades["net_pnl"].astype(float) if not trades.empty else pd.Series(dtype="float64")
    wins = trade_returns > 0
    losses = trade_returns < 0
    winning_returns = trade_returns.loc[wins]
    losing_returns = trade_returns.loc[losses]
    gross_profit = float(net_pnl.loc[net_pnl > 0].sum())
    gross_loss = abs(float(net_pnl.loc[net_pnl < 0].sum()))
    average_win = float(winning_returns.mean()) if not winning_returns.empty else 0.0
    average_loss = float(losing_returns.mean()) if not losing_returns.empty else 0.0
    win_rate = float(wins.mean()) if len(trade_returns) else 0.0
    loss_rate = float(losses.mean()) if len(trade_returns) else 0.0
    total_holding_days = int(trades["holding_days"].sum()) if not trades.empty else 0

    return {
        "start_date": start_date.date().isoformat(),
        "end_date": end_date.date().isoformat(),
        "initial_capital": start_capital,
        "ending_capital": end_capital,
        "net_profit": end_capital - start_capital,
        "total_return": total_return,
        "cagr": cagr,
        "maximum_drawdown": max_drawdown,
        "maximum_drawdown_start_date": drawdown_start,
        "maximum_drawdown_end_date": drawdown_end,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "annual_volatility": annual_volatility,
        "total_trades": int(len(trades)),
        "winning_trades": int(wins.sum()),
        "losing_trades": int(losses.sum()),
        "win_rate": win_rate,
        "average_trade_return": float(trade_returns.mean()) if len(trade_returns) else 0.0,
        "average_win": average_win,
        "average_loss": average_loss,
        "profit_factor": _safe_ratio(gross_profit, gross_loss),
        "payoff_ratio": _safe_ratio(average_win, abs(average_loss)),
        "expected_trade_return": win_rate * average_win + loss_rate * average_loss,
        "largest_win": float(trade_returns.max()) if len(trade_returns) else 0.0,
        "largest_loss": float(trade_returns.min()) if len(trade_returns) else 0.0,
        "average_holding_days": (
            float(trades["holding_days"].mean()) if not trades.empty else 0.0
        ),
        "maximum_holding_days": int(trades["holding_days"].max()) if not trades.empty else 0,
        "time_in_market": min(total_holding_days / len(daily_equity), 1.0),
        "total_commission": (
            float((trades["entry_commission"] + trades["exit_commission"]).sum())
            if not trades.empty
            else 0.0
        ),
        "total_slippage_cost": float(trades["slippage_cost"].sum()) if not trades.empty else 0.0,
        "maximum_consecutive_wins": _longest_streak(wins, True),
        "maximum_consecutive_losses": _longest_streak(losses, True),
    }


def _benchmark_equity_curve(dates: pd.Series, values: list[float], initial_capital: float) -> pd.DataFrame:
    equity = pd.DataFrame({"date": pd.to_datetime(dates).reset_index(drop=True), "total_equity": values})
    equity["daily_return"] = equity["total_equity"].pct_change(fill_method=None).fillna(0.0)
    equity["cumulative_return"] = equity["total_equity"] / initial_capital - 1
    equity["running_peak"] = equity["total_equity"].cummax()
    equity["drawdown"] = equity["total_equity"] / equity["running_peak"] - 1
    return equity


def calculate_tuprs_buy_hold(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> BenchmarkResult:
    eligible = frame.loc[
        (~frame["is_indicator_warmup"].astype(bool))
        & (frame["tuprs_open"] > 0)
        & (frame["tuprs_close"] > 0)
        & (frame["tuprs_volume"] > 0)
    ].reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No eligible rows for the TUPRS buy-and-hold benchmark")

    raw_entry = float(eligible["tuprs_open"].iloc[0])
    effective_entry = raw_entry * (1 + config.slippage_rate)
    quantity = math.floor(
        config.initial_capital / (effective_entry * (1 + config.commission_rate))
    )
    if quantity <= 0:
        raise ValueError("Initial capital is insufficient for the TUPRS benchmark")
    entry_commission = quantity * effective_entry * config.commission_rate
    cash = config.initial_capital - quantity * effective_entry - entry_commission

    values = [cash + quantity * float(close) for close in eligible["tuprs_close"]]
    raw_exit = float(eligible["tuprs_close"].iloc[-1])
    effective_exit = raw_exit * (1 - config.slippage_rate)
    exit_commission = quantity * effective_exit * config.commission_rate
    values[-1] = cash + quantity * effective_exit - exit_commission
    slippage = quantity * (effective_entry - raw_entry) + quantity * (raw_exit - effective_exit)
    equity = _benchmark_equity_curve(eligible["date"], values, config.initial_capital)
    return BenchmarkResult(
        name="tuprs_buy_hold",
        daily_equity=equity,
        total_commission=entry_commission + exit_commission,
        total_slippage=slippage,
    )


def calculate_xu100_buy_hold(
    frame: pd.DataFrame,
    config: BacktestConfig,
) -> BenchmarkResult:
    eligible = frame.loc[
        (~frame["is_indicator_warmup"].astype(bool))
        & frame["xu100_close"].notna()
        & (frame["xu100_close"] > 0)
    ].reset_index(drop=True)
    if eligible.empty:
        raise ValueError("No eligible rows for the XU100 buy-and-hold benchmark")

    starting_index = float(eligible["xu100_close"].iloc[0])
    values = (config.initial_capital * eligible["xu100_close"] / starting_index).astype(float).tolist()
    equity = _benchmark_equity_curve(eligible["date"], values, config.initial_capital)
    return BenchmarkResult(name="xu100_buy_hold", daily_equity=equity)


def benchmark_metrics(
    benchmark: BenchmarkResult,
    config: BacktestConfig,
) -> dict[str, object]:
    metrics = calculate_metrics(benchmark.daily_equity, pd.DataFrame(), config)
    metrics["total_commission"] = benchmark.total_commission
    metrics["total_slippage_cost"] = benchmark.total_slippage
    return metrics
