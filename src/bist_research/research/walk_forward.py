from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from bist_research.backtest.engine import BacktestEngine
from bist_research.backtest.metrics import (
    benchmark_metrics,
    calculate_metrics,
    calculate_tuprs_adjusted_total_return,
)
from bist_research.backtest.models import BacktestConfig, BacktestResult

from .experiment import Experiment
from .strategies import ResearchStrategy, build_strategy, strategy_is_causal


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


def generate_walk_forward_windows(
    frame: pd.DataFrame,
    train_years: int = 4,
    validation_years: int = 1,
    oos_years: int = 1,
) -> list[WalkForwardWindow]:
    dates = pd.to_datetime(frame["date"])
    first_year = int(dates.min().year)
    last_year = int(dates.max().year)
    total_years = train_years + validation_years + oos_years
    windows: list[WalkForwardWindow] = []
    for start_year in range(first_year, last_year - total_years + 2):
        train_end_year = start_year + train_years - 1
        validation_start_year = train_end_year + 1
        validation_end_year = validation_start_year + validation_years - 1
        oos_start_year = validation_end_year + 1
        oos_end_year = oos_start_year + oos_years - 1
        if oos_end_year > last_year:
            continue
        windows.append(
            WalkForwardWindow(
                window_id=f"wf_{start_year}_{oos_end_year}",
                train_start=pd.Timestamp(start_year, 1, 1),
                train_end=pd.Timestamp(train_end_year, 12, 31),
                validation_start=pd.Timestamp(validation_start_year, 1, 1),
                validation_end=pd.Timestamp(validation_end_year, 12, 31),
                oos_start=pd.Timestamp(oos_start_year, 1, 1),
                oos_end=pd.Timestamp(oos_end_year, 12, 31),
            )
        )
    return windows


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    return frame.loc[dates.between(start, end)].reset_index(drop=True)


def _result_quality(result: BacktestResult, metrics: dict[str, object]) -> bool:
    equity = result.daily_equity
    if equity.empty or equity["date"].duplicated().any():
        return False
    numeric_equity = equity.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    numeric_metrics = [
        float(value)
        for value in metrics.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    return bool(
        np.isfinite(numeric_equity).all()
        and all(math.isfinite(value) for value in numeric_metrics)
    )


def run_strategy_period(
    prepared_frame: pd.DataFrame,
    strategy: ResearchStrategy,
    base_config: BacktestConfig,
    *,
    entry_signal_override: Callable[[pd.Series, BacktestConfig], bool] | None = None,
) -> tuple[BacktestResult, dict[str, object], dict[str, object], bool]:
    config = strategy.backtest_config(base_config)
    entry_signal = entry_signal_override or strategy.entry_signal
    result = BacktestEngine(
        config=config,
        entry_signal=entry_signal,
        exit_signal=strategy.exit_signal,
    ).run(prepared_frame)
    metrics = calculate_metrics(result.daily_equity, result.trades, config)
    benchmark = calculate_tuprs_adjusted_total_return(prepared_frame, config)
    adjusted_metrics = benchmark_metrics(benchmark, config)
    metrics["benchmark_excess_return"] = float(metrics["total_return"]) - float(
        adjusted_metrics["total_return"]
    )
    return result, metrics, adjusted_metrics, _result_quality(result, metrics)


def run_full_experiment(
    frame: pd.DataFrame,
    experiment: Experiment,
    base_config: BacktestConfig,
) -> tuple[dict[str, object], BacktestResult]:
    strategy = build_strategy(experiment.strategy_name, experiment.parameters)
    prepared = strategy.prepare(frame)
    result, metrics, benchmark, quality = run_strategy_period(prepared, strategy, base_config)
    config = strategy.backtest_config(base_config)
    row = {
        **experiment.to_record(),
        "config": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
        "data_quality_pass": quality,
        "lookahead_pass": strategy_is_causal(strategy, frame),
        "benchmark_total_return": benchmark["total_return"],
        **metrics,
    }
    return row, result


def run_walk_forward_experiment(
    frame: pd.DataFrame,
    experiment: Experiment,
    base_config: BacktestConfig,
    windows: list[WalkForwardWindow] | None = None,
) -> list[dict[str, object]]:
    active_windows = windows or generate_walk_forward_windows(frame)
    strategy = build_strategy(experiment.strategy_name, experiment.parameters)
    prepared = strategy.prepare(frame)
    config_json = json.dumps(
        asdict(strategy.backtest_config(base_config)),
        sort_keys=True,
        separators=(",", ":"),
    )
    lookahead_pass = strategy_is_causal(strategy, frame)
    rows: list[dict[str, object]] = []
    for window in active_windows:
        splits = (
            ("train", window.train_start, window.train_end),
            ("validation", window.validation_start, window.validation_end),
            ("oos", window.oos_start, window.oos_end),
        )
        for split_name, start, end in splits:
            split_frame = _slice(prepared, start, end)
            if split_frame.empty:
                continue
            result, metrics, benchmark, quality = run_strategy_period(
                split_frame,
                strategy,
                base_config,
            )
            rows.append(
                {
                    **experiment.to_record(),
                    "config": config_json,
                    "window_id": window.window_id,
                    "split": split_name,
                    "split_start": start.date().isoformat(),
                    "split_end": end.date().isoformat(),
                    "data_quality_pass": quality,
                    "lookahead_pass": lookahead_pass,
                    "benchmark_total_return": benchmark["total_return"],
                    "total_trades": metrics["total_trades"],
                    "total_return": metrics["total_return"],
                    "cagr": metrics["cagr"],
                    "maximum_drawdown": metrics["maximum_drawdown"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "sortino_ratio": metrics["sortino_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "win_rate": metrics["win_rate"],
                    "time_in_market": metrics["time_in_market"],
                    "benchmark_excess_return": metrics["benchmark_excess_return"],
                }
            )
    return rows
