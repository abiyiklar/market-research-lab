from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
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


MAX_SELECTED_PER_WINDOW = 5
MINIMUM_TRAIN_TRADES = 3
MINIMUM_TRAIN_DRAWDOWN = -0.60


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
    strategy_return = float(metrics["total_return"])
    benchmark_return = float(adjusted_metrics["total_return"])
    metrics["benchmark_excess_return"] = strategy_return - benchmark_return
    metrics["absolute_return_difference"] = strategy_return - benchmark_return
    metrics["sharpe_difference"] = float(metrics["sharpe_ratio"]) - float(
        adjusted_metrics["sharpe_ratio"]
    )
    metrics["calmar_difference"] = float(metrics["calmar_ratio"]) - float(
        adjusted_metrics["calmar_ratio"]
    )
    metrics["maximum_drawdown_difference"] = float(metrics["maximum_drawdown"]) - float(
        adjusted_metrics["maximum_drawdown"]
    )
    exposure = float(metrics["time_in_market"])
    metrics["exposure_adjusted_return"] = strategy_return / exposure if exposure > 0 else 0.0
    benchmark_returns = benchmark.daily_equity["daily_return"].astype(float).iloc[1:]
    exposure_matched_returns = (benchmark_returns * exposure).clip(lower=-0.999999)
    metrics["exposure_matched_benchmark_return"] = float(
        (1 + exposure_matched_returns).prod() - 1
    )
    benchmark_volatility = float(adjusted_metrics["annual_volatility"])
    volatility_scale = (
        float(metrics["annual_volatility"]) / benchmark_volatility
        if benchmark_volatility > 0
        else 0.0
    )
    scaled_returns = (benchmark_returns * volatility_scale).clip(lower=-0.999999)
    volatility_matched_benchmark_return = float((1 + scaled_returns).prod() - 1)
    metrics["volatility_matched_benchmark_return"] = volatility_matched_benchmark_return
    metrics["volatility_matched_excess_return"] = (
        strategy_return - volatility_matched_benchmark_return
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


WALK_FORWARD_METRICS = (
    "total_trades",
    "total_return",
    "cagr",
    "maximum_drawdown",
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "annual_volatility",
    "profit_factor",
    "win_rate",
    "time_in_market",
    "benchmark_excess_return",
    "absolute_return_difference",
    "volatility_matched_benchmark_return",
    "volatility_matched_excess_return",
    "sharpe_difference",
    "calmar_difference",
    "maximum_drawdown_difference",
    "exposure_adjusted_return",
    "exposure_matched_benchmark_return",
)


def _walk_forward_row(
    experiment: Experiment,
    strategy: ResearchStrategy,
    base_config: BacktestConfig,
    window: WalkForwardWindow,
    split_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    metrics: dict[str, object],
    benchmark: dict[str, object],
    quality: bool,
    lookahead_pass: bool,
) -> dict[str, object]:
    return {
        **experiment.to_record(),
        "config": json.dumps(
            asdict(strategy.backtest_config(base_config)),
            sort_keys=True,
            separators=(",", ":"),
        ),
        "window_id": window.window_id,
        "methodology_version": "nested_walk_forward_v2",
        "split": split_name,
        "split_start": start.date().isoformat(),
        "split_end": end.date().isoformat(),
        "data_quality_pass": quality,
        "lookahead_pass": lookahead_pass,
        "benchmark_total_return": benchmark["total_return"],
        "benchmark_sharpe_ratio": benchmark["sharpe_ratio"],
        "benchmark_calmar_ratio": benchmark["calmar_ratio"],
        "benchmark_maximum_drawdown": benchmark["maximum_drawdown"],
        "selected_for_oos": False,
        "selected_on_validation": False,
        "selection_rank": 0,
        "family_selection_rank": 0,
        "selection_score": -1.0e12,
        "oos_run_count": 0,
        "train_quality_pass": False,
        **{key: metrics[key] for key in WALK_FORWARD_METRICS},
    }


def train_quality_pass(metrics: dict[str, object], quality: bool, causal: bool) -> bool:
    return bool(
        quality
        and causal
        and int(metrics["total_trades"]) >= MINIMUM_TRAIN_TRADES
        and float(metrics["maximum_drawdown"]) >= MINIMUM_TRAIN_DRAWDOWN
        and all(
            math.isfinite(float(metrics[key]))
            for key in (
                "total_return",
                "sharpe_ratio",
                "profit_factor",
                "maximum_drawdown",
            )
        )
    )


def validation_selection_score(metrics: dict[str, object]) -> float:
    """Rank frozen candidates using validation evidence only."""
    return float(
        float(metrics["total_return"]) * 35
        + float(metrics["volatility_matched_excess_return"]) * 25
        + float(metrics["sharpe_ratio"]) * 2
        + float(metrics["calmar_ratio"])
        + min(float(metrics["profit_factor"]), 4.0)
        + float(metrics["maximum_drawdown"]) * 12
        + min(int(metrics["total_trades"]) / 5, 2.0)
    )


def _selected_experiment_ids(selection: pd.DataFrame, maximum: int) -> list[str]:
    if selection.empty:
        return []
    ordered = selection.sort_values(
        ["selection_rank", "family_selection_rank", "experiment_id"]
    )
    selected: list[str] = []
    for row in ordered.loc[ordered["family_selection_rank"].eq(1)].itertuples(index=False):
        if row.experiment_id not in selected:
            selected.append(row.experiment_id)
        if len(selected) >= maximum:
            return selected
    for experiment_id in ordered["experiment_id"]:
        if experiment_id not in selected:
            selected.append(experiment_id)
        if len(selected) >= maximum:
            break
    return selected


def run_nested_walk_forward(
    frame: pd.DataFrame,
    experiments: Sequence[Experiment],
    base_config: BacktestConfig,
    windows: list[WalkForwardWindow] | None = None,
    maximum_selected_per_window: int = MAX_SELECTED_PER_WINDOW,
    causality_results: dict[str, bool] | None = None,
) -> list[dict[str, object]]:
    """Select on train/validation, then run each frozen selection once on OOS."""
    if maximum_selected_per_window <= 0:
        raise ValueError("maximum_selected_per_window must be positive")
    active_windows = windows or generate_walk_forward_windows(frame)
    strategies = {
        experiment.experiment_id: build_strategy(
            experiment.strategy_name,
            experiment.parameters,
        )
        for experiment in experiments
    }
    prepared_frames = {
        experiment_id: strategy.prepare(frame)
        for experiment_id, strategy in strategies.items()
    }
    causality = causality_results or {
        experiment_id: strategy_is_causal(strategy, frame)
        for experiment_id, strategy in strategies.items()
    }
    missing_causality = sorted(set(strategies).difference(causality))
    if missing_causality:
        raise ValueError(
            "Missing causality results: " + ", ".join(missing_causality)
        )
    experiment_map = {experiment.experiment_id: experiment for experiment in experiments}
    rows: list[dict[str, object]] = []

    for window in active_windows:
        window_rows: dict[str, list[dict[str, object]]] = {}
        validation_metrics: dict[str, dict[str, object]] = {}
        selection_records: list[dict[str, object]] = []
        for experiment in experiments:
            experiment_id = experiment.experiment_id
            strategy = strategies[experiment_id]
            candidate_rows: list[dict[str, object]] = []
            train_pass = False
            for split_name, start, end in (
                ("train", window.train_start, window.train_end),
                ("validation", window.validation_start, window.validation_end),
            ):
                split_frame = _slice(prepared_frames[experiment_id], start, end)
                if split_frame.empty:
                    continue
                _, metrics, benchmark, quality = run_strategy_period(
                    split_frame,
                    strategy,
                    base_config,
                )
                if split_name == "train":
                    train_pass = train_quality_pass(
                        metrics,
                        quality,
                        causality[experiment_id],
                    )
                else:
                    validation_metrics[experiment_id] = metrics
                candidate_rows.append(
                    _walk_forward_row(
                        experiment,
                        strategy,
                        base_config,
                        window,
                        split_name,
                        start,
                        end,
                        metrics,
                        benchmark,
                        quality,
                        causality[experiment_id],
                    )
                )
            for row in candidate_rows:
                row["train_quality_pass"] = train_pass
            window_rows[experiment_id] = candidate_rows
            if train_pass and experiment_id in validation_metrics:
                selection_records.append(
                    {
                        "experiment_id": experiment_id,
                        "strategy_name": experiment.strategy_name,
                        "selection_score": validation_selection_score(
                            validation_metrics[experiment_id]
                        ),
                    }
                )

        selection = pd.DataFrame(selection_records)
        if not selection.empty:
            selection = selection.sort_values(
                ["selection_score", "experiment_id"],
                ascending=[False, True],
            ).reset_index(drop=True)
            selection["selection_rank"] = selection.index + 1
            selection["family_selection_rank"] = (
                selection.groupby("strategy_name")["selection_score"]
                .rank(method="first", ascending=False)
                .astype(int)
            )
        selected_ids = _selected_experiment_ids(selection, maximum_selected_per_window)
        selection_by_id = (
            selection.set_index("experiment_id").to_dict("index")
            if not selection.empty
            else {}
        )

        for experiment_id, candidate_rows in window_rows.items():
            selection_values = selection_by_id.get(experiment_id)
            selected = experiment_id in selected_ids
            for row in candidate_rows:
                if selection_values is not None:
                    row["selection_score"] = selection_values["selection_score"]
                    row["selection_rank"] = selection_values["selection_rank"]
                    row["family_selection_rank"] = selection_values[
                        "family_selection_rank"
                    ]
                row["selected_for_oos"] = selected
                row["selected_on_validation"] = selected
                rows.append(row)

        for experiment_id in selected_ids:
            experiment = experiment_map[experiment_id]
            strategy = strategies[experiment_id]
            split_frame = _slice(
                prepared_frames[experiment_id],
                window.oos_start,
                window.oos_end,
            )
            if split_frame.empty:
                continue
            _, metrics, benchmark, quality = run_strategy_period(
                split_frame,
                strategy,
                base_config,
            )
            row = _walk_forward_row(
                experiment,
                strategy,
                base_config,
                window,
                "oos",
                window.oos_start,
                window.oos_end,
                metrics,
                benchmark,
                quality,
                causality[experiment_id],
            )
            selection_values = selection_by_id[experiment_id]
            row.update(
                {
                    "selected_for_oos": True,
                    "selected_on_validation": True,
                    "selection_score": selection_values["selection_score"],
                    "selection_rank": selection_values["selection_rank"],
                    "family_selection_rank": selection_values["family_selection_rank"],
                    "oos_run_count": 1,
                    "train_quality_pass": True,
                }
            )
            rows.append(row)
    return rows
