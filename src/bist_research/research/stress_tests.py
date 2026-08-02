from __future__ import annotations

from dataclasses import replace
from typing import Callable

import pandas as pd

from bist_research.backtest.metrics import calculate_metrics
from bist_research.backtest.models import BacktestConfig, BacktestResult, TRADE_COLUMNS

from .experiment import Experiment
from .strategies import ResearchStrategy, build_strategy, deterministic_signal_filter, scale_parameters
from .walk_forward import _slice, run_strategy_period


def _metric_row(
    experiment: Experiment,
    scenario: str,
    metrics: dict[str, object],
    selected_window_count: int,
) -> dict[str, object]:
    return {
        "experiment_id": experiment.experiment_id,
        "strategy_name": experiment.strategy_name,
        "parameters": experiment.parameters_json,
        "scenario": scenario,
        "data_scope": "selected_oos",
        "methodology_version": "nested_walk_forward_v2",
        "selected_window_count": selected_window_count,
        "total_return": metrics["total_return"],
        "maximum_drawdown": metrics["maximum_drawdown"],
        "profit_factor": metrics["profit_factor"],
        "total_trades": metrics["total_trades"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "calmar_ratio": metrics["calmar_ratio"],
        "annual_volatility": metrics["annual_volatility"],
    }


def _stitch_results(
    results: list[BacktestResult],
    config: BacktestConfig,
) -> tuple[BacktestResult, dict[str, object]]:
    if not results:
        raise ValueError("No selected OOS results to stitch")
    return_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    for result in results:
        equity = result.daily_equity.loc[:, ["date", "daily_return"]].copy()
        if not equity.empty:
            equity.iloc[0, equity.columns.get_loc("daily_return")] = 0.0
            return_parts.append(equity)
        if not result.trades.empty:
            trade_parts.append(result.trades.copy())
    stitched = pd.concat(return_parts, ignore_index=True).sort_values("date")
    if stitched["date"].duplicated().any():
        raise ValueError("Selected OOS windows overlap chronologically")
    stitched["total_equity"] = config.initial_capital * (
        1 + stitched["daily_return"].astype(float)
    ).cumprod()
    trades = (
        pd.concat(trade_parts, ignore_index=True)
        if trade_parts
        else pd.DataFrame(columns=TRADE_COLUMNS)
    )
    result = BacktestResult(trades=trades, daily_equity=stitched, config=config)
    return result, calculate_metrics(stitched, trades, config)


def run_selected_oos_scenario(
    frame: pd.DataFrame,
    selection_rows: pd.DataFrame,
    strategy: ResearchStrategy,
    config: BacktestConfig,
    *,
    disabled_signal_dates: frozenset[pd.Timestamp] = frozenset(),
    start_offset: int = 0,
    keep_signal: Callable[[pd.Series], bool] | None = None,
) -> tuple[BacktestResult, dict[str, object]]:
    prepared = strategy.prepare(frame)
    results: list[BacktestResult] = []

    def entry_signal(row: pd.Series, active_config: BacktestConfig) -> bool:
        date = pd.Timestamp(row["date"]).normalize()
        return bool(
            date not in disabled_signal_dates
            and (keep_signal(row) if keep_signal is not None else True)
            and strategy.entry_signal(row, active_config)
        )

    ordered = selection_rows.sort_values("split_start")
    for selected in ordered.itertuples(index=False):
        segment = _slice(
            prepared,
            pd.Timestamp(selected.split_start),
            pd.Timestamp(selected.split_end),
        )
        if start_offset:
            segment = segment.iloc[start_offset:].reset_index(drop=True)
        if segment.empty:
            continue
        result, _, _, _ = run_strategy_period(
            segment,
            strategy,
            config,
            entry_signal_override=entry_signal,
        )
        results.append(result)
    return _stitch_results(results, strategy.backtest_config(config))


def _classify_stability(group: pd.DataFrame) -> tuple[str, str]:
    stressed = group.loc[~group["scenario"].eq("base")]
    positive_fraction = float(stressed["total_return"].ge(0).mean())
    median_return = float(stressed["total_return"].median())
    worst_drawdown = float(stressed["maximum_drawdown"].min())
    if positive_fraction >= 0.70 and median_return >= 0 and worst_drawdown >= -0.45:
        return "passed", "stable"
    if positive_fraction >= 0.40 and worst_drawdown >= -0.50:
        return "degraded", "fragile"
    return "failed", "unstable"


def run_stress_tests(
    frame: pd.DataFrame,
    leaderboard: pd.DataFrame,
    walk_forward: pd.DataFrame,
    experiments: dict[str, Experiment],
    base_config: BacktestConfig,
    random_seed: int,
    top_n: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in leaderboard.head(top_n).itertuples(index=False):
        experiment = experiments[candidate.experiment_id]
        selected = walk_forward.loc[
            walk_forward["experiment_id"].eq(experiment.experiment_id)
            & walk_forward["split"].eq("oos")
            & walk_forward["selected_for_oos"].astype(bool)
        ]
        if selected.empty:
            continue
        strategy = build_strategy(experiment.strategy_name, experiment.parameters)
        base_result, base_metrics = run_selected_oos_scenario(
            frame,
            selected,
            strategy,
            base_config,
        )
        count = len(selected)
        rows.append(_metric_row(experiment, "base", base_metrics, count))

        doubled_costs = replace(
            base_config,
            commission_rate=base_config.commission_rate * 2,
            slippage_rate=base_config.slippage_rate * 2,
        )
        _, metrics = run_selected_oos_scenario(frame, selected, strategy, doubled_costs)
        rows.append(_metric_row(experiment, "double_costs", metrics, count))

        for factor in (0.8, 0.9, 1.1, 1.2):
            varied = build_strategy(
                experiment.strategy_name,
                scale_parameters(experiment.parameters, factor),
            )
            _, metrics = run_selected_oos_scenario(frame, selected, varied, base_config)
            rows.append(_metric_row(experiment, f"parameters_x_{factor:.1f}", metrics, count))

        for offset in (20, 40, 60):
            _, metrics = run_selected_oos_scenario(
                frame,
                selected,
                strategy,
                base_config,
                start_offset=offset,
            )
            rows.append(_metric_row(experiment, f"start_offset_{offset}", metrics, count))

        best_signal_dates = [
            pd.Timestamp(value).normalize()
            for value in base_result.trades.sort_values("net_pnl", ascending=False)[
                "signal_date"
            ].head(3)
        ]
        for remove_count in (1, 3):
            disabled = frozenset(best_signal_dates[:remove_count])
            _, metrics = run_selected_oos_scenario(
                frame,
                selected,
                strategy,
                base_config,
                disabled_signal_dates=disabled,
            )
            rows.append(
                _metric_row(
                    experiment,
                    f"remove_best_{remove_count}_trade" + ("s" if remove_count > 1 else ""),
                    metrics,
                    count,
                )
            )

        delayed = replace(base_config, entry_delay_days=base_config.entry_delay_days + 1)
        _, metrics = run_selected_oos_scenario(frame, selected, strategy, delayed)
        rows.append(_metric_row(experiment, "extra_entry_delay_1_day", metrics, count))

        keep_signal = deterministic_signal_filter(random_seed, 0.10)
        _, metrics = run_selected_oos_scenario(
            frame,
            selected,
            strategy,
            base_config,
            keep_signal=keep_signal,
        )
        rows.append(_metric_row(experiment, "skip_10_percent_signals", metrics, count))

    results = pd.DataFrame(rows)
    if results.empty:
        return results
    classifications = {
        experiment_id: _classify_stability(group)
        for experiment_id, group in results.groupby("experiment_id")
    }
    results["stress_test_result"] = results["experiment_id"].map(
        lambda value: classifications[value][0]
    )
    results["stability_class"] = results["experiment_id"].map(
        lambda value: classifications[value][1]
    )
    return results


def stress_summary(stress_results: pd.DataFrame) -> pd.DataFrame:
    if stress_results.empty:
        return pd.DataFrame(
            columns=["experiment_id", "stress_test_result", "stability_class"]
        )
    return (
        stress_results.loc[
            :,
            ["experiment_id", "stress_test_result", "stability_class"],
        ]
        .drop_duplicates("experiment_id")
        .reset_index(drop=True)
    )
