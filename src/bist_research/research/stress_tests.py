from __future__ import annotations

from dataclasses import replace

import pandas as pd

from bist_research.backtest.models import BacktestConfig

from .experiment import Experiment
from .strategies import (
    build_strategy,
    deterministic_signal_filter,
    scale_parameters,
)
from .walk_forward import run_strategy_period


def _metric_row(
    experiment: Experiment,
    scenario: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment_id": experiment.experiment_id,
        "strategy_name": experiment.strategy_name,
        "parameters": experiment.parameters_json,
        "scenario": scenario,
        "total_return": metrics["total_return"],
        "maximum_drawdown": metrics["maximum_drawdown"],
        "profit_factor": metrics["profit_factor"],
        "total_trades": metrics["total_trades"],
        "sharpe_ratio": metrics["sharpe_ratio"],
    }


def _without_best_trade_metrics(
    base_metrics: dict[str, object],
    trades: pd.DataFrame,
    initial_capital: float,
    count: int,
) -> dict[str, object]:
    reduced = trades.sort_values("net_pnl", ascending=False).iloc[count:]
    gains = float(reduced.loc[reduced["net_pnl"] > 0, "net_pnl"].sum())
    losses = abs(float(reduced.loc[reduced["net_pnl"] < 0, "net_pnl"].sum()))
    metrics = dict(base_metrics)
    metrics["total_return"] = float(reduced["net_pnl"].sum()) / initial_capital
    metrics["profit_factor"] = gains / losses if losses > 0 else 0.0
    metrics["total_trades"] = int(len(reduced))
    return metrics


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
    experiments: dict[str, Experiment],
    base_config: BacktestConfig,
    random_seed: int,
    top_n: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in leaderboard.head(top_n).itertuples(index=False):
        experiment = experiments[candidate.experiment_id]
        strategy = build_strategy(experiment.strategy_name, experiment.parameters)
        prepared = strategy.prepare(frame)
        base_result, base_metrics, _, _ = run_strategy_period(
            prepared, strategy, base_config
        )
        rows.append(_metric_row(experiment, "base", base_metrics))

        doubled_costs = replace(
            base_config,
            commission_rate=base_config.commission_rate * 2,
            slippage_rate=base_config.slippage_rate * 2,
        )
        _, metrics, _, _ = run_strategy_period(prepared, strategy, doubled_costs)
        rows.append(_metric_row(experiment, "double_costs", metrics))

        for factor in (0.8, 0.9, 1.1, 1.2):
            varied = build_strategy(
                experiment.strategy_name,
                scale_parameters(experiment.parameters, factor),
            )
            varied_frame = varied.prepare(frame)
            _, metrics, _, _ = run_strategy_period(varied_frame, varied, base_config)
            rows.append(_metric_row(experiment, f"parameters_x_{factor:.1f}", metrics))

        for offset in (20, 40, 60):
            _, metrics, _, _ = run_strategy_period(
                prepared.iloc[offset:].reset_index(drop=True), strategy, base_config
            )
            rows.append(_metric_row(experiment, f"start_offset_{offset}", metrics))

        rows.append(
            _metric_row(
                experiment,
                "remove_best_1_trade",
                _without_best_trade_metrics(
                    base_metrics, base_result.trades, base_config.initial_capital, 1
                ),
            )
        )
        rows.append(
            _metric_row(
                experiment,
                "remove_best_3_trades",
                _without_best_trade_metrics(
                    base_metrics, base_result.trades, base_config.initial_capital, 3
                ),
            )
        )

        delayed = replace(base_config, entry_delay_days=base_config.entry_delay_days + 1)
        _, metrics, _, _ = run_strategy_period(prepared, strategy, delayed)
        rows.append(_metric_row(experiment, "extra_entry_delay_1_day", metrics))

        keep_signal = deterministic_signal_filter(random_seed, 0.10)

        def skipped_entry(
            row: pd.Series,
            config: BacktestConfig,
            active_strategy=strategy,
        ) -> bool:
            return active_strategy.entry_signal(row, config) and keep_signal(row)

        _, metrics, _, _ = run_strategy_period(
            prepared,
            strategy,
            base_config,
            entry_signal_override=skipped_entry,
        )
        rows.append(_metric_row(experiment, "skip_10_percent_signals", metrics))

    results = pd.DataFrame(rows)
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
    return (
        stress_results.loc[
            :,
            ["experiment_id", "stress_test_result", "stability_class"],
        ]
        .drop_duplicates("experiment_id")
        .reset_index(drop=True)
    )
