from __future__ import annotations

import pandas as pd

from bist_research.backtest.models import BacktestConfig

from .experiment import Experiment
from .strategies import FILTER_NAMES, build_strategy
from .walk_forward import _slice, run_strategy_period


def run_ablation_analysis(
    frame: pd.DataFrame,
    leaderboard: pd.DataFrame,
    walk_forward: pd.DataFrame,
    experiments: dict[str, Experiment],
    base_config: BacktestConfig,
    top_n: int = 10,
) -> pd.DataFrame:
    """Run diagnostic ablations on train/validation only, never selected OOS."""
    rows: list[dict[str, object]] = []
    for candidate in leaderboard.head(top_n).itertuples(index=False):
        experiment = experiments[candidate.experiment_id]
        strategy = build_strategy(experiment.strategy_name, experiment.parameters)
        selected_windows = walk_forward.loc[
            walk_forward["experiment_id"].eq(experiment.experiment_id)
            & walk_forward["split"].eq("oos")
            & walk_forward["selected_for_oos"].astype(bool),
            "window_id",
        ].unique()
        for window_id in selected_windows:
            audit_splits = walk_forward.loc[
                walk_forward["experiment_id"].eq(experiment.experiment_id)
                & walk_forward["window_id"].eq(window_id)
                & walk_forward["split"].isin(["train", "validation"])
            ]
            for audit_split in audit_splits.itertuples(index=False):
                segment = _slice(
                    frame,
                    pd.Timestamp(audit_split.split_start),
                    pd.Timestamp(audit_split.split_end),
                )
                prepared = strategy.prepare(segment)
                _, base_metrics, _, _ = run_strategy_period(prepared, strategy, base_config)
                for filter_name in FILTER_NAMES:
                    ablated = strategy.without_filter(filter_name)
                    ablated_frame = ablated.prepare(segment)
                    _, metrics, _, _ = run_strategy_period(
                        ablated_frame,
                        ablated,
                        base_config,
                    )
                    rows.append(
                        {
                            "experiment_id": experiment.experiment_id,
                            "strategy_name": experiment.strategy_name,
                            "parameters": experiment.parameters_json,
                            "window_id": window_id,
                            "data_scope": audit_split.split,
                            "methodology_version": "nested_walk_forward_v2",
                            "diagnostic_only": True,
                            "used_for_selection": False,
                            "removed_filter": filter_name,
                            "base_total_trades": base_metrics["total_trades"],
                            "ablated_total_trades": metrics["total_trades"],
                            "trade_count_impact": int(metrics["total_trades"])
                            - int(base_metrics["total_trades"]),
                            "base_total_return": base_metrics["total_return"],
                            "ablated_total_return": metrics["total_return"],
                            "total_return_impact": float(metrics["total_return"])
                            - float(base_metrics["total_return"]),
                            "base_maximum_drawdown": base_metrics["maximum_drawdown"],
                            "ablated_maximum_drawdown": metrics["maximum_drawdown"],
                            "maximum_drawdown_impact": float(metrics["maximum_drawdown"])
                            - float(base_metrics["maximum_drawdown"]),
                            "base_profit_factor": base_metrics["profit_factor"],
                            "ablated_profit_factor": metrics["profit_factor"],
                            "profit_factor_impact": float(metrics["profit_factor"])
                            - float(base_metrics["profit_factor"]),
                        }
                    )
    return pd.DataFrame(rows)
