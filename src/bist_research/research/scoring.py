from __future__ import annotations

import numpy as np
import pandas as pd


MINIMUM_OOS_TRADES = 30
MINIMUM_POSITIVE_WINDOW_RATIO = 0.60
MINIMUM_MEDIAN_PROFIT_FACTOR = 1.10
MINIMUM_WORST_DRAWDOWN = -0.35


def _finite_group(group: pd.DataFrame) -> bool:
    columns = (
        "total_return",
        "cagr",
        "maximum_drawdown",
        "sharpe_ratio",
        "sortino_ratio",
        "profit_factor",
        "win_rate",
        "time_in_market",
        "benchmark_excess_return",
    )
    values = group.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy()
    return bool(np.isfinite(values).all())


def composite_score(
    median_cagr: float,
    median_profit_factor: float,
    median_sharpe: float,
    worst_drawdown: float,
    return_std: float,
    median_benchmark_excess: float,
    total_trades: int,
) -> float:
    trade_adequacy = min(total_trades / MINIMUM_OOS_TRADES, 2.0)
    return float(
        median_cagr * 100
        + min(median_profit_factor, 4.0) * 2
        + median_sharpe * 2
        - abs(worst_drawdown) * 30
        - return_std * 10
        + median_benchmark_excess * 50
        + trade_adequacy
    )


def score_experiments(walk_forward_results: pd.DataFrame) -> pd.DataFrame:
    oos = walk_forward_results.loc[walk_forward_results["split"].eq("oos")].copy()
    rows: list[dict[str, object]] = []
    for experiment_id, group in oos.groupby("experiment_id", sort=False):
        total_trades = int(group["total_trades"].sum())
        positive_ratio = float(group["total_return"].gt(0).mean())
        median_cagr = float(group["cagr"].median())
        median_profit_factor = float(group["profit_factor"].median())
        median_sharpe = float(group["sharpe_ratio"].median())
        worst_drawdown = float(group["maximum_drawdown"].min())
        return_std = float(group["total_return"].std(ddof=0))
        median_excess = float(group["benchmark_excess_return"].median())
        quality_pass = bool(group["data_quality_pass"].all()) and _finite_group(group)
        lookahead_pass = bool(group["lookahead_pass"].all())
        gate_results = {
            "minimum_trades": total_trades >= MINIMUM_OOS_TRADES,
            "positive_windows": positive_ratio >= MINIMUM_POSITIVE_WINDOW_RATIO,
            "profit_factor": median_profit_factor >= MINIMUM_MEDIAN_PROFIT_FACTOR,
            "drawdown": worst_drawdown >= MINIMUM_WORST_DRAWDOWN,
            "data_quality": quality_pass,
            "lookahead": lookahead_pass,
        }
        hard_gate_pass = all(gate_results.values())
        failed_gates = ";".join(name for name, passed in gate_results.items() if not passed)
        score = composite_score(
            median_cagr,
            median_profit_factor,
            median_sharpe,
            worst_drawdown,
            return_std,
            median_excess,
            total_trades,
        )
        first = group.iloc[0]
        rows.append(
            {
                "experiment_id": experiment_id,
                "strategy_name": first["strategy_name"],
                "parameters": first["parameters"],
                "random_seed": first["random_seed"],
                "git_commit": first["git_commit"],
                "total_oos_trades": total_trades,
                "oos_window_count": int(len(group)),
                "positive_window_ratio": positive_ratio,
                "median_oos_cagr": median_cagr,
                "median_profit_factor": median_profit_factor,
                "median_sharpe": median_sharpe,
                "worst_drawdown": worst_drawdown,
                "oos_return_std": return_std,
                "median_benchmark_excess_return": median_excess,
                "data_quality_pass": quality_pass,
                "lookahead_pass": lookahead_pass,
                "hard_gate_pass": hard_gate_pass,
                "failed_gates": failed_gates,
                "composite_score": score,
                "stress_test_result": "not_run",
                "stability_class": "not_run",
                "decision": "robust_candidate" if hard_gate_pass else "rejected",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["hard_gate_pass", "composite_score"], ascending=[False, False]
    ).reset_index(drop=True)


def apply_stress_gate(leaderboard: pd.DataFrame) -> pd.DataFrame:
    finalized = leaderboard.copy()
    finalized["decision"] = "rejected"
    reliable = (
        finalized["hard_gate_pass"].astype(bool)
        & finalized["stress_test_result"].eq("passed")
        & finalized["stability_class"].eq("stable")
    )
    finalized.loc[reliable, "decision"] = "robust_candidate"
    return finalized
