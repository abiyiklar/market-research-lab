from __future__ import annotations

import numpy as np
import pandas as pd


MINIMUM_OOS_TRADES = 30
MINIMUM_POSITIVE_WINDOW_RATIO = 0.60
MINIMUM_MEDIAN_PROFIT_FACTOR = 1.10
MINIMUM_WORST_DRAWDOWN = -0.35

LEADERBOARD_COLUMNS = (
    "experiment_id",
    "methodology_version",
    "strategy_name",
    "parameters",
    "random_seed",
    "git_commit",
    "total_oos_trades",
    "oos_window_count",
    "oos_run_count",
    "positive_window_ratio",
    "median_oos_cagr",
    "median_profit_factor",
    "median_sharpe",
    "worst_drawdown",
    "oos_return_std",
    "median_benchmark_excess_return",
    "median_volatility_matched_excess_return",
    "median_sharpe_difference",
    "median_calmar_difference",
    "median_maximum_drawdown_difference",
    "median_exposure_adjusted_return",
    "median_exposure_matched_benchmark_return",
    "data_quality_pass",
    "lookahead_pass",
    "hard_gate_pass",
    "stress_stability_pass",
    "statistically_robust",
    "economically_competitive",
    "economic_absolute_value_pass",
    "economic_risk_adjusted_value_pass",
    "deployment_ready",
    "failed_gates",
    "composite_score",
    "stress_test_result",
    "stability_class",
    "decision",
)


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
        "volatility_matched_excess_return",
        "sharpe_difference",
        "calmar_difference",
        "maximum_drawdown_difference",
        "exposure_adjusted_return",
        "exposure_matched_benchmark_return",
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
    selected = walk_forward_results.get(
        "selected_for_oos",
        pd.Series(False, index=walk_forward_results.index),
    ).astype(bool)
    oos = walk_forward_results.loc[
        walk_forward_results["split"].eq("oos") & selected
    ].copy()
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
        median_volatility_matched_excess = float(
            group["volatility_matched_excess_return"].median()
        )
        median_sharpe_difference = float(group["sharpe_difference"].median())
        median_calmar_difference = float(group["calmar_difference"].median())
        median_drawdown_difference = float(
            group["maximum_drawdown_difference"].median()
        )
        median_exposure_adjusted_return = float(
            group["exposure_adjusted_return"].median()
        )
        median_exposure_matched_benchmark = float(
            group["exposure_matched_benchmark_return"].median()
        )
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
        failed_gates = failed_gates or "none"
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
        absolute_value_pass = median_excess >= 0
        risk_adjusted_value_pass = bool(
            median_volatility_matched_excess > 0
            and median_sharpe_difference > 0
            and median_calmar_difference > 0
            and median_drawdown_difference >= 0
            and float(group["total_return"].median())
            > median_exposure_matched_benchmark
        )
        economically_competitive = hard_gate_pass and (
            absolute_value_pass or risk_adjusted_value_pass
        )
        rows.append(
            {
                "experiment_id": experiment_id,
                "methodology_version": "nested_walk_forward_v2",
                "strategy_name": first["strategy_name"],
                "parameters": first["parameters"],
                "random_seed": first["random_seed"],
                "git_commit": first["git_commit"],
                "total_oos_trades": total_trades,
                "oos_window_count": int(len(group)),
                "oos_run_count": int(group["oos_run_count"].sum()),
                "positive_window_ratio": positive_ratio,
                "median_oos_cagr": median_cagr,
                "median_profit_factor": median_profit_factor,
                "median_sharpe": median_sharpe,
                "worst_drawdown": worst_drawdown,
                "oos_return_std": return_std,
                "median_benchmark_excess_return": median_excess,
                "median_volatility_matched_excess_return": median_volatility_matched_excess,
                "median_sharpe_difference": median_sharpe_difference,
                "median_calmar_difference": median_calmar_difference,
                "median_maximum_drawdown_difference": median_drawdown_difference,
                "median_exposure_adjusted_return": median_exposure_adjusted_return,
                "median_exposure_matched_benchmark_return": median_exposure_matched_benchmark,
                "data_quality_pass": quality_pass,
                "lookahead_pass": lookahead_pass,
                "hard_gate_pass": hard_gate_pass,
                "stress_stability_pass": False,
                "statistically_robust": False,
                "economically_competitive": economically_competitive,
                "economic_absolute_value_pass": absolute_value_pass,
                "economic_risk_adjusted_value_pass": risk_adjusted_value_pass,
                "deployment_ready": False,
                "failed_gates": failed_gates,
                "composite_score": score,
                "stress_test_result": "not_run",
                "stability_class": "not_run",
                "decision": "preliminary_research_candidate" if hard_gate_pass else "rejected",
            }
        )
    leaderboard = pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)
    if leaderboard.empty:
        return leaderboard
    return leaderboard.sort_values(
        ["hard_gate_pass", "composite_score"], ascending=[False, False]
    ).reset_index(drop=True)


def apply_stress_gate(leaderboard: pd.DataFrame) -> pd.DataFrame:
    finalized = leaderboard.copy()
    finalized["stress_test_result"] = finalized["stress_test_result"].fillna("not_run")
    finalized["stability_class"] = finalized["stability_class"].fillna("not_run")
    finalized["decision"] = "rejected"
    finalized["stress_stability_pass"] = (
        finalized["stress_test_result"].eq("passed")
        & finalized["stability_class"].eq("stable")
    )
    finalized["statistically_robust"] = (
        finalized["hard_gate_pass"].astype(bool)
        & finalized["stress_stability_pass"].astype(bool)
    )
    preliminary = finalized["statistically_robust"].astype(bool)
    competitive = preliminary & finalized["economically_competitive"].astype(bool)
    finalized.loc[preliminary, "decision"] = "preliminary_research_candidate"
    finalized.loc[competitive, "decision"] = "economically_competitive_candidate"
    finalized["deployment_ready"] = False
    return finalized
