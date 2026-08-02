from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ResearchArtifacts:
    experiment_results: Path
    walk_forward_results: Path
    stress_test_results: Path
    ablation_results: Path
    candidate_parameters: Path
    research_report: Path
    research_leaderboard: Path
    top_candidates: Path
    robustness_heatmap: Path
    walk_forward_returns: Path
    strategy_comparison: Path
    parameter_stability: Path


def append_csv(
    path: Path,
    new_rows: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined = combined.drop_duplicates(key_columns, keep="first").reset_index(drop=True)
    combined.to_csv(path, index=False)
    return combined


def write_research_outputs(
    experiment_results: pd.DataFrame,
    walk_forward_results: pd.DataFrame,
    stress_results: pd.DataFrame,
    ablation_results: pd.DataFrame,
    leaderboard: pd.DataFrame,
    output_dir: Path,
    report_dir: Path,
    random_seed: int,
    git_commit: str,
) -> ResearchArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts = ResearchArtifacts(
        experiment_results=output_dir / "experiment_results.csv",
        walk_forward_results=output_dir / "walk_forward_results.csv",
        stress_test_results=output_dir / "stress_test_results.csv",
        ablation_results=output_dir / "ablation_results.csv",
        candidate_parameters=output_dir / "candidate_parameters.csv",
        research_report=report_dir / "research_report.md",
        research_leaderboard=report_dir / "research_leaderboard.csv",
        top_candidates=report_dir / "top_candidates.md",
        robustness_heatmap=report_dir / "robustness_heatmap.png",
        walk_forward_returns=report_dir / "walk_forward_returns.png",
        strategy_comparison=report_dir / "strategy_comparison.png",
        parameter_stability=report_dir / "parameter_stability.png",
    )
    append_csv(artifacts.experiment_results, experiment_results, ["experiment_id"])
    append_csv(
        artifacts.walk_forward_results,
        walk_forward_results,
        ["experiment_id", "window_id", "split"],
    )
    append_csv(
        artifacts.stress_test_results,
        stress_results,
        ["experiment_id", "scenario"],
    )
    append_csv(
        artifacts.ablation_results,
        ablation_results,
        ["experiment_id", "removed_filter"],
    )
    append_csv(artifacts.candidate_parameters, leaderboard, ["experiment_id"])
    leaderboard.to_csv(artifacts.research_leaderboard, index=False)
    _write_markdown_report(
        artifacts.research_report,
        leaderboard,
        walk_forward_results,
        stress_results,
        random_seed,
        git_commit,
    )
    _write_top_candidates(artifacts.top_candidates, leaderboard)
    _plot_robustness_heatmap(
        artifacts.robustness_heatmap, leaderboard, walk_forward_results
    )
    _plot_walk_forward_returns(
        artifacts.walk_forward_returns, leaderboard, walk_forward_results
    )
    _plot_strategy_comparison(artifacts.strategy_comparison, leaderboard)
    _plot_parameter_stability(artifacts.parameter_stability, leaderboard)
    return artifacts


def _percent(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def _write_markdown_report(
    path: Path,
    leaderboard: pd.DataFrame,
    walk_forward: pd.DataFrame,
    stress_results: pd.DataFrame,
    random_seed: int,
    git_commit: str,
) -> None:
    robust_count = int(leaderboard["decision"].eq("robust_candidate").sum())
    hard_gate_count = int(leaderboard["hard_gate_pass"].sum())
    lines = [
        "# TUPRS Strategy Research Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Random seed: `{random_seed}`",
        f"Git commit: `{git_commit}`",
        f"Experiments in this run: {len(leaderboard)}",
        f"Hard-gate passes: {hard_gate_count}",
        "",
        "## Methodology",
        "",
        "The engine runs a fixed experiment budget without OOS-driven retuning. Each strategy is tested with rolling 4-year train, 1-year validation, and 1-year OOS windows. Entry and close-based exit signals execute at a later open; intraday stops retain priority. No machine learning is used.",
        "",
        "The raw-price strategy and raw-price buy-and-hold benchmark are separate from the Adjusted Close total-return benchmark. Adjusted Close reflects Yahoo's dividend and split adjustments and is used only as a benchmark, never as strategy OHLC.",
        "",
        "## Top 5",
        "",
        "Rank | Strategy | OOS Trades | Positive Windows | Median CAGR | Median PF | Worst DD | Stability | Score | Decision",
        "---:|---|---:|---:|---:|---:|---:|---|---:|---",
    ]
    for rank, row in enumerate(leaderboard.head(5).itertuples(index=False), start=1):
        lines.append(
            f"{rank} | {row.strategy_name} | {row.total_oos_trades} | "
            f"{_percent(row.positive_window_ratio)} | {_percent(row.median_oos_cagr)} | "
            f"{row.median_profit_factor:.2f} | {_percent(row.worst_drawdown)} | "
            f"{row.stability_class} | {row.composite_score:.3f} | {row.decision}"
        )
    oos = walk_forward.loc[walk_forward["split"].eq("oos")]
    lines.extend(
        [
            "",
            "## Walk-Forward Summary",
            "",
            f"OOS window rows: {len(oos)}",
            f"Median OOS return across experiments/windows: {_percent(oos['total_return'].median())}",
            f"Median OOS benchmark excess: {_percent(oos['benchmark_excess_return'].median())}",
            "",
            "## Stress Summary",
            "",
        ]
    )
    for stability, count in stress_results.drop_duplicates("experiment_id")[
        "stability_class"
    ].value_counts().items():
        lines.append(f"- {stability}: {count}")
    lines.extend(["", "## Research Decision", ""])
    if robust_count:
        lines.append(f"{robust_count} strategy is classified as a robust candidate.")
    else:
        lines.append("robust candidate bulunamadı")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_top_candidates(path: Path, leaderboard: pd.DataFrame) -> None:
    lines = ["# Top Strategy Candidates", ""]
    for rank, row in enumerate(leaderboard.head(10).itertuples(index=False), start=1):
        lines.extend(
            [
                f"## {rank}. {row.strategy_name}",
                "",
                f"- Experiment: `{row.experiment_id}`",
                f"- Parameters: `{row.parameters}`",
                f"- OOS trades: {row.total_oos_trades}",
                f"- Positive windows: {_percent(row.positive_window_ratio)}",
                f"- Median CAGR: {_percent(row.median_oos_cagr)}",
                f"- Median profit factor: {row.median_profit_factor:.3f}",
                f"- Worst drawdown: {_percent(row.worst_drawdown)}",
                f"- Stress/stability: {row.stress_test_result} / {row.stability_class}",
                f"- Composite score: {row.composite_score:.3f}",
                f"- Decision: {row.decision}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_robustness_heatmap(
    path: Path, leaderboard: pd.DataFrame, walk_forward: pd.DataFrame
) -> None:
    top_ids = leaderboard.head(10)["experiment_id"]
    selected = walk_forward.loc[
        walk_forward["experiment_id"].isin(top_ids) & walk_forward["split"].eq("oos")
    ]
    pivot = selected.pivot(index="experiment_id", columns="window_id", values="total_return")
    pivot = pivot.reindex(top_ids)
    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(pivot.to_numpy() * 100, aspect="auto", cmap="RdYlGn")
    axis.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(pivot.index)), pivot.index)
    axis.set_title("Top Candidate OOS Returns (%)")
    figure.colorbar(image, ax=axis, label="Return (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_walk_forward_returns(
    path: Path, leaderboard: pd.DataFrame, walk_forward: pd.DataFrame
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    for experiment_id in leaderboard.head(5)["experiment_id"]:
        rows = walk_forward.loc[
            walk_forward["experiment_id"].eq(experiment_id)
            & walk_forward["split"].eq("oos")
        ].sort_values("window_id")
        axis.plot(rows["window_id"], rows["total_return"] * 100, marker="o", label=experiment_id)
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set(title="Walk-Forward OOS Returns", xlabel="Window", ylabel="Return (%)")
    axis.tick_params(axis="x", rotation=45)
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_strategy_comparison(path: Path, leaderboard: pd.DataFrame) -> None:
    comparison = leaderboard.groupby("strategy_name")["composite_score"].median().sort_values()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.barh(comparison.index, comparison.values, color="#176B87")
    axis.set(title="Median Composite Score by Strategy", xlabel="Composite Score")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_parameter_stability(path: Path, leaderboard: pd.DataFrame) -> None:
    records = []
    for row in leaderboard.itertuples(index=False):
        parameters = json.loads(row.parameters)
        records.append(
            {
                "ema_fast": parameters.get("ema_fast", 20),
                "score": row.composite_score,
                "strategy": row.strategy_name,
            }
        )
    values = pd.DataFrame(records)
    figure, axis = plt.subplots(figsize=(9, 5))
    for strategy, group in values.groupby("strategy"):
        axis.scatter(group["ema_fast"], group["score"], label=strategy, alpha=0.75)
    axis.set(title="Parameter Stability", xlabel="Fast EMA", ylabel="Composite Score")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
