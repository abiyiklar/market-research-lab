from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from bist_research.backtest.engine import load_feature_data
from bist_research.backtest.models import BacktestConfig

from .ablation import run_ablation_analysis
from .experiment import (
    DEFAULT_RANDOM_SEED,
    MAX_EXPERIMENTS,
    Experiment,
    current_git_commit,
    generate_experiments,
)
from .report import ResearchArtifacts, write_research_outputs
from .scoring import apply_stress_gate, score_experiments
from .strategies import STRATEGY_NAMES
from .stress_tests import run_stress_tests, stress_summary
from .walk_forward import (
    generate_walk_forward_windows,
    run_full_experiment,
    run_walk_forward_experiment,
)


@dataclass(frozen=True)
class ResearchRunResult:
    leaderboard: pd.DataFrame
    experiments: pd.DataFrame
    walk_forward: pd.DataFrame
    stress_tests: pd.DataFrame
    ablation: pd.DataFrame
    artifacts: ResearchArtifacts


def configure_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("bist_research.research")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "research.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled TUPRS strategy research.")
    parser.add_argument("--input", type=Path, default=Path("data/features/tuprs_features.parquet"))
    parser.add_argument("--max-experiments", type=int, default=MAX_EXPERIMENTS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--strategy", nargs="+", choices=STRATEGY_NAMES)
    parser.add_argument("--output-dir", type=Path, default=Path("data/research"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def run_research_pipeline(
    input_path: Path,
    max_experiments: int,
    random_seed: int,
    strategy_names: Sequence[str] | None,
    output_dir: Path,
    report_dir: Path,
    logger: logging.Logger | None = None,
) -> ResearchRunResult:
    active_logger = logger or logging.getLogger("bist_research.research")
    frame = load_feature_data(input_path)
    commit = current_git_commit(Path.cwd())
    experiments = generate_experiments(
        max_experiments=max_experiments,
        random_seed=random_seed,
        strategy_names=strategy_names,
        git_commit=commit,
    )
    base_config = BacktestConfig()
    windows = generate_walk_forward_windows(frame)
    experiment_rows: list[dict[str, object]] = []
    walk_rows: list[dict[str, object]] = []
    for index, experiment in enumerate(experiments, start=1):
        active_logger.info(
            "Experiment %s/%s: %s (%s)",
            index,
            len(experiments),
            experiment.strategy_name,
            experiment.experiment_id,
        )
        full_row, _ = run_full_experiment(frame, experiment, base_config)
        experiment_rows.append(full_row)
        walk_rows.extend(
            run_walk_forward_experiment(frame, experiment, base_config, windows)
        )
    experiment_results = pd.DataFrame(experiment_rows)
    walk_forward = pd.DataFrame(walk_rows)
    leaderboard = score_experiments(walk_forward)
    experiment_map: dict[str, Experiment] = {
        experiment.experiment_id: experiment for experiment in experiments
    }
    stress_tests = run_stress_tests(
        frame,
        leaderboard,
        experiment_map,
        base_config,
        random_seed,
    )
    leaderboard = leaderboard.drop(
        columns=["stress_test_result", "stability_class"]
    ).merge(stress_summary(stress_tests), on="experiment_id", how="left")
    leaderboard = apply_stress_gate(leaderboard)
    ablation = run_ablation_analysis(
        frame,
        leaderboard,
        experiment_map,
        base_config,
    )
    leaderboard = leaderboard.merge(
        experiment_results.loc[:, ["experiment_id", "config"]],
        on="experiment_id",
        how="left",
    )
    artifacts = write_research_outputs(
        experiment_results,
        walk_forward,
        stress_tests,
        ablation,
        leaderboard,
        output_dir,
        report_dir,
        random_seed,
        commit,
    )
    return ResearchRunResult(
        leaderboard=leaderboard,
        experiments=experiment_results,
        walk_forward=walk_forward,
        stress_tests=stress_tests,
        ablation=ablation,
        artifacts=artifacts,
    )


def _terminal_table(leaderboard: pd.DataFrame) -> str:
    columns = [
        "strategy_name",
        "parameters",
        "total_oos_trades",
        "positive_window_ratio",
        "median_oos_cagr",
        "median_profit_factor",
        "worst_drawdown",
        "stress_test_result",
        "stability_class",
        "composite_score",
        "decision",
    ]
    return leaderboard.head(10).loc[:, columns].to_string(index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging()
    try:
        result = run_research_pipeline(
            input_path=args.input,
            max_experiments=args.max_experiments,
            random_seed=args.random_seed,
            strategy_names=args.strategy,
            output_dir=args.output_dir,
            report_dir=args.report_dir,
            logger=logger,
        )
    except Exception:
        logger.exception("Research run failed")
        return 1
    print("Top 10 strategy candidates")
    print(_terminal_table(result.leaderboard))
    robust = int(result.leaderboard["decision"].eq("robust_candidate").sum())
    if robust == 0:
        print("robust candidate bulunamadı")
    else:
        print(f"Robust candidates: {robust}")
    print(f"Report: {result.artifacts.research_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
