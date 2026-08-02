from __future__ import annotations

import pandas as pd
import pytest

from bist_research.research.experiment import generate_experiments
from bist_research.research.report import append_csv
from bist_research.research.scoring import apply_stress_gate, score_experiments
from bist_research.research.strategies import (
    STRATEGY_NAMES,
    build_strategy,
    deterministic_signal_filter,
    strategy_is_causal,
)
from bist_research.research.walk_forward import generate_walk_forward_windows


def _research_frame(row_count: int = 80) -> pd.DataFrame:
    index = pd.Series(range(row_count), dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.date_range("2013-01-02", periods=row_count, freq="B"),
            "tuprs_close": 100.0 + index,
            "xu100_close": 1_000.0 + index,
            "relative_strength_xu100": 1.0 + index / 100,
            "relative_strength_xusin": 1.0 + index / 120,
        }
    )


def test_experiment_generation_is_bounded_unique_and_deterministic() -> None:
    first = generate_experiments(25, random_seed=7, git_commit="abc123")
    second = generate_experiments(25, random_seed=7, git_commit="abc123")

    assert first == second
    assert len(first) == 25
    assert len({experiment.experiment_id for experiment in first}) == 25
    assert {experiment.strategy_name for experiment in first} == set(STRATEGY_NAMES)
    with pytest.raises(ValueError, match="between 1 and 150"):
        generate_experiments(151)


def test_git_commit_is_part_of_experiment_identity() -> None:
    first = generate_experiments(1, git_commit="first")[0]
    second = generate_experiments(1, git_commit="second")[0]

    assert first.experiment_id != second.experiment_id


def test_breakout_level_uses_only_prior_closes() -> None:
    strategy = build_strategy(
        "breakout",
        {
            "ema_fast": 10,
            "ema_medium": 50,
            "ema_long": 100,
            "breakout_window": 20,
        },
    )
    prepared = strategy.prepare(_research_frame())
    row_number = 30

    assert prepared.loc[row_number, "research_breakout_high"] == prepared.loc[
        row_number - 20 : row_number - 1, "tuprs_close"
    ].max()
    assert prepared.loc[row_number, "research_breakout_high"] < prepared.loc[
        row_number, "tuprs_close"
    ]


@pytest.mark.parametrize("strategy_name", STRATEGY_NAMES)
def test_strategy_features_are_causal(strategy_name: str) -> None:
    parameters = {
        "ema_fast": 10,
        "ema_medium": 50,
        "ema_long": 100,
        "breakout_window": 20,
    }

    assert strategy_is_causal(build_strategy(strategy_name, parameters), _research_frame())


def test_walk_forward_windows_roll_one_year_without_overlap() -> None:
    frame = pd.DataFrame(
        {"date": pd.date_range("2013-01-01", "2024-12-31", freq="B")}
    )
    windows = generate_walk_forward_windows(frame)

    assert windows[0].train_start == pd.Timestamp("2013-01-01")
    assert windows[0].train_end < windows[0].validation_start
    assert windows[0].validation_end < windows[0].oos_start
    assert windows[1].train_start.year == windows[0].train_start.year + 1


def test_scoring_applies_all_hard_gates() -> None:
    rows = []
    for window in range(5):
        rows.append(
            {
                "experiment_id": "exp_pass",
                "strategy_name": "trend_following",
                "parameters": "{}",
                "random_seed": 1,
                "git_commit": "abc",
                "window_id": f"wf_{window}",
                "split": "oos",
                "total_trades": 6,
                "total_return": 0.10,
                "cagr": 0.10,
                "maximum_drawdown": -0.20,
                "sharpe_ratio": 1.0,
                "sortino_ratio": 1.2,
                "profit_factor": 1.3,
                "win_rate": 0.55,
                "time_in_market": 0.40,
                "benchmark_excess_return": 0.02,
                "data_quality_pass": True,
                "lookahead_pass": True,
            }
        )

    leaderboard = score_experiments(pd.DataFrame(rows))

    assert bool(leaderboard.loc[0, "hard_gate_pass"])
    assert leaderboard.loc[0, "decision"] == "robust_candidate"


def test_append_csv_preserves_existing_experiment(tmp_path) -> None:
    path = tmp_path / "results.csv"
    append_csv(path, pd.DataFrame([{"experiment_id": "one", "score": 1.0}]), ["experiment_id"])
    combined = append_csv(
        path,
        pd.DataFrame(
            [
                {"experiment_id": "one", "score": 99.0},
                {"experiment_id": "two", "score": 2.0},
            ]
        ),
        ["experiment_id"],
    )

    assert combined.set_index("experiment_id").loc["one", "score"] == 1.0
    assert len(combined) == 2


def test_robust_decision_requires_stable_passed_stress_result() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "experiment_id": "stable",
                "hard_gate_pass": True,
                "stress_test_result": "passed",
                "stability_class": "stable",
            },
            {
                "experiment_id": "fragile",
                "hard_gate_pass": True,
                "stress_test_result": "degraded",
                "stability_class": "fragile",
            },
        ]
    )

    finalized = apply_stress_gate(leaderboard).set_index("experiment_id")

    assert finalized.loc["stable", "decision"] == "robust_candidate"
    assert finalized.loc["fragile", "decision"] == "rejected"


def test_signal_skipping_is_seeded_and_repeatable() -> None:
    first = deterministic_signal_filter(42)
    second = deterministic_signal_filter(42)
    rows = [pd.Series({"date": date}) for date in pd.date_range("2024-01-01", periods=50)]

    assert [first(row) for row in rows] == [second(row) for row in rows]
