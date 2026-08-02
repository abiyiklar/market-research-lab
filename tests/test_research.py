from __future__ import annotations

import pandas as pd
import pytest

from bist_research.backtest.models import BacktestConfig, BacktestResult
from bist_research.research.experiment import (
    BASELINE_PARAMETERS,
    Experiment,
    generate_experiments,
)
from bist_research.research.report import append_csv
from bist_research.research.scoring import apply_stress_gate, score_experiments
from bist_research.research.strategies import (
    STRATEGY_NAMES,
    build_strategy,
    deterministic_signal_filter,
    strong_causality_audit,
    strategy_is_causal,
)
from bist_research.research.stress_tests import run_stress_tests
from bist_research.research.walk_forward import generate_walk_forward_windows
from bist_research.research.walk_forward import (
    WalkForwardWindow,
    run_nested_walk_forward,
)


class _FakeStrategy:
    def __init__(self, name: str, parameters: dict[str, float | int]) -> None:
        self.name = name
        self.parameters = parameters

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.copy()

    def backtest_config(self, base: BacktestConfig) -> BacktestConfig:
        return base


def _nested_metrics(value: float) -> dict[str, object]:
    return {
        "total_trades": 5,
        "total_return": value,
        "cagr": value,
        "maximum_drawdown": -0.10,
        "sharpe_ratio": value * 10,
        "sortino_ratio": value * 12,
        "calmar_ratio": value * 5,
        "annual_volatility": 0.15,
        "profit_factor": 1.5,
        "win_rate": 0.55,
        "time_in_market": 0.4,
        "benchmark_excess_return": value - 0.05,
        "absolute_return_difference": value - 0.05,
        "volatility_matched_benchmark_return": 0.04,
        "volatility_matched_excess_return": value - 0.04,
        "sharpe_difference": value * 5,
        "calmar_difference": value * 3,
        "maximum_drawdown_difference": 0.05,
        "exposure_adjusted_return": value / 0.4,
        "exposure_matched_benchmark_return": 0.02,
    }


def _patch_nested_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bist_research.research.walk_forward.build_strategy",
        lambda name, parameters: _FakeStrategy(name, parameters),
    )
    monkeypatch.setattr(
        "bist_research.research.walk_forward.strategy_is_causal",
        lambda strategy, frame: True,
    )

    def fake_period(
        prepared: pd.DataFrame,
        strategy: _FakeStrategy,
        base_config: BacktestConfig,
        **kwargs: object,
    ) -> tuple[None, dict[str, object], dict[str, object], bool]:
        del base_config, kwargs
        column = f"{strategy.parameters['candidate']}_score"
        value = float(prepared[column].mean())
        benchmark = {
            "total_return": 0.05,
            "sharpe_ratio": 0.5,
            "calmar_ratio": 0.4,
            "maximum_drawdown": -0.15,
        }
        return None, _nested_metrics(value), benchmark, True

    monkeypatch.setattr(
        "bist_research.research.walk_forward.run_strategy_period",
        fake_period,
    )


def _selection_experiments() -> list[Experiment]:
    return [
        Experiment("exp_a", "trend_following", {"candidate": "a"}, 1, "abc"),
        Experiment("exp_b", "trend_following", {"candidate": "b"}, 1, "abc"),
    ]


def _selection_frame() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2013-01-02",
            "2013-06-03",
            "2014-01-02",
            "2014-06-03",
            "2015-01-02",
            "2015-06-03",
            "2016-01-04",
            "2016-06-03",
            "2017-01-02",
            "2017-06-02",
        ]
    )
    frame = pd.DataFrame({"date": dates, "a_score": 0.10, "b_score": 0.10})
    frame.loc[frame["date"].dt.year.eq(2014), ["a_score", "b_score"]] = [0.30, 0.10]
    frame.loc[frame["date"].dt.year.eq(2015), ["a_score", "b_score"]] = [-0.50, 0.90]
    frame.loc[frame["date"].dt.year.eq(2017), ["a_score", "b_score"]] = [0.20, 0.40]
    return frame


def _selection_windows() -> list[WalkForwardWindow]:
    return [
        WalkForwardWindow(
            "wf_1",
            pd.Timestamp("2013-01-01"),
            pd.Timestamp("2013-12-31"),
            pd.Timestamp("2014-01-01"),
            pd.Timestamp("2014-12-31"),
            pd.Timestamp("2015-01-01"),
            pd.Timestamp("2015-12-31"),
        ),
        WalkForwardWindow(
            "wf_2",
            pd.Timestamp("2015-01-01"),
            pd.Timestamp("2015-12-31"),
            pd.Timestamp("2016-01-01"),
            pd.Timestamp("2016-12-31"),
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2017-12-31"),
        ),
    ]


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
    assert all(
        current.oos_end < following.oos_start
        for current, following in zip(windows, windows[1:])
    )


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
                "selected_for_oos": True,
                "selected_on_validation": True,
                "oos_run_count": 1,
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
                "volatility_matched_excess_return": 0.03,
                "sharpe_difference": 0.5,
                "calmar_difference": 0.4,
                "maximum_drawdown_difference": 0.1,
                "exposure_adjusted_return": 0.25,
                "exposure_matched_benchmark_return": 0.05,
                "data_quality_pass": True,
                "lookahead_pass": True,
            }
        )

    leaderboard = score_experiments(pd.DataFrame(rows))

    assert bool(leaderboard.loc[0, "hard_gate_pass"])
    assert leaderboard.loc[0, "decision"] == "preliminary_research_candidate"


def test_economic_label_requires_hard_gate_evidence() -> None:
    row = {
        "experiment_id": "thin_evidence",
        "strategy_name": "relative_strength",
        "parameters": "{}",
        "random_seed": 1,
        "git_commit": "abc",
        "window_id": "wf_1",
        "split": "oos",
        "selected_for_oos": True,
        "selected_on_validation": True,
        "oos_run_count": 1,
        "total_trades": 2,
        "total_return": 0.50,
        "cagr": 0.50,
        "maximum_drawdown": -0.05,
        "sharpe_ratio": 2.0,
        "sortino_ratio": 2.5,
        "profit_factor": 3.0,
        "win_rate": 0.75,
        "time_in_market": 0.20,
        "benchmark_excess_return": 0.20,
        "volatility_matched_excess_return": 0.20,
        "sharpe_difference": 1.0,
        "calmar_difference": 1.0,
        "maximum_drawdown_difference": 0.10,
        "exposure_adjusted_return": 2.5,
        "exposure_matched_benchmark_return": 0.10,
        "data_quality_pass": True,
        "lookahead_pass": True,
    }

    scored = score_experiments(pd.DataFrame([row])).iloc[0]

    assert not bool(scored["hard_gate_pass"])
    assert not bool(scored["economically_competitive"])


def test_append_csv_preserves_existing_experiment(tmp_path) -> None:
    path = tmp_path / "results.csv"
    append_csv(
        path,
        pd.DataFrame(
            [
                {
                    "experiment_id": "one",
                    "score": 1.0,
                    "failed_gates": None,
                    "stress_test_result": None,
                    "stability_class": None,
                }
            ]
        ),
        ["experiment_id"],
    )
    combined = append_csv(
        path,
        pd.DataFrame(
            [
                {"experiment_id": "one", "score": 99.0, "failed_gates": "drawdown"},
                {"experiment_id": "two", "score": 2.0, "failed_gates": "none"},
            ]
        ),
        ["experiment_id"],
    )

    assert combined.set_index("experiment_id").loc["one", "score"] == 1.0
    assert combined.set_index("experiment_id").loc["one", "failed_gates"] == "none"
    assert not combined.isna().any().any()
    assert len(combined) == 2


def test_robust_decision_requires_stable_passed_stress_result() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "experiment_id": "stable",
                "hard_gate_pass": True,
                "stress_test_result": "passed",
                "stability_class": "stable",
                "economically_competitive": True,
            },
            {
                "experiment_id": "fragile",
                "hard_gate_pass": True,
                "stress_test_result": "degraded",
                "stability_class": "fragile",
                "economically_competitive": False,
            },
        ]
    )

    finalized = apply_stress_gate(leaderboard).set_index("experiment_id")

    assert finalized.loc["stable", "decision"] == "economically_competitive_candidate"
    assert bool(finalized.loc["stable", "statistically_robust"])
    assert not bool(finalized.loc["stable", "deployment_ready"])
    assert finalized.loc["fragile", "decision"] == "rejected"


def test_signal_skipping_is_seeded_and_repeatable() -> None:
    first = deterministic_signal_filter(42)
    second = deterministic_signal_filter(42)
    rows = [pd.Series({"date": date}) for date in pd.date_range("2024-01-01", periods=50)]

    assert [first(row) for row in rows] == [second(row) for row in rows]


def test_oos_prices_do_not_change_validation_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_nested_runner(monkeypatch)
    frame = _selection_frame()
    window = _selection_windows()[:1]
    original = pd.DataFrame(
        run_nested_walk_forward(
            frame,
            _selection_experiments(),
            BacktestConfig(),
            window,
            maximum_selected_per_window=1,
        )
    )
    changed = frame.copy()
    changed.loc[changed["date"].dt.year.eq(2015), ["a_score", "b_score"]] = [99.0, -99.0]
    mutated = pd.DataFrame(
        run_nested_walk_forward(
            changed,
            _selection_experiments(),
            BacktestConfig(),
            window,
            maximum_selected_per_window=1,
        )
    )

    assert original.loc[original["split"].eq("oos"), "experiment_id"].tolist() == ["exp_a"]
    assert mutated.loc[mutated["split"].eq("oos"), "experiment_id"].tolist() == ["exp_a"]


def test_validation_not_oos_determines_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_nested_runner(monkeypatch)
    results = pd.DataFrame(
        run_nested_walk_forward(
            _selection_frame(),
            _selection_experiments(),
            BacktestConfig(),
            _selection_windows()[:1],
            maximum_selected_per_window=1,
        )
    )

    oos = results.loc[results["split"].eq("oos")]
    assert oos["experiment_id"].tolist() == ["exp_a"]
    assert bool(oos.iloc[0]["selected_on_validation"])
    assert oos.iloc[0]["selection_rank"] == 1


def test_each_selected_parameter_runs_once_per_oos_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_nested_runner(monkeypatch)
    results = pd.DataFrame(
        run_nested_walk_forward(
            _selection_frame(),
            _selection_experiments(),
            BacktestConfig(),
            _selection_windows(),
            maximum_selected_per_window=1,
        )
    )
    oos = results.loc[results["split"].eq("oos")]

    assert not oos.duplicated(["window_id", "experiment_id"]).any()
    assert oos["oos_run_count"].eq(1).all()


def test_future_window_does_not_influence_earlier_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_nested_runner(monkeypatch)
    frame = _selection_frame()
    original = pd.DataFrame(
        run_nested_walk_forward(
            frame,
            _selection_experiments(),
            BacktestConfig(),
            _selection_windows(),
            maximum_selected_per_window=1,
        )
    )
    changed = frame.copy()
    changed.loc[changed["date"].dt.year.ge(2016), ["a_score", "b_score"]] = [-50.0, 50.0]
    mutated = pd.DataFrame(
        run_nested_walk_forward(
            changed,
            _selection_experiments(),
            BacktestConfig(),
            _selection_windows(),
            maximum_selected_per_window=1,
        )
    )

    first_original = original.loc[
        original["window_id"].eq("wf_1") & original["split"].eq("oos"),
        "experiment_id",
    ].tolist()
    first_mutated = mutated.loc[
        mutated["window_id"].eq("wf_1") & mutated["split"].eq("oos"),
        "experiment_id",
    ].tolist()
    assert first_original == first_mutated == ["exp_a"]


def test_remove_best_trade_stress_reruns_and_recomputes_path_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = Experiment(
        "exp_base",
        "baseline_v1_fixed",
        BASELINE_PARAMETERS.copy(),
        1,
        "abc",
    )
    leaderboard = pd.DataFrame([{"experiment_id": experiment.experiment_id}])
    walk_forward = pd.DataFrame(
        [
            {
                "experiment_id": experiment.experiment_id,
                "window_id": "wf_1",
                "split": "oos",
                "split_start": "2024-01-01",
                "split_end": "2024-12-31",
                "selected_for_oos": True,
            }
        ]
    )
    trades = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2024-01-02", "2024-02-01", "2024-03-01"]),
            "net_pnl": [300.0, 200.0, 100.0],
        }
    )
    base_result = BacktestResult(trades, pd.DataFrame(), BacktestConfig())

    def fake_scenario(*args: object, **kwargs: object) -> tuple[BacktestResult, dict[str, object]]:
        del args
        removed = len(kwargs.get("disabled_signal_dates", frozenset()))
        metrics = {
            "total_return": 0.20 - removed * 0.08,
            "maximum_drawdown": -0.10 - removed * 0.07,
            "profit_factor": 2.0 - removed * 0.3,
            "total_trades": 3 - removed,
            "sharpe_ratio": 1.0 - removed * 0.4,
            "calmar_ratio": 1.5 - removed * 0.5,
            "annual_volatility": 0.15 + removed * 0.03,
        }
        return base_result, metrics

    monkeypatch.setattr(
        "bist_research.research.stress_tests.run_selected_oos_scenario",
        fake_scenario,
    )
    results = run_stress_tests(
        pd.DataFrame({"date": pd.to_datetime(["2024-01-02"])}),
        leaderboard,
        walk_forward,
        {experiment.experiment_id: experiment},
        BacktestConfig(),
        random_seed=1,
        top_n=1,
    ).set_index("scenario")

    assert results.loc["remove_best_1_trade", "maximum_drawdown"] == pytest.approx(-0.17)
    assert results.loc["remove_best_1_trade", "sharpe_ratio"] == pytest.approx(0.6)
    assert results.loc["remove_best_3_trades", "calmar_ratio"] == pytest.approx(0.0)
    assert results.loc["remove_best_3_trades", "annual_volatility"] == pytest.approx(0.24)
    assert results.loc["remove_best_1_trade", "data_scope"] == "selected_oos"


def test_strong_causality_audit_covers_features_signals_trades_and_equity() -> None:
    row_count = 240
    index = pd.Series(range(row_count), dtype="float64")
    close = 100.0 + index * 0.2
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-02", periods=row_count, freq="B"),
            "tuprs_open": close,
            "tuprs_high": close + 2.0,
            "tuprs_low": close - 2.0,
            "tuprs_close": close + 0.5,
            "tuprs_adj_close": close + 0.5,
            "tuprs_volume": 10_000.0 + index,
            "xu100_close": 1_000.0 + index,
            "xu100_return_1d": 0.001,
            "relative_strength_xu100": (close + 0.5) / (1_000.0 + index),
            "relative_strength_xusin": (close + 0.5) / (800.0 + index),
            "relative_momentum_20d": 0.02,
            "relative_momentum_60d": 0.03,
            "volume_ratio_20d": 1.2,
            "ema_20": close - 1.0,
            "ema_50": close - 2.0,
            "ema_100": close - 3.0,
            "ema_200": close - 4.0,
            "rsi_14": 60.0,
            "atr_14": 4.0,
            "brent_return_1d": 0.001,
            "usdtry_return_1d": 0.001,
            "dividend_per_share": 0.0,
            "stock_split_factor": 0.0,
            "corporate_action_flag": False,
            "price_basis": "raw_ohlc_split_adjusted",
            "is_indicator_warmup": pd.Series(range(row_count)).lt(200),
        }
    )
    strategy = build_strategy("baseline_v1_fixed", BASELINE_PARAMETERS.copy())

    assert strong_causality_audit(strategy, frame, random_seed=17, cutpoint_count=3)
