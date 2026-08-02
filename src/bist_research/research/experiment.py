from __future__ import annotations

import hashlib
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .strategies import STRATEGY_NAMES


MAX_EXPERIMENTS = 150
DEFAULT_RANDOM_SEED = 20240801

PARAMETER_SPACE: dict[str, tuple[float | int, ...]] = {
    "ema_fast": (10, 20, 30),
    "ema_medium": (50, 75, 100),
    "ema_long": (100, 150, 200),
    "atr_stop": (2.0, 2.5, 3.0),
    "atr_trailing": (2.5, 3.0, 4.0),
    "rsi_lower": (40, 45, 50),
    "rsi_upper": (65, 70, 75),
    "volume_ratio": (0.8, 1.0, 1.2),
    "breakout_window": (20, 55),
    "maximum_holding": (30, 60, 120),
}

BASELINE_PARAMETERS: dict[str, float | int] = {
    "ema_fast": 20,
    "ema_medium": 50,
    "ema_long": 200,
    "atr_stop": 2.5,
    "atr_trailing": 3.0,
    "rsi_lower": 50,
    "rsi_upper": 72,
    "volume_ratio": 1.1,
    "breakout_window": 20,
    "maximum_holding": 60,
}


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    strategy_name: str
    parameters: dict[str, float | int]
    random_seed: int
    git_commit: str

    @property
    def parameters_json(self) -> str:
        return json.dumps(self.parameters, sort_keys=True, separators=(",", ":"))

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["parameters"] = self.parameters_json
        return record


def current_git_commit(repo_dir: Path = Path.cwd()) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _experiment_id(
    strategy_name: str,
    parameters: dict[str, float | int],
    git_commit: str,
) -> str:
    canonical = json.dumps(
        {"strategy": strategy_name, "parameters": parameters, "git_commit": git_commit},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"exp_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"


def _draw_parameters(rng: random.Random) -> dict[str, float | int]:
    while True:
        parameters = {key: rng.choice(values) for key, values in PARAMETER_SPACE.items()}
        if int(parameters["ema_fast"]) < int(parameters["ema_medium"]) < int(
            parameters["ema_long"]
        ):
            return parameters


def generate_experiments(
    max_experiments: int = MAX_EXPERIMENTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    strategy_names: Sequence[str] | None = None,
    git_commit: str = "unknown",
) -> list[Experiment]:
    if not 1 <= max_experiments <= MAX_EXPERIMENTS:
        raise ValueError(f"max_experiments must be between 1 and {MAX_EXPERIMENTS}")
    selected = tuple(strategy_names or STRATEGY_NAMES)
    unknown = sorted(set(selected).difference(STRATEGY_NAMES))
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")

    rng = random.Random(random_seed)
    experiments: list[Experiment] = []
    seen: set[str] = set()
    if "baseline_v1_fixed" in selected:
        baseline_id = _experiment_id("baseline_v1_fixed", BASELINE_PARAMETERS, git_commit)
        experiments.append(
            Experiment(
                experiment_id=baseline_id,
                strategy_name="baseline_v1_fixed",
                parameters=BASELINE_PARAMETERS.copy(),
                random_seed=random_seed,
                git_commit=git_commit,
            )
        )
        seen.add(baseline_id)

    searchable = tuple(name for name in selected if name != "baseline_v1_fixed")
    if not searchable:
        return experiments[:max_experiments]
    attempts = 0
    while len(experiments) < max_experiments and attempts < max_experiments * 100:
        name = searchable[attempts % len(searchable)]
        parameters = _draw_parameters(rng)
        experiment_id = _experiment_id(name, parameters, git_commit)
        attempts += 1
        if experiment_id in seen:
            continue
        seen.add(experiment_id)
        experiments.append(
            Experiment(
                experiment_id=experiment_id,
                strategy_name=name,
                parameters=parameters,
                random_seed=random_seed,
                git_commit=git_commit,
            )
        )
    if len(experiments) < max_experiments:
        raise RuntimeError("Could not generate enough unique experiments")
    return experiments
