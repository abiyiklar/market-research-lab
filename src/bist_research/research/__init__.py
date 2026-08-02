"""Controlled, reproducible strategy research for the TUPRS feature dataset."""

from .experiment import Experiment, generate_experiments
from .strategies import STRATEGY_NAMES, ResearchStrategy, build_strategy

__all__ = [
    "Experiment",
    "ResearchStrategy",
    "STRATEGY_NAMES",
    "build_strategy",
    "generate_experiments",
]
