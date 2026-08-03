"""Transparent cross-sectional scoring and causal signal research."""

from .pipeline import SignalPipelineOutputs, run_signal_pipeline
from .rules import add_signal_rules
from .scoring import build_score_panel, prepare_symbol_inputs

__all__ = [
    "SignalPipelineOutputs",
    "add_signal_rules",
    "build_score_panel",
    "prepare_symbol_inputs",
    "run_signal_pipeline",
]
