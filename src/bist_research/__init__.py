"""BIST market research data collection and feature engineering package."""

from typing import Any

from .config import DEFAULT_SYMBOLS, INITIAL_BIST_SYMBOLS, START_DATE, SymbolConfig


def collect_all(*args: Any, **kwargs: Any) -> Any:
    """Load the collector lazily so module CLI execution stays warning-free."""

    from .collector import collect_all as run_collection

    return run_collection(*args, **kwargs)


__all__ = [
    "DEFAULT_SYMBOLS",
    "INITIAL_BIST_SYMBOLS",
    "START_DATE",
    "SymbolConfig",
    "collect_all",
]
