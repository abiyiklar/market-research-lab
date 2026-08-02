"""BIST market research data collection and feature engineering package."""

from typing import Any

from .config import DEFAULT_SYMBOLS, START_DATE, SymbolConfig


def collect_all(*args: Any, **kwargs: Any) -> Any:
    from .collector import collect_all as run_collection

    return run_collection(*args, **kwargs)

__all__ = [
    "DEFAULT_SYMBOLS",
    "START_DATE",
    "SymbolConfig",
    "collect_all",
]
