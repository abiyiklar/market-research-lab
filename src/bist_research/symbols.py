from __future__ import annotations

import re
from collections.abc import Iterable

from .config import INITIAL_BIST_PANEL, INITIAL_BIST_SYMBOLS, SymbolConfig, universe_symbols


def safe_symbol_name(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", symbol).strip("_")
    return cleaned or "symbol"


def panel_configs(
    universe: str | None = None,
    symbols: Iterable[str] | None = None,
) -> tuple[SymbolConfig, ...]:
    known = {config.symbol: config for config in INITIAL_BIST_SYMBOLS}
    if symbols:
        return tuple(
            known.get(symbol, SymbolConfig(symbol, "equity", symbol))
            for symbol in symbols
        )
    return universe_symbols(universe or INITIAL_BIST_PANEL)
