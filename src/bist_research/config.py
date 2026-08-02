from dataclasses import dataclass
from typing import Mapping


START_DATE = "2013-01-01"
INITIAL_BIST_PANEL = "initial_bist_panel"
MARKET_BENCHMARK = "XU100.IS"


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    asset_type: str
    short_name: str
    preferred_sector_benchmark: str | None = None
    market_benchmark: str = MARKET_BENCHMARK
    enabled: bool = True

    @property
    def description(self) -> str:
        """Backward-compatible alias used by the original collector."""

        return self.short_name


DEFAULT_SYMBOLS: tuple[SymbolConfig, ...] = (
    SymbolConfig("TUPRS.IS", "equity", "Tupras"),
    SymbolConfig("XU100.IS", "index", "BIST 100"),
    SymbolConfig("XU030.IS", "index", "BIST 30"),
    SymbolConfig("XUSIN.IS", "index", "BIST Industrials"),
    SymbolConfig("TRY=X", "fx", "USD/TRY"),
    SymbolConfig("BZ=F", "future", "Brent crude futures"),
    SymbolConfig("CL=F", "future", "WTI crude futures"),
)


INITIAL_BIST_SYMBOLS: tuple[SymbolConfig, ...] = (
    SymbolConfig("TUPRS.IS", "equity", "Tupras", "XUSIN.IS"),
    SymbolConfig("THYAO.IS", "equity", "Turk Hava Yollari", "XUHIZ.IS"),
    SymbolConfig("ASELS.IS", "equity", "Aselsan", "XUTEK.IS"),
    SymbolConfig("FROTO.IS", "equity", "Ford Otosan", "XUSIN.IS"),
    SymbolConfig("EREGL.IS", "equity", "Eregli Demir Celik", "XUSIN.IS"),
    SymbolConfig("SISE.IS", "equity", "Sisecam", "XUSIN.IS"),
    SymbolConfig("AKBNK.IS", "equity", "Akbank", "XBANK.IS"),
    SymbolConfig("ISCTR.IS", "equity", "Is Bankasi C", "XBANK.IS"),
    SymbolConfig("KCHOL.IS", "equity", "Koc Holding", "XUMAL.IS"),
    SymbolConfig("BIMAS.IS", "equity", "BIM", "XUHIZ.IS"),
    SymbolConfig("TCELL.IS", "equity", "Turkcell", "XUHIZ.IS"),
    SymbolConfig("ENKAI.IS", "equity", "Enka Insaat", "XUSIN.IS"),
)


BENCHMARK_SYMBOLS: tuple[SymbolConfig, ...] = (
    SymbolConfig(MARKET_BENCHMARK, "index", "BIST 100"),
    SymbolConfig("XUSIN.IS", "index", "BIST Industrials"),
    SymbolConfig("XUHIZ.IS", "index", "BIST Services"),
    SymbolConfig("XBANK.IS", "index", "BIST Banks"),
    SymbolConfig("XUMAL.IS", "index", "BIST Financials"),
    SymbolConfig("XUTEK.IS", "index", "BIST Technology"),
)


UNIVERSES: Mapping[str, tuple[SymbolConfig, ...]] = {
    INITIAL_BIST_PANEL: INITIAL_BIST_SYMBOLS,
}


def universe_symbols(name: str) -> tuple[SymbolConfig, ...]:
    try:
        return tuple(config for config in UNIVERSES[name] if config.enabled)
    except KeyError as exc:
        choices = ", ".join(sorted(UNIVERSES))
        raise ValueError(f"Unknown universe {name!r}; available: {choices}") from exc


EQUITY_ASSET_TYPES = {"equity", "stock", "share"}
