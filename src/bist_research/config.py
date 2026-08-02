from dataclasses import dataclass


START_DATE = "2013-01-01"


@dataclass(frozen=True)
class SymbolConfig:
    symbol: str
    asset_type: str
    description: str


DEFAULT_SYMBOLS: tuple[SymbolConfig, ...] = (
    SymbolConfig("TUPRS.IS", "equity", "Tupras"),
    SymbolConfig("XU100.IS", "index", "BIST 100"),
    SymbolConfig("XU030.IS", "index", "BIST 30"),
    SymbolConfig("XUSIN.IS", "index", "BIST Industrials"),
    SymbolConfig("TRY=X", "fx", "USD/TRY"),
    SymbolConfig("BZ=F", "future", "Brent crude futures"),
    SymbolConfig("CL=F", "future", "WTI crude futures"),
)


EQUITY_ASSET_TYPES = {"equity", "stock", "share"}
