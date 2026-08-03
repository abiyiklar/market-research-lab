from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from bist_research.config import SymbolConfig


MINIMUM_CROSS_SECTION = 8
TRADING_DAYS_PER_YEAR = 252
RISK_FLOOR = 1e-8

MARKET_SCORE_INPUTS = (
    "relative_momentum_market_20",
    "relative_momentum_market_60",
    "relative_momentum_market_120",
    "market_relative_strength_change_20",
)
SECTOR_SCORE_INPUTS = (
    "relative_momentum_sector_20",
    "relative_momentum_sector_60",
    "relative_momentum_sector_120",
)
MOMENTUM_SCORE_INPUTS = (
    "momentum_return_20",
    "momentum_return_60",
    "momentum_return_120",
    "momentum_return_252_skip_21",
    "distance_from_prior_252_high",
)
TREND_SCORE_INPUTS = (
    "trend_above_ema_count",
    "trend_ema_alignment",
    "ema_20_slope_10",
    "ema_50_slope_20",
    "ema_20_distance_quality",
    "ema_50_distance_quality",
    "trend_consistency_20",
)
VOLUME_SCORE_INPUTS = (
    "volume_ratio_20_capped",
    "positive_negative_volume_ratio_20",
    "volume_progress_support_20",
)
RISK_SCORE_INPUTS = (
    "volatility_20",
    "volatility_60",
    "atr_pct_signal_close",
    "downside_volatility_20",
)

COMPONENT_COLUMNS = (
    "market_relative_strength_score",
    "sector_relative_strength_score",
    "momentum_score",
    "trend_quality_score",
    "volume_confirmation_score",
    "risk_quality_score",
)

REQUIRED_SOURCE_COLUMNS = {
    "symbol",
    "date",
    "signal_open",
    "signal_high",
    "signal_low",
    "signal_close",
    "volume",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",
    "rsi_14",
    "atr_14",
    "volatility_20",
    "volatility_60",
    "volume_ratio_20",
    "relative_strength_market",
    "relative_momentum_market_20",
    "relative_momentum_market_60",
    "relative_momentum_market_120",
    "relative_momentum_sector_20",
    "relative_momentum_sector_60",
    "relative_momentum_sector_120",
    "market_close",
    "sector_close",
    "market_benchmark",
    "sector_benchmark",
    "is_indicator_warmup",
    "quality_valid_ohlc",
    "quality_positive_volume",
    "quality_sector_fallback",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    usable = denominator.where(denominator.abs().gt(RISK_FLOOR))
    return numerator / usable


def _finite_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return pd.Series(
        np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1),
        index=frame.index,
    )


def prepare_symbol_inputs(config: SymbolConfig, source: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_SOURCE_COLUMNS.difference(source.columns))
    if missing:
        raise ValueError(f"{config.symbol} feature data is missing columns: {', '.join(missing)}")

    frame = source.copy()
    observed = set(frame["symbol"].dropna().astype(str).unique())
    if observed.difference({config.symbol}):
        raise ValueError(f"Cross-symbol rows found for {config.symbol}: {sorted(observed)}")
    if "source_symbol" in frame:
        source_symbols = set(frame["source_symbol"].dropna().astype(str).unique())
        if source_symbols.difference({config.symbol}):
            raise ValueError(
                f"Cross-symbol source data found for {config.symbol}: {sorted(source_symbols)}"
            )

    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.tz_localize(None).dt.normalize()
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError(f"{config.symbol} feature data contains duplicate dates")
    frame["_stock_session_index"] = np.arange(len(frame), dtype="int64")

    close = pd.to_numeric(frame["signal_close"], errors="coerce")
    high = pd.to_numeric(frame["signal_high"], errors="coerce")
    atr = pd.to_numeric(frame["atr_14"], errors="coerce")
    returns = close.pct_change(fill_method=None)

    frame["market_relative_strength_change_20"] = pd.to_numeric(
        frame["relative_strength_market"], errors="coerce"
    ).pct_change(20, fill_method=None)
    for periods in (20, 60, 120):
        frame[f"momentum_return_{periods}"] = close.pct_change(periods, fill_method=None)
    frame["momentum_return_252_skip_21"] = close.shift(21) / close.shift(252) - 1.0
    frame["prior_252_high"] = high.rolling(252, min_periods=252).max().shift(1)
    frame["distance_from_prior_252_high"] = close / frame["prior_252_high"] - 1.0

    for periods in (20, 50, 100, 200):
        frame[f"above_ema_{periods}"] = close.gt(pd.to_numeric(frame[f"ema_{periods}"], errors="coerce"))
    frame["trend_above_ema_count"] = frame[
        [f"above_ema_{periods}" for periods in (20, 50, 100, 200)]
    ].sum(axis=1)
    frame["trend_ema_alignment"] = (
        frame["ema_20"].gt(frame["ema_50"])
        & frame["ema_50"].gt(frame["ema_100"])
        & frame["ema_100"].gt(frame["ema_200"])
    ).astype(float)
    frame["ema_20_slope_10"] = pd.to_numeric(frame["ema_20"], errors="coerce").pct_change(
        10, fill_method=None
    )
    frame["ema_50_slope_20"] = pd.to_numeric(frame["ema_50"], errors="coerce").pct_change(
        20, fill_method=None
    )
    frame["atr_pct_signal_close"] = _safe_divide(atr, close)
    frame["signal_distance_ema_20_atr"] = _safe_divide(close - frame["ema_20"], atr)
    frame["signal_distance_ema_50_atr"] = _safe_divide(close - frame["ema_50"], atr)
    frame["ema_20_distance_quality"] = -(frame["signal_distance_ema_20_atr"] - 0.75).abs()
    frame["ema_50_distance_quality"] = -(frame["signal_distance_ema_50_atr"] - 1.00).abs()

    positive_ratio = returns.gt(0).rolling(20, min_periods=20).mean()
    path = returns.abs().rolling(20, min_periods=20).sum()
    largest_step = returns.abs().rolling(20, min_periods=20).max()
    concentration_penalty = _safe_divide(largest_step, path).clip(0.0, 1.0)
    frame["positive_session_ratio_20"] = positive_ratio
    frame["trend_return_concentration_20"] = concentration_penalty
    frame["trend_consistency_20"] = positive_ratio * (1.0 - concentration_penalty)

    volume = pd.to_numeric(frame["volume"], errors="coerce")
    frame["volume_ratio_20_capped"] = pd.to_numeric(
        frame["volume_ratio_20"], errors="coerce"
    ).clip(lower=0.0, upper=3.0)
    positive_volume = volume.where(returns.gt(0), 0.0).rolling(20, min_periods=20).sum()
    negative_volume = volume.where(returns.lt(0), 0.0).rolling(20, min_periods=20).sum()
    volume_ratio = _safe_divide(positive_volume, negative_volume)
    volume_ratio = volume_ratio.mask(negative_volume.eq(0) & positive_volume.gt(0), 5.0)
    frame["positive_negative_volume_ratio_20"] = volume_ratio.clip(lower=0.0, upper=5.0)
    directional_volume = volume.where(returns.ne(0), 0.0).rolling(20, min_periods=20).sum()
    frame["volume_progress_support_20"] = _safe_divide(positive_volume, directional_volume)

    downside_returns = returns.clip(upper=0.0)
    frame["downside_volatility_20"] = (
        downside_returns.rolling(20, min_periods=20).std() * (TRADING_DAYS_PER_YEAR**0.5)
    )
    risk_values = frame.loc[:, RISK_SCORE_INPUTS].apply(pd.to_numeric, errors="coerce")
    core_risk_values = frame.loc[
        :, ["volatility_20", "volatility_60", "atr_pct_signal_close"]
    ].apply(pd.to_numeric, errors="coerce")
    frame["risk_data_quality_problem"] = (
        ~np.isfinite(risk_values.to_numpy(dtype="float64")).all(axis=1)
        | core_risk_values.le(RISK_FLOOR).any(axis=1)
    )

    sector_fallback = frame["quality_sector_fallback"].fillna(False).astype(bool)
    same_benchmark = frame["sector_benchmark"].astype(str).eq(frame["market_benchmark"].astype(str))
    sector_missing = frame.get(
        "quality_sector_missing",
        pd.Series(False, index=frame.index),
    ).fillna(False).astype(bool)
    frame["sector_score_available"] = ~(sector_fallback | same_benchmark | sector_missing)
    frame["sector_fallback_neutralized"] = ~frame["sector_score_available"]

    always_required = [
        *MARKET_SCORE_INPUTS,
        *MOMENTUM_SCORE_INPUTS,
        *TREND_SCORE_INPUTS,
        *VOLUME_SCORE_INPUTS,
        *RISK_SCORE_INPUTS,
    ]
    valid = (
        ~frame["is_indicator_warmup"].fillna(True).astype(bool)
        & _finite_columns(frame, always_required)
        & ~frame["risk_data_quality_problem"]
    )
    sector_valid = _finite_columns(frame, SECTOR_SCORE_INPUTS)
    valid &= ~frame["sector_score_available"] | sector_valid
    signal_ohlc = frame.loc[
        :, ["signal_open", "signal_high", "signal_low", "signal_close"]
    ].apply(pd.to_numeric, errors="coerce")
    valid &= np.isfinite(signal_ohlc.to_numpy(dtype="float64")).all(axis=1)
    valid &= signal_ohlc.gt(0).all(axis=1) & atr.gt(0) & volume.gt(0)
    valid &= frame["quality_valid_ohlc"].fillna(False).astype(bool)
    valid &= frame["quality_positive_volume"].fillna(False).astype(bool)
    valid &= ~frame.get(
        "quality_market_missing",
        pd.Series(False, index=frame.index),
    ).fillna(False).astype(bool)
    frame["valid_score_input"] = valid
    return frame


def _cross_sectional_mean_rank(
    panel: pd.DataFrame,
    eligible: pd.Series,
    columns: Sequence[str],
    higher_is_better: bool = True,
) -> pd.Series:
    result = pd.Series(np.nan, index=panel.index, dtype="float64")
    if not eligible.any():
        return result

    ranked_inputs: dict[str, pd.Series] = {}
    dates = panel.loc[eligible, "date"]
    for column in columns:
        values = pd.to_numeric(panel.loc[eligible, column], errors="coerce")
        grouped = values.groupby(dates, sort=False)
        ranks = grouped.rank(method="average", ascending=higher_is_better)
        counts = grouped.transform("count")
        ranked_inputs[column] = ((ranks - 1.0) / (counts - 1.0) * 100.0).where(
            counts.gt(1),
            50.0,
        )
    result.loc[eligible] = pd.DataFrame(ranked_inputs).mean(axis=1)
    return result


def _score_state(score: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                score.ge(75),
                score.ge(65),
                score.ge(50),
                score.ge(40),
                score.notna(),
            ],
            ["STRONG", "POSITIVE", "NEUTRAL", "WEAKENING", "WEAK"],
            default="UNSCORED",
        ),
        index=score.index,
        dtype="string",
    )


def build_score_panel(
    feature_frames: Mapping[str, pd.DataFrame],
    configs: Sequence[SymbolConfig],
    minimum_cross_section: int = MINIMUM_CROSS_SECTION,
) -> pd.DataFrame:
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    prepared: list[pd.DataFrame] = []
    for config in configs:
        if config.symbol not in feature_frames:
            continue
        prepared.append(prepare_symbol_inputs(config, feature_frames[config.symbol]))
    if not prepared:
        return pd.DataFrame()

    panel = pd.concat(prepared, ignore_index=True, sort=False)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    if panel.duplicated(["symbol", "date"]).any():
        raise ValueError("Panel contains duplicate symbol/date rows")
    valid = panel["valid_score_input"].fillna(False).astype(bool)
    panel["cross_section_size"] = valid.groupby(panel["date"], sort=False).transform("sum").astype(
        "int64"
    )
    panel["insufficient_cross_section"] = panel["cross_section_size"].lt(
        minimum_cross_section
    )
    eligible = valid & ~panel["insufficient_cross_section"]

    panel["market_relative_strength_score"] = _cross_sectional_mean_rank(
        panel, eligible, MARKET_SCORE_INPUTS
    )
    sector_eligible = eligible & panel["sector_score_available"]
    panel["sector_relative_strength_score"] = np.nan
    panel.loc[eligible, "sector_relative_strength_score"] = 50.0
    panel.loc[sector_eligible, "sector_relative_strength_score"] = _cross_sectional_mean_rank(
        panel,
        sector_eligible,
        SECTOR_SCORE_INPUTS,
    ).loc[sector_eligible]
    panel["momentum_score"] = _cross_sectional_mean_rank(panel, eligible, MOMENTUM_SCORE_INPUTS)
    panel["trend_quality_score"] = _cross_sectional_mean_rank(panel, eligible, TREND_SCORE_INPUTS)
    panel["volume_confirmation_score"] = _cross_sectional_mean_rank(
        panel, eligible, VOLUME_SCORE_INPUTS
    )
    panel["risk_quality_score"] = _cross_sectional_mean_rank(
        panel,
        eligible,
        RISK_SCORE_INPUTS,
        higher_is_better=False,
    )
    panel["final_score"] = (
        0.25 * panel["market_relative_strength_score"]
        + 0.15 * panel["sector_relative_strength_score"]
        + 0.20 * panel["momentum_score"]
        + 0.20 * panel["trend_quality_score"]
        + 0.10 * panel["volume_confirmation_score"]
        + 0.10 * panel["risk_quality_score"]
    ).clip(0.0, 100.0)
    panel["cross_section_rank"] = np.nan
    panel.loc[eligible, "cross_section_rank"] = panel.loc[eligible, "final_score"].groupby(
        panel.loc[eligible, "date"],
        sort=False,
    ).rank(method="min", ascending=False)
    panel["score_state"] = _score_state(panel["final_score"])
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)
