from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from collections.abc import Callable

import numpy as np
import pandas as pd

from bist_research.backtest.models import BacktestConfig
from bist_research.backtest.strategy import (
    baseline_entry_signal,
    close_exit_reason,
    prepare_strategy_data,
)


STRATEGY_NAMES = (
    "baseline_v1_fixed",
    "trend_following",
    "breakout",
    "pullback_in_uptrend",
    "relative_strength",
)

FILTER_NAMES = (
    "market_regime",
    "volume",
    "rsi",
    "relative_strength",
    "trend",
    "oil",
    "usdtry",
)


def _finite(row: pd.Series, *columns: str) -> bool:
    return all(column in row and math.isfinite(float(row[column])) for column in columns)


@dataclass(frozen=True)
class ResearchStrategy:
    name: str
    parameters: dict[str, float | int]
    disabled_filters: frozenset[str] = frozenset()

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = prepare_strategy_data(frame)
        fast = int(self.parameters.get("ema_fast", 20))
        medium = int(self.parameters.get("ema_medium", 50))
        long = int(self.parameters.get("ema_long", 200))
        breakout_window = int(self.parameters.get("breakout_window", 20))
        prepared = prepared.copy()
        prepared["research_ema_fast"] = prepared["tuprs_signal_close"].ewm(
            span=fast, adjust=False, min_periods=fast
        ).mean()
        prepared["research_ema_medium"] = prepared["tuprs_signal_close"].ewm(
            span=medium, adjust=False, min_periods=medium
        ).mean()
        prepared["research_ema_long"] = prepared["tuprs_signal_close"].ewm(
            span=long, adjust=False, min_periods=long
        ).mean()
        prepared["research_breakout_high"] = (
            prepared["tuprs_signal_close"]
            .rolling(breakout_window, min_periods=breakout_window)
            .max()
            .shift(1)
        )
        prepared["research_rs_xu100_mean"] = prepared["relative_strength_xu100"].rolling(
            20, min_periods=20
        ).mean()
        prepared["research_rs_xusin_mean"] = prepared["relative_strength_xusin"].rolling(
            20, min_periods=20
        ).mean()
        return prepared

    def backtest_config(self, base: BacktestConfig) -> BacktestConfig:
        return replace(
            base,
            atr_stop_multiplier=float(self.parameters.get("atr_stop", 2.5)),
            atr_trailing_multiplier=float(self.parameters.get("atr_trailing", 3.0)),
            maximum_holding_days=int(self.parameters.get("maximum_holding", 60)),
            minimum_volume_ratio=float(self.parameters.get("volume_ratio", 1.0)),
            minimum_rsi=float(self.parameters.get("rsi_lower", 45.0)),
            maximum_entry_rsi=float(self.parameters.get("rsi_upper", 70.0)),
        )

    def without_filter(self, filter_name: str) -> ResearchStrategy:
        if filter_name not in FILTER_NAMES:
            raise ValueError(f"Unknown filter: {filter_name}")
        return replace(self, disabled_filters=self.disabled_filters | {filter_name})

    def _enabled(self, filter_name: str) -> bool:
        return filter_name not in self.disabled_filters

    def _market_regime(self, row: pd.Series) -> bool:
        return (not self._enabled("market_regime")) or bool(
            _finite(row, "xu100_close", "xu100_ema_200", "xu100_return_1d")
            and row["xu100_close"] > row["xu100_ema_200"]
            and row["xu100_return_1d"] >= -0.03
        )

    def _volume(self, row: pd.Series) -> bool:
        return (not self._enabled("volume")) or bool(
            _finite(row, "volume_ratio_20d", "tuprs_volume")
            and row["volume_ratio_20d"] >= float(self.parameters.get("volume_ratio", 1.0))
            and row["tuprs_volume"] > 0
        )

    def _rsi(self, row: pd.Series) -> bool:
        return (not self._enabled("rsi")) or bool(
            _finite(row, "rsi_14")
            and float(self.parameters.get("rsi_lower", 45))
            <= row["rsi_14"]
            <= float(self.parameters.get("rsi_upper", 70))
        )

    def _trend(self, row: pd.Series) -> bool:
        return (not self._enabled("trend")) or bool(
            _finite(
                row,
                "tuprs_signal_close",
                "research_ema_fast",
                "research_ema_medium",
                "research_ema_long",
            )
            and row["tuprs_signal_close"] > row["research_ema_long"]
            and row["research_ema_fast"] > row["research_ema_medium"]
            and row["research_ema_medium"] >= row["research_ema_long"]
        )

    def _relative_strength(self, row: pd.Series) -> bool:
        return (not self._enabled("relative_strength")) or bool(
            _finite(
                row,
                "relative_strength_xu100",
                "relative_strength_xusin",
                "research_rs_xu100_mean",
                "research_rs_xusin_mean",
                "relative_momentum_20d",
                "relative_momentum_60d",
            )
            and row["relative_strength_xu100"] > row["research_rs_xu100_mean"]
            and row["relative_strength_xusin"] > row["research_rs_xusin_mean"]
            and row["relative_momentum_20d"] > 0
            and row["relative_momentum_60d"] > 0
        )

    def _macro(self, row: pd.Series) -> bool:
        oil_ok = (not self._enabled("oil")) or bool(
            _finite(row, "brent_return_1d") and row["brent_return_1d"] > -0.05
        )
        usdtry_ok = (not self._enabled("usdtry")) or bool(
            _finite(row, "usdtry_return_1d") and row["usdtry_return_1d"] < 0.04
        )
        return oil_ok and usdtry_ok

    def entry_signal(self, row: pd.Series, config: BacktestConfig) -> bool:
        raise NotImplementedError

    def exit_signal(
        self,
        row: pd.Series,
        holding_days: int,
        config: BacktestConfig,
    ) -> str | None:
        raise NotImplementedError

    def _maximum_holding_exit(self, holding_days: int, config: BacktestConfig) -> str | None:
        return "maximum_holding_days" if holding_days >= config.maximum_holding_days else None


@dataclass(frozen=True)
class BaselineV1FixedStrategy(ResearchStrategy):
    def entry_signal(self, row: pd.Series, config: BacktestConfig) -> bool:
        if not self.disabled_filters:
            return baseline_entry_signal(row, config)
        market = (not self._enabled("market_regime")) or bool(
            _finite(row, "xu100_close", "xu100_ema_200", "xu100_return_1d")
            and row["xu100_close"] > row["xu100_ema_200"]
            and row["xu100_return_1d"] >= config.minimum_market_return
        )
        trend = (not self._enabled("trend")) or bool(
            _finite(
                row,
                "tuprs_signal_close",
                "ema_20",
                "ema_50",
                "ema_100",
                "ema_200",
            )
            and row["tuprs_signal_close"] > row["ema_200"]
            and row["ema_20"] > row["ema_50"] > row["ema_100"]
        )
        relative_strength = (not self._enabled("relative_strength")) or bool(
            _finite(
                row,
                "relative_momentum_20d",
                "relative_momentum_60d",
                "relative_strength_xu100",
                "relative_strength_xu100_mean_20d",
            )
            and row["relative_momentum_20d"] > 0
            and row["relative_momentum_60d"] > 0
            and row["relative_strength_xu100"]
            > row["relative_strength_xu100_mean_20d"]
        )
        momentum = (not self._enabled("rsi")) or bool(
            _finite(row, "rsi_14")
            and config.minimum_rsi <= row["rsi_14"] <= config.maximum_entry_rsi
        )
        volume = (not self._enabled("volume")) or bool(
            _finite(row, "volume_ratio_20d", "tuprs_volume")
            and row["volume_ratio_20d"] >= config.minimum_volume_ratio
            and row["tuprs_volume"] > 0
        )
        return bool(
            not bool(row["is_indicator_warmup"])
            and market
            and trend
            and relative_strength
            and momentum
            and volume
        )

    def exit_signal(
        self, row: pd.Series, holding_days: int, config: BacktestConfig
    ) -> str | None:
        return close_exit_reason(row, holding_days, config)


@dataclass(frozen=True)
class TrendFollowingStrategy(ResearchStrategy):
    def entry_signal(self, row: pd.Series, config: BacktestConfig) -> bool:
        del config
        return bool(
            self._market_regime(row)
            and self._trend(row)
            and self._rsi(row)
            and self._volume(row)
            and self._macro(row)
        )

    def exit_signal(
        self, row: pd.Series, holding_days: int, config: BacktestConfig
    ) -> str | None:
        if _finite(row, "tuprs_signal_close", "research_ema_medium") and row[
            "tuprs_signal_close"
        ] < row["research_ema_medium"]:
            return "trend_exit"
        return self._maximum_holding_exit(holding_days, config)


@dataclass(frozen=True)
class BreakoutStrategy(ResearchStrategy):
    def entry_signal(self, row: pd.Series, config: BacktestConfig) -> bool:
        del config
        breakout = bool(
            _finite(row, "tuprs_signal_close", "research_breakout_high")
            and row["tuprs_signal_close"] > row["research_breakout_high"]
        )
        return bool(
            breakout
            and self._market_regime(row)
            and self._trend(row)
            and self._volume(row)
            and self._macro(row)
        )

    def exit_signal(
        self, row: pd.Series, holding_days: int, config: BacktestConfig
    ) -> str | None:
        if _finite(row, "tuprs_signal_close", "research_ema_fast") and row[
            "tuprs_signal_close"
        ] < row["research_ema_fast"]:
            return "breakout_failure_exit"
        return self._maximum_holding_exit(holding_days, config)


@dataclass(frozen=True)
class PullbackInUptrendStrategy(ResearchStrategy):
    def entry_signal(self, row: pd.Series, config: BacktestConfig) -> bool:
        del config
        pullback = bool(
            _finite(row, "tuprs_signal_close", "research_ema_fast", "rsi_14")
            and row["tuprs_signal_close"] <= row["research_ema_fast"] * 1.015
            and row["rsi_14"] <= float(self.parameters.get("rsi_upper", 70))
        )
        return bool(
            pullback
            and self._market_regime(row)
            and self._trend(row)
            and self._rsi(row)
            and self._volume(row)
            and self._macro(row)
        )

    def exit_signal(
        self, row: pd.Series, holding_days: int, config: BacktestConfig
    ) -> str | None:
        if _finite(row, "tuprs_signal_close", "research_ema_long") and row[
            "tuprs_signal_close"
        ] < row["research_ema_long"]:
            return "trend_exit"
        if _finite(row, "rsi_14") and row["rsi_14"] > min(
            float(self.parameters.get("rsi_upper", 70)) + 10, 90
        ):
            return "rsi_exit"
        return self._maximum_holding_exit(holding_days, config)


@dataclass(frozen=True)
class RelativeStrengthStrategy(ResearchStrategy):
    def entry_signal(self, row: pd.Series, config: BacktestConfig) -> bool:
        del config
        return bool(
            self._market_regime(row)
            and self._relative_strength(row)
            and self._trend(row)
            and self._rsi(row)
            and self._volume(row)
            and self._macro(row)
        )

    def exit_signal(
        self, row: pd.Series, holding_days: int, config: BacktestConfig
    ) -> str | None:
        if _finite(row, "relative_momentum_20d") and row["relative_momentum_20d"] < 0:
            return "relative_strength_exit"
        if _finite(row, "tuprs_signal_close", "research_ema_medium") and row[
            "tuprs_signal_close"
        ] < row["research_ema_medium"]:
            return "trend_exit"
        return self._maximum_holding_exit(holding_days, config)


STRATEGY_CLASSES: dict[str, type[ResearchStrategy]] = {
    "baseline_v1_fixed": BaselineV1FixedStrategy,
    "trend_following": TrendFollowingStrategy,
    "breakout": BreakoutStrategy,
    "pullback_in_uptrend": PullbackInUptrendStrategy,
    "relative_strength": RelativeStrengthStrategy,
}


def build_strategy(
    name: str,
    parameters: dict[str, float | int],
    disabled_filters: frozenset[str] = frozenset(),
) -> ResearchStrategy:
    try:
        strategy_class = STRATEGY_CLASSES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc
    return strategy_class(name=name, parameters=parameters, disabled_filters=disabled_filters)


def deterministic_signal_filter(
    seed: int,
    skip_fraction: float = 0.10,
) -> Callable[[pd.Series], bool]:
    if not 0 <= skip_fraction < 1:
        raise ValueError("skip_fraction must be in [0, 1)")

    def keep(row: pd.Series) -> bool:
        date = pd.Timestamp(row["date"]).date().isoformat()
        digest = hashlib.sha256(f"{seed}:{date}".encode("ascii")).digest()
        draw = int.from_bytes(digest[:8], "big") / (2**64 - 1)
        return draw >= skip_fraction

    return keep


def _mutate_future(frame: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    changed = frame.copy()
    future = changed.index[cutoff + 1 :]
    price_columns = [
        column
        for column in (
            "tuprs_open",
            "tuprs_high",
            "tuprs_low",
            "tuprs_close",
            "tuprs_signal_open",
            "tuprs_signal_high",
            "tuprs_signal_low",
            "tuprs_signal_close",
            "tuprs_adj_close",
        )
        if column in changed
    ]
    if price_columns:
        changed.loc[future, price_columns] = changed.loc[future, price_columns] * 1.73
    for column in (
        "tuprs_volume",
        "xu100_open",
        "xu100_high",
        "xu100_low",
        "xu100_close",
        "xu100_adj_close",
        "xusin_open",
        "xusin_high",
        "xusin_low",
        "xusin_close",
        "xusin_adj_close",
        "usdtry_open",
        "usdtry_high",
        "usdtry_low",
        "usdtry_close",
        "usdtry_adj_close",
        "brent_open",
        "brent_high",
        "brent_low",
        "brent_close",
        "brent_adj_close",
        "wti_open",
        "wti_high",
        "wti_low",
        "wti_close",
        "wti_adj_close",
    ):
        if column in changed:
            changed[column] = pd.to_numeric(changed[column], errors="coerce").astype(float)
            changed.loc[future, column] = changed.loc[future, column] * 2.31
    if "dividend_per_share" in changed:
        changed.loc[future, "dividend_per_share"] = (
            changed.loc[future, "dividend_per_share"].astype(float) + 7.0
        )
    if "stock_split_factor" in changed:
        changed.loc[future, "stock_split_factor"] = 3.0
    if "corporate_action_flag" in changed:
        changed.loc[future, "corporate_action_flag"] = True
    return changed


def strong_causality_audit(
    strategy: ResearchStrategy,
    frame: pd.DataFrame,
    *,
    random_seed: int = 20240801,
    cutpoint_count: int = 3,
) -> bool:
    if len(frame) < 3:
        return True
    rng = np.random.default_rng(random_seed)
    first_active = 1
    if "is_indicator_warmup" in frame:
        active_indices = np.flatnonzero(~frame["is_indicator_warmup"].astype(bool).to_numpy())
        if len(active_indices):
            first_active = int(active_indices[0]) + 1
    candidates = np.arange(max(1, len(frame) // 5, first_active), len(frame) - 1)
    if not len(candidates):
        return True
    count = min(cutpoint_count, len(candidates))
    cutpoints = sorted(int(value) for value in rng.choice(candidates, count, replace=False))
    full_engine_audit = {
        "date",
        "tuprs_open",
        "tuprs_high",
        "tuprs_low",
        "tuprs_close",
        "tuprs_signal_open",
        "tuprs_signal_high",
        "tuprs_signal_low",
        "tuprs_signal_close",
        "tuprs_volume",
        "is_indicator_warmup",
    }.issubset(frame.columns)

    for cutoff in cutpoints:
        audit_start = max(0, cutoff - 260)
        audit_end = min(len(frame), cutoff + 81)
        audit_frame = frame.iloc[audit_start:audit_end].reset_index(drop=True)
        local_cutoff = cutoff - audit_start
        changed = _mutate_future(audit_frame, local_cutoff)
        original_prepared = strategy.prepare(audit_frame)
        changed_prepared = strategy.prepare(changed)
        try:
            pd.testing.assert_frame_equal(
                original_prepared.iloc[: local_cutoff + 1].reset_index(drop=True),
                changed_prepared.iloc[: local_cutoff + 1].reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError:
            return False
        if not full_engine_audit:
            continue

        config = strategy.backtest_config(BacktestConfig())
        for index in range(local_cutoff + 1):
            original_row = original_prepared.iloc[index]
            changed_row = changed_prepared.iloc[index]
            if strategy.entry_signal(original_row, config) != strategy.entry_signal(
                changed_row,
                config,
            ):
                return False
            holding_days = index % max(config.maximum_holding_days, 1) + 1
            if strategy.exit_signal(
                original_row,
                holding_days,
                config,
            ) != strategy.exit_signal(changed_row, holding_days, config):
                return False

        from bist_research.backtest.engine import BacktestEngine

        engine = BacktestEngine(
            config=config,
            entry_signal=strategy.entry_signal,
            exit_signal=strategy.exit_signal,
        )
        original_result = engine.run(original_prepared)
        changed_result = engine.run(changed_prepared)
        cutoff_date = pd.Timestamp(original_prepared.iloc[local_cutoff]["date"])
        original_equity = original_result.daily_equity.loc[
            original_result.daily_equity["date"].le(cutoff_date)
        ].reset_index(drop=True)
        changed_equity = changed_result.daily_equity.loc[
            changed_result.daily_equity["date"].le(cutoff_date)
        ].reset_index(drop=True)
        original_trades = original_result.trades.loc[
            pd.to_datetime(original_result.trades["exit_date"]).le(cutoff_date)
        ].reset_index(drop=True)
        changed_trades = changed_result.trades.loc[
            pd.to_datetime(changed_result.trades["exit_date"]).le(cutoff_date)
        ].reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(original_equity, changed_equity, check_exact=True)
            pd.testing.assert_frame_equal(original_trades, changed_trades, check_exact=True)
        except AssertionError:
            return False
    return True


def strategy_is_causal(strategy: ResearchStrategy, frame: pd.DataFrame) -> bool:
    return strong_causality_audit(strategy, frame)


def scale_parameters(
    parameters: dict[str, float | int],
    factor: float,
) -> dict[str, float | int]:
    scaled: dict[str, float | int] = {}
    integer_keys = {"ema_fast", "ema_medium", "ema_long", "breakout_window", "maximum_holding"}
    for key, value in parameters.items():
        if key in integer_keys:
            scaled[key] = max(2, int(round(float(value) * factor)))
        else:
            scaled[key] = round(float(value) * factor, 6)
    if "rsi_lower" in scaled and "rsi_upper" in scaled:
        scaled["rsi_lower"] = min(float(scaled["rsi_lower"]), 80.0)
        scaled["rsi_upper"] = min(max(float(scaled["rsi_upper"]), float(scaled["rsi_lower"])), 95.0)
    return scaled
