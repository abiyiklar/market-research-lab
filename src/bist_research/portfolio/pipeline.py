from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from bist_research.config import SymbolConfig
from bist_research.symbols import safe_symbol_name

from .analytics import (
    build_calendar_year_contribution,
    build_contribution_tables,
    build_period_tables,
    build_research_gates,
    calculate_performance_metrics,
)
from .benchmarks import build_equal_weight_benchmark, build_xu100_benchmark
from .engine import PortfolioBacktestEngine, build_portfolio_context
from .models import PortfolioConfig
from .report import build_portfolio_report, write_portfolio_report


@dataclass(frozen=True)
class PortfolioPipelineOutputs:
    output_dir: Path
    report_path: Path
    performance_summary: pd.DataFrame
    stress_summary: pd.DataFrame
    research_gates: pd.DataFrame
    completed_trades: int
    open_positions: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp


def load_portfolio_data(
    configs: Sequence[SymbolConfig],
    panel_path: Path = Path("data/signals/panel_daily_scores.parquet"),
    feature_dir: Path = Path("data/features/equities"),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if not panel_path.exists():
        raise FileNotFoundError(f"Signal panel not found: {panel_path}")
    panel = pd.read_parquet(panel_path)
    features: dict[str, pd.DataFrame] = {}
    for config in configs:
        path = feature_dir / safe_symbol_name(config.symbol) / "features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        features[config.symbol] = pd.read_parquet(path)
    return panel, features


def _stress_configs(primary: PortfolioConfig) -> list[tuple[str, PortfolioConfig]]:
    return [
        ("primary", primary),
        (
            "max_positions_3",
            PortfolioConfig(
                primary.starting_capital,
                3,
                primary.target_weight,
                primary.transaction_cost_rate,
                0,
            ),
        ),
        (
            "max_positions_5",
            PortfolioConfig(
                primary.starting_capital,
                5,
                primary.target_weight,
                primary.transaction_cost_rate,
                0,
            ),
        ),
        *[
            (
                f"cost_{cost:.4f}",
                PortfolioConfig(
                    primary.starting_capital,
                    4,
                    primary.target_weight,
                    cost,
                    0,
                ),
            )
            for cost in (0.0, 0.003, 0.005)
        ],
        (
            "execution_delay_1",
            PortfolioConfig(
                primary.starting_capital,
                4,
                primary.target_weight,
                primary.transaction_cost_rate,
                1,
            ),
        ),
    ]


def run_portfolio_pipeline(
    configs: Sequence[SymbolConfig],
    panel_path: Path = Path("data/signals/panel_daily_scores.parquet"),
    feature_dir: Path = Path("data/features/equities"),
    output_dir: Path = Path("data/portfolio"),
    report_path: Path = Path("reports/portfolio_backtest_research.md"),
    config: PortfolioConfig | None = None,
) -> PortfolioPipelineOutputs:
    primary_config = config or PortfolioConfig()
    source_panel, source_features = load_portfolio_data(configs, panel_path, feature_dir)
    context = build_portfolio_context(source_panel, source_features, configs)
    panel, features = context.panel, context.features
    primary = PortfolioBacktestEngine(primary_config).run_prepared(context)

    xu100 = build_xu100_benchmark(primary.daily["date"], panel, primary_config)
    equal_weight = build_equal_weight_benchmark(
        primary.daily["date"], features, configs, primary_config
    )
    primary.daily["benchmark_value"] = xu100["total_equity"].to_numpy()
    primary.daily["benchmark_relative_equity"] = (
        primary.daily["total_portfolio_equity"] / primary.daily["benchmark_value"]
    )

    filled_orders = primary.orders.loc[primary.orders["order_status"].eq("filled")]
    gross_traded = float(filled_orders["gross_transaction_value"].sum())
    primary_metrics = calculate_performance_metrics(
        "primary_portfolio",
        primary.daily,
        primary_config,
        primary.trades,
        primary.open_positions,
        gross_traded_value=gross_traded,
    )
    xu100_metrics = calculate_performance_metrics(
        "xu100_price_index",
        xu100,
        primary_config,
        gross_traded_value=primary_config.starting_capital,
        total_transaction_costs=float(xu100["cumulative_transaction_costs"].iloc[-1]),
        total_dividends=0.0,
    )
    equal_metrics = calculate_performance_metrics(
        "equal_weight_12_stock",
        equal_weight,
        primary_config,
        gross_traded_value=primary_config.starting_capital - float(equal_weight["cash"].iloc[0]),
        total_transaction_costs=float(equal_weight["cumulative_transaction_costs"].iloc[-1]),
        total_dividends=float(equal_weight["cumulative_dividends"].iloc[-1]),
    )
    performance = pd.DataFrame([primary_metrics, xu100_metrics, equal_metrics])

    yearly, monthly, rolling, rolling_summary = build_period_tables(
        primary.daily,
        {"xu100": xu100, "equal_weight_12_stock": equal_weight},
    )
    calendar_year_contribution = build_calendar_year_contribution(primary)
    (
        symbol_contribution,
        signal_type_contribution,
        exit_reason_contribution,
        score_bucket_contribution,
        sector_fallback_contribution,
        concentration,
    ) = build_contribution_tables(primary)

    stress_rows: list[dict[str, object]] = []
    for scenario, scenario_config in _stress_configs(primary_config):
        if scenario == "primary":
            scenario_result = primary
            metrics = primary_metrics
        else:
            scenario_result = PortfolioBacktestEngine(scenario_config).run_prepared(
                context
            )
            scenario_filled = scenario_result.orders.loc[
                scenario_result.orders["order_status"].eq("filled")
            ]
            metrics = calculate_performance_metrics(
                scenario,
                scenario_result.daily,
                scenario_config,
                scenario_result.trades,
                scenario_result.open_positions,
                gross_traded_value=float(
                    scenario_filled["gross_transaction_value"].sum()
                ),
            )
        stress_rows.append(
            {
                "scenario": scenario,
                "max_positions": scenario_config.max_positions,
                "transaction_cost_rate": scenario_config.transaction_cost_rate,
                "execution_delay_sessions": scenario_config.execution_delay_sessions,
                "ending_value": metrics["ending_value"],
                "total_return": metrics["total_return"],
                "cagr": metrics["cagr"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "completed_trades": metrics["completed_trades"],
                "total_transaction_costs": metrics["total_transaction_costs"],
                "total_dividends_received": metrics["total_dividends_received"],
            }
        )
    stress = pd.DataFrame(stress_rows)
    gates = build_research_gates(
        primary_metrics,
        xu100_metrics,
        rolling_summary,
        concentration,
        stress,
        yearly,
    )

    performance.loc[performance["portfolio"].eq("primary_portfolio"), list(rolling_summary)] = list(
        rolling_summary.values()
    )
    performance.loc[performance["portfolio"].eq("primary_portfolio"), list(concentration)] = list(
        concentration.values()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    primary.daily.to_csv(output_dir / "portfolio_daily.csv", index=False)
    primary.daily.to_parquet(output_dir / "portfolio_daily.parquet", index=False)
    primary.orders.to_csv(output_dir / "orders.csv", index=False)
    primary.trades.to_csv(output_dir / "trades.csv", index=False)
    primary.open_positions.to_csv(output_dir / "open_positions.csv", index=False)
    primary.position_daily_audit.to_csv(
        output_dir / "position_daily_audit.csv", index=False
    )
    performance.to_csv(output_dir / "performance_summary.csv", index=False)
    yearly.to_csv(output_dir / "yearly_returns.csv", index=False)
    calendar_year_contribution.to_csv(
        output_dir / "calendar_year_contribution.csv", index=False
    )
    monthly.to_csv(output_dir / "monthly_returns.csv", index=False)
    rolling.to_csv(output_dir / "rolling_metrics.csv", index=False)
    performance.to_csv(output_dir / "benchmark_comparison.csv", index=False)
    symbol_contribution.to_csv(output_dir / "symbol_contribution.csv", index=False)
    signal_type_contribution.to_csv(
        output_dir / "signal_type_contribution.csv", index=False
    )
    exit_reason_contribution.to_csv(
        output_dir / "exit_reason_contribution.csv", index=False
    )
    score_bucket_contribution.to_csv(
        output_dir / "score_bucket_contribution.csv", index=False
    )
    sector_fallback_contribution.to_csv(
        output_dir / "sector_fallback_contribution.csv", index=False
    )
    stress.to_csv(output_dir / "stress_test_summary.csv", index=False)
    gates.to_csv(output_dir / "research_gates.csv", index=False)
    primary.orders.loc[primary.orders["order_status"].ne("filled")].to_csv(
        output_dir / "rejected_orders.csv", index=False
    )

    report = build_portfolio_report(
        performance,
        yearly,
        rolling_summary,
        symbol_contribution,
        stress,
        gates,
        concentration,
        primary.orders,
        primary.trades,
        primary.open_positions,
        primary_config.max_positions,
        primary_config.transaction_cost_rate,
    )
    write_portfolio_report(report, report_path)
    return PortfolioPipelineOutputs(
        output_dir=output_dir,
        report_path=report_path,
        performance_summary=performance,
        stress_summary=stress,
        research_gates=gates,
        completed_trades=len(primary.trades),
        open_positions=len(primary.open_positions),
        start_date=primary.start_date,
        end_date=primary.end_date,
    )
