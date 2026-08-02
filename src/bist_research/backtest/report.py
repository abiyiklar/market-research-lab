from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import BacktestEngine, load_feature_data
from .corporate_actions import audit_corporate_actions, raw_strategy_prices_are_consistent
from .metrics import (
    BenchmarkResult,
    benchmark_metrics,
    calculate_drawdowns,
    calculate_metrics,
    calculate_tuprs_adjusted_total_return,
    calculate_tuprs_buy_hold,
    calculate_xu100_buy_hold,
)
from .models import BacktestConfig, BacktestResult, PERIOD_DEFINITIONS, PeriodDefinition
from .strategy import prepare_strategy_data


@dataclass(frozen=True)
class BacktestArtifacts:
    trades: Path
    daily_equity: Path
    metrics: Path
    period_metrics: Path
    benchmark: Path
    drawdowns: Path
    data_quality: Path
    markdown_report: Path
    equity_curve: Path
    drawdown_chart: Path
    annual_returns: Path
    trade_returns: Path
    strategy_vs_benchmark: Path


@dataclass(frozen=True)
class BacktestPipelineResult:
    artifacts: BacktestArtifacts
    metrics: dict[str, object]
    benchmark_table: pd.DataFrame


def slice_period(frame: pd.DataFrame, period: PeriodDefinition) -> pd.DataFrame:
    mask = frame["date"].ge(period.start)
    if period.end is not None:
        mask &= frame["date"].le(period.end)
    return frame.loc[mask].reset_index(drop=True)


def _comparison_row(
    period: str,
    name: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    columns = (
        "start_date",
        "end_date",
        "initial_capital",
        "ending_capital",
        "net_profit",
        "total_return",
        "cagr",
        "maximum_drawdown",
        "sharpe_ratio",
        "annual_volatility",
        "total_commission",
        "total_slippage_cost",
    )
    return {"period": period, "benchmark": name, **{column: metrics[column] for column in columns}}


def _run_period(
    frame: pd.DataFrame,
    config: BacktestConfig,
    logger: logging.Logger,
) -> tuple[
    BacktestResult,
    dict[str, object],
    BenchmarkResult,
    BenchmarkResult,
    BenchmarkResult,
]:
    result = BacktestEngine(config=config, logger=logger).run(frame)
    metrics = calculate_metrics(result.daily_equity, result.trades, config)
    tuprs_benchmark = calculate_tuprs_buy_hold(frame, config)
    adjusted_tuprs_benchmark = calculate_tuprs_adjusted_total_return(frame, config)
    xu100_benchmark = calculate_xu100_buy_hold(frame, config)
    return result, metrics, tuprs_benchmark, adjusted_tuprs_benchmark, xu100_benchmark


def run_backtest_pipeline(
    input_path: Path = Path("data/features/tuprs_features.parquet"),
    output_dir: Path = Path("data/backtest"),
    report_dir: Path = Path("reports"),
    config: BacktestConfig | None = None,
    logger: logging.Logger | None = None,
) -> BacktestPipelineResult:
    active_config = config or BacktestConfig()
    active_logger = logger or logging.getLogger("bist_research.backtest")
    features = prepare_strategy_data(load_feature_data(input_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    full_result, full_metrics, full_tuprs, full_adjusted_tuprs, full_xu100 = _run_period(
        features,
        active_config,
        active_logger,
    )
    period_rows: list[dict[str, object]] = []
    comparison_rows = [_comparison_row("all", "baseline_v1", full_metrics)]
    comparison_rows.append(
        _comparison_row("all", full_tuprs.name, benchmark_metrics(full_tuprs, active_config))
    )
    comparison_rows.append(
        _comparison_row(
            "all",
            full_adjusted_tuprs.name,
            benchmark_metrics(full_adjusted_tuprs, active_config),
        )
    )
    comparison_rows.append(
        _comparison_row("all", full_xu100.name, benchmark_metrics(full_xu100, active_config))
    )

    for period in PERIOD_DEFINITIONS:
        period_frame = slice_period(features, period)
        if period_frame.empty:
            raise ValueError(f"No data available for the {period.name} period")
        _, metrics, tuprs_benchmark, adjusted_tuprs_benchmark, xu100_benchmark = _run_period(
            period_frame,
            active_config,
            active_logger,
        )
        period_rows.append({"period": period.name, **metrics})
        comparison_rows.append(_comparison_row(period.name, "baseline_v1", metrics))
        comparison_rows.append(
            _comparison_row(
                period.name,
                tuprs_benchmark.name,
                benchmark_metrics(tuprs_benchmark, active_config),
            )
        )
        comparison_rows.append(
            _comparison_row(
                period.name,
                adjusted_tuprs_benchmark.name,
                benchmark_metrics(adjusted_tuprs_benchmark, active_config),
            )
        )
        comparison_rows.append(
            _comparison_row(
                period.name,
                xu100_benchmark.name,
                benchmark_metrics(xu100_benchmark, active_config),
            )
        )

    metrics_table = pd.DataFrame([{"period": "all", "strategy": "baseline_v1", **full_metrics}])
    period_table = pd.DataFrame(period_rows)
    benchmark_table = pd.DataFrame(comparison_rows)
    drawdown_table = calculate_drawdowns(full_result.daily_equity)
    data_quality = audit_corporate_actions(features)
    data_quality["raw_strategy_prices_consistent"] = raw_strategy_prices_are_consistent(
        features.loc[~features["is_indicator_warmup"].astype(bool)]
    )

    artifacts = BacktestArtifacts(
        trades=output_dir / "baseline_v1_trades.csv",
        daily_equity=output_dir / "baseline_v1_daily_equity.csv",
        metrics=output_dir / "baseline_v1_metrics.csv",
        period_metrics=output_dir / "baseline_v1_period_metrics.csv",
        benchmark=output_dir / "baseline_v1_benchmark.csv",
        drawdowns=output_dir / "baseline_v1_drawdowns.csv",
        data_quality=output_dir / "baseline_v1_data_quality.csv",
        markdown_report=report_dir / "baseline_v1_report.md",
        equity_curve=report_dir / "baseline_v1_equity_curve.png",
        drawdown_chart=report_dir / "baseline_v1_drawdown.png",
        annual_returns=report_dir / "baseline_v1_annual_returns.png",
        trade_returns=report_dir / "baseline_v1_trade_returns.png",
        strategy_vs_benchmark=report_dir / "baseline_v1_strategy_vs_benchmark.png",
    )

    full_result.trades.to_csv(artifacts.trades, index=False)
    full_result.daily_equity.to_csv(artifacts.daily_equity, index=False)
    metrics_table.to_csv(artifacts.metrics, index=False)
    period_table.to_csv(artifacts.period_metrics, index=False)
    benchmark_table.to_csv(artifacts.benchmark, index=False)
    drawdown_table.to_csv(artifacts.drawdowns, index=False)
    data_quality.to_csv(artifacts.data_quality, index=False)

    _write_markdown_report(
        artifacts.markdown_report,
        full_metrics,
        period_table,
        benchmark_table,
        data_quality,
    )
    _plot_equity_curve(full_result.daily_equity, artifacts.equity_curve)
    _plot_drawdown(drawdown_table, artifacts.drawdown_chart)
    _plot_annual_returns(full_result.daily_equity, artifacts.annual_returns)
    _plot_trade_returns(full_result.trades, artifacts.trade_returns)
    _plot_strategy_vs_benchmarks(
        full_result.daily_equity,
        full_tuprs,
        full_adjusted_tuprs,
        full_xu100,
        artifacts.strategy_vs_benchmark,
    )

    active_logger.info(
        "Backtest completed with %s trades and %.2f%% total return",
        full_metrics["total_trades"],
        float(full_metrics["total_return"]) * 100,
    )
    return BacktestPipelineResult(artifacts, full_metrics, benchmark_table)


def _format_percent(value: object) -> str:
    return f"{float(value) * 100:.2f}%"


def _write_markdown_report(
    path: Path,
    metrics: dict[str, object],
    period_table: pd.DataFrame,
    benchmark_table: pd.DataFrame,
    data_quality: pd.DataFrame,
) -> None:
    comparison = benchmark_table.loc[
        benchmark_table["period"].eq("all"),
        ["benchmark", "ending_capital", "total_return", "maximum_drawdown"],
    ]
    lines = [
        "# Baseline V1 Backtest Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Methodology",
        "",
        "Signals are calculated from daily closes and entries execute at the next eligible open. "
        "Close-based exits also execute at the following trading day's open. "
        "The engine is long-only, unleveraged, uses whole shares, and carries one position at a time. "
        "Stops use only levels known before the intraday low is evaluated.",
        "",
        "## Full Period",
        "",
        "Metric | Value",
        "---|---:",
        f"Trades | {metrics['total_trades']}",
        f"Ending capital | {float(metrics['ending_capital']):,.2f} TL",
        f"Total return | {_format_percent(metrics['total_return'])}",
        f"CAGR | {_format_percent(metrics['cagr'])}",
        f"Maximum drawdown | {_format_percent(metrics['maximum_drawdown'])}",
        f"Sharpe ratio | {float(metrics['sharpe_ratio']):.3f}",
        f"Win rate | {_format_percent(metrics['win_rate'])}",
        "",
        "## Period Results",
        "",
        "Period | Trades | Total Return | Max Drawdown | Ending Capital",
        "---|---:|---:|---:|---:",
    ]
    for row in period_table.itertuples(index=False):
        lines.append(
            f"{row.period} | {row.total_trades} | {_format_percent(row.total_return)} | "
            f"{_format_percent(row.maximum_drawdown)} | {row.ending_capital:,.2f} TL"
        )
    lines.extend(
        [
            "",
            "## Full-Period Benchmarks",
            "",
            "Series | Ending Capital | Total Return | Max Drawdown",
            "---|---:|---:|---:",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"{row.benchmark} | {row.ending_capital:,.2f} TL | "
            f"{_format_percent(row.total_return)} | {_format_percent(row.maximum_drawdown)}"
        )
    lines.extend(
        [
            "",
            "## Bias Controls",
            "",
            "- Indicator warm-up rows are excluded.",
            "- Entry signals execute on the following trading day's open.",
            "- Close-based exit signals execute on the following trading day's open.",
            "- External market values are inherited from the causal feature pipeline only.",
            "- No parameter optimization or machine learning is used.",
            "- The fixed TUPRS symbol avoids index-constituent selection during this baseline stage.",
            "",
            "## Corporate Actions",
            "",
            f"- Corporate-action candidates: {int(data_quality['corporate_action_candidate'].sum())}.",
            f"- Raw OHLC consistency check: {bool(data_quality['raw_strategy_prices_consistent'].all())}.",
            "- Strategy execution remains on raw OHLC; adjusted OHLC is not used for signals or fills.",
            "- `tuprs_raw_buy_hold` uses tradable raw prices and execution costs.",
            "- `tuprs_adjusted_total_return` uses Adjusted Close as a return index, so Yahoo's dividend and split adjustments are reflected; it does not apply explicit transaction costs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_equity_curve(equity: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(equity["date"], equity["total_equity"], color="#176B87", linewidth=1.5)
    axis.set(title="Baseline V1 Equity Curve", xlabel="Date", ylabel="Equity (TL)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_drawdown(drawdowns: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.fill_between(drawdowns["date"], drawdowns["drawdown"] * 100, 0, color="#C0392B", alpha=0.8)
    axis.set(title="Baseline V1 Drawdown", xlabel="Date", ylabel="Drawdown (%)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_annual_returns(equity: pd.DataFrame, path: Path) -> None:
    annual = (
        equity.assign(year=pd.to_datetime(equity["date"]).dt.year)
        .groupby("year")["daily_return"]
        .apply(lambda returns: (1 + returns).prod() - 1)
    )
    colors = ["#2E8B57" if value >= 0 else "#C0392B" for value in annual]
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(annual.index.astype(str), annual.values * 100, color=colors)
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set(title="Baseline V1 Annual Returns", xlabel="Year", ylabel="Return (%)")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_trade_returns(trades: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    if trades.empty:
        axis.text(0.5, 0.5, "No completed trades", ha="center", va="center")
    else:
        axis.hist(trades["return_pct"] * 100, bins=20, color="#176B87", edgecolor="white")
    axis.set(title="Baseline V1 Trade Returns", xlabel="Trade Return (%)", ylabel="Frequency")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_strategy_vs_benchmarks(
    strategy: pd.DataFrame,
    tuprs: BenchmarkResult,
    adjusted_tuprs: BenchmarkResult,
    xu100: BenchmarkResult,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.plot(strategy["date"], strategy["cumulative_return"] * 100, label="Baseline V1")
    axis.plot(
        tuprs.daily_equity["date"],
        tuprs.daily_equity["cumulative_return"] * 100,
        label="TUPRS Buy & Hold",
    )
    axis.plot(
        adjusted_tuprs.daily_equity["date"],
        adjusted_tuprs.daily_equity["cumulative_return"] * 100,
        label="TUPRS Adjusted Total Return",
    )
    axis.plot(
        xu100.daily_equity["date"],
        xu100.daily_equity["cumulative_return"] * 100,
        label="XU100",
    )
    axis.set(title="Strategy vs Benchmarks", xlabel="Date", ylabel="Cumulative Return (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
