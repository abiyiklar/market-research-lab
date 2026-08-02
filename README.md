# Market Research Lab

Python starter platform for BIST and financial market research data collection.

## Project Layout

File or Folder | Purpose
---------|----------
`src/bist_research/` | Python BIST research collector package
`tests/` | pytest coverage for the collector
`data/raw/` | yfinance raw downloads, ignored by git
`data/processed/` | cleaned datasets and reports, ignored by git
`data/features/` | generated analysis datasets and quality summaries, ignored by git
`data/backtest/` | generated backtest tables, ignored by git
`data/research/` | generated experiment, walk-forward, stress, and ablation tables, ignored by git
`reports/` | generated backtest reports and charts, ignored by git
`logs/` | collector logs, ignored by git

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Collect Data

Run the collector with the default starter universe and start date:

```powershell
python -m bist_research
```

Defaults:

- Start date: `2013-01-01`
- Symbols: `TUPRS.IS`, `XU100.IS`, `XU030.IS`, `XUSIN.IS`, `TRY=X`, `BZ=F`, `CL=F`
- Raw outputs: `data/raw/*.csv` and `data/raw/*.parquet`
- Processed outputs: `data/processed/*.csv` and `data/processed/*.parquet`
- Zero-volume equity report: `data/processed/zero_volume_equity_records.*`
- Collection summary: `data/processed/collection_summary.*`
- Logs: `logs/collector.log`

Optional example:

```powershell
python -m bist_research --start-date 2020-01-01 --symbols TUPRS.IS XU100.IS
```

The collector removes rows with missing `Close` values. Zero-volume records are reported separately only for equity symbols.

## Build TUPRS Features

Build the analysis dataset from the cleaned Parquet files in `data/processed`:

```powershell
python -m bist_research.features
```

The pipeline uses TUPRS trading dates as its calendar, left joins the other markets, and only forward-fills external values. All returns and indicators use current and historical observations only. The first 200 TUPRS rows are marked with `is_indicator_warmup`.

Outputs:

- `data/features/tuprs_features.csv`
- `data/features/tuprs_features.parquet`
- `data/features/tuprs_feature_summary.csv`

Relative momentum is calculated from the TUPRS/XU100 relative-strength ratio. Rolling volatility is annualized using 252 trading days.

## Run Baseline V1 Backtest

Run the long-only daily reference strategy from the generated TUPRS feature dataset:

```powershell
python -m bist_research.backtest.cli
```

The engine calculates signals from daily closes and executes entries at the next eligible open. It excludes indicator warm-up rows, blocks new entries on zero-volume days, uses whole shares without leverage, and applies configurable commission and slippage costs.

Baseline V1 keeps its parameters fixed: the XU100 daily-return floor is `-3%`, the entry RSI range is `50-72`, the minimum volume ratio is `1.10`, the initial ATR stop is `2.5x`, the trailing ATR stop is `3.0x`, and the maximum holding period is 60 trading days. Close-based exits are signaled at the close and filled at the next trading day's open. Intraday stops use levels known before that day's low is tested; gap stops fill at the open and normal stops fill at the stop level.

Results are reported independently for train (`2013-2019`), validation (`2020-2022`), and test (`2023-latest`) periods, plus the full history. No parameter optimization or machine learning is performed.

Optional example:

```powershell
python -m bist_research.backtest.cli `
  --input data/features/tuprs_features.parquet `
  --output-dir data/backtest `
  --initial-capital 100000 `
  --commission-rate 0.001 `
  --slippage-rate 0.0005
```

The command writes trade, daily-equity, metric, period, benchmark, drawdown, and corporate-action data-quality tables to `data/backtest/`. The Markdown report and equity, drawdown, annual-return, trade-return, and benchmark charts are written to `reports/`.

The strategy uses raw TUPRS OHLC for signals and execution. The benchmark table reports `tuprs_raw_buy_hold` separately from `tuprs_adjusted_total_return`. The adjusted benchmark uses Yahoo Adjusted Close as a return index, which reflects dividend and split adjustments; adjusted OHLC is not used to run the strategy.

## Run Controlled Strategy Research

Run the deterministic research engine with its default cap of 150 experiments:

```powershell
python -m bist_research.research.cli
```

The engine evaluates `baseline_v1_fixed`, `trend_following`, `breakout`, `pullback_in_uptrend`, and `relative_strength` families. It uses rolling 4-year train, 1-year validation, and 1-year out-of-sample windows. Candidate ranking combines OOS CAGR, profit factor, Sharpe ratio, drawdown, result dispersion, benchmark excess, and trade adequacy. OOS results are never used to generate or retune parameters.

Bounded example:

```powershell
python -m bist_research.research.cli `
  --max-experiments 30 `
  --random-seed 20240801 `
  --strategy trend_following breakout `
  --output-dir data/research `
  --report-dir reports
```

The best 10 experiments receive doubled-cost, parameter perturbation, start-offset, best-trade-removal, delayed-entry, and deterministic signal-skip stress tests. The same candidates receive one-filter-at-a-time ablation for market regime, volume, RSI, relative strength, trend, oil, and USDTRY filters.

Research CSV files are appended by deterministic experiment keys so prior experiment rows are retained. Reports include the leaderboard, top-candidate summary, robustness heatmap, walk-forward returns, family comparison, and parameter-stability chart. No machine learning or open-ended "search until good" loop is used.

## Tests

```powershell
python -m pytest
```
