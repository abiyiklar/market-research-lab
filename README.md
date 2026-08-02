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
`data/quality/` | generated per-symbol and panel data-quality reports, ignored by git
`data/backtest/` | generated backtest tables, ignored by git
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

## Multi-Stock BIST Panel

Collect the enabled `initial_bist_panel` universe with one command:

```powershell
python -m bist_research.collector --universe initial_bist_panel
```

The initial panel contains `TUPRS.IS`, `THYAO.IS`, `ASELS.IS`, `FROTO.IS`, `EREGL.IS`, `SISE.IS`, `AKBNK.IS`, `ISCTR.IS`, `KCHOL.IS`, `BIMAS.IS`, `TCELL.IS`, and `ENKAI.IS`. `XU100.IS` is the common market benchmark. The configured sector benchmarks are `XUSIN.IS`, `XUHIZ.IS`, `XBANK.IS`, `XUMAL.IS`, and `XUTEK.IS`.

Run a single-symbol smoke collection:

```powershell
python -m bist_research.collector --symbols ASELS.IS
```

Raw downloads are immutable timestamped snapshots under `data/raw/equities/<symbol>/`. Incremental runs start after the latest processed date where practical, merge new and prior observations, and remove duplicate dates. Canonical cleaned files are written to `data/processed/equities/<symbol>/prices.*`. Benchmark data uses the same layout under `data/raw/benchmarks/` and `data/processed/benchmarks/`.

Collection preserves raw and adjusted OHLC data, volume, dividends, stock splits, repaired-row flags, source symbols, and collection timestamps. A failed symbol is isolated and marked excluded. If a preferred sector benchmark cannot be downloaded, the assignment explicitly records the fallback to `XU100.IS`; the substitution is never silent.

Quality outputs include:

- `data/quality/<symbol>_collection_quality.csv`
- `data/quality/benchmark_assignments.csv`
- `data/quality/multi_stock_collection_summary.csv`

Build generic features for the panel or one symbol:

```powershell
python -m bist_research.features --universe initial_bist_panel
python -m bist_research.features --symbols ASELS.IS
```

Each symbol is written independently to `data/features/equities/<symbol>/features.csv` and `features.parquet`. The panel summary is `data/quality/multi_stock_feature_summary.csv`.

The feature calendar contains only the stock's valid tradable sessions. Stock OHLC and volume are never filled. Market and sector index values may be forward-filled from past observations only. Indicators use causal total-return-continuous signal OHLC, while `execution_open`, `execution_high`, `execution_low`, and `execution_close` preserve raw prices.

Sprint 1 provides data collection, quality reporting, and deterministic feature generation only. It does not perform portfolio optimization, machine learning, strategy parameter search, backtesting, or a research run.

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

Baseline V1 keeps its parameters fixed: the XU100 daily-return floor is `-3%`, the entry RSI range is `50-72`, the minimum volume ratio is `1.10`, the initial ATR stop is `2.5x`, the trailing ATR stop is `3.0x`, and the maximum holding period is 60 trading days. Close-based exits are evaluated at the close; intraday stops use levels known before that day's low is tested.

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

The command writes trade, daily-equity, metric, period, benchmark, and drawdown tables to `data/backtest/`. The Markdown report and equity, drawdown, annual-return, trade-return, and benchmark charts are written to `reports/`.

## Tests

```powershell
python -m pytest
```
