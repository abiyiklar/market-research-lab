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

Existing raw files are preserved unless `--overwrite-raw` is supplied explicitly. Collect the action-enriched TUPRS history and its audit in a separate derived directory with:

```powershell
python -m bist_research.collector --actions-only --symbols TUPRS.IS
```

This writes raw OHLC, Adjusted Close, dividends, stock splits, and the yfinance repaired-data indicator under `data/processed/corporate_actions/` without replacing `data/raw/`.

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

The pipeline uses only tradable TUPRS sessions: positive volume, finite positive OHLC, and internally valid high/low bounds. It left joins the other markets and only forward-fills external values from prior observations. TUPRS prices are never forward-filled. TUPRS returns and indicators use a causal total-return-continuous signal OHLC built from raw prices and the dividend reported on that date; raw OHLC remains unchanged. The first 200 tradable TUPRS sessions are marked with `is_indicator_warmup`.

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

The engine calculates signals from total-return-continuous daily signal closes and executes entries at the next eligible raw tradable open. Pending entries and exits carry across non-tradable rows; holding days, stops, and period liquidation use the same tradable-session calendar. It uses whole shares without leverage and applies configurable commission and slippage costs.

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

The strategy uses `tuprs_signal_open/high/low/close` for indicators and signals while retaining raw TUPRS OHLC for fills. Reported cash dividends are credited once while long, net of the configurable `--dividend-withholding-rate` (default `0`); raw-basis stop and trailing references are reduced by the gross dividend before ex-date stop comparisons. Yahoo TUPRS raw OHLC is audited around reported splits before any quantity treatment, and split-adjusted prices are not adjusted a second time. The benchmark table reports raw buy-and-hold, explicit-dividend total return, and Yahoo Adjusted Close total return separately.

## Run Controlled Strategy Research

Run the deterministic research engine with its default cap of 150 experiments:

```powershell
python -m bist_research.research.cli
```

The engine evaluates `baseline_v1_fixed`, `trend_following`, `breakout`, `pullback_in_uptrend`, and `relative_strength` families. For each rolling 4-year train, 1-year validation, and 1-year OOS window, train quality gates are applied first and surviving candidates are ranked only on validation. At most one candidate per family (up to five total) is frozen and run exactly once on that window's OOS period. Only these selected OOS runs feed the final leaderboard.

Bounded example:

```powershell
python -m bist_research.research.cli `
  --max-experiments 30 `
  --random-seed 20240801 `
  --strategy trend_following breakout `
  --output-dir data/research `
  --report-dir reports
```

The best 10 selected-OOS candidates receive doubled-cost, parameter perturbation, start-offset, best-trade-removal, delayed-entry, and deterministic signal-skip stress tests. Best-trade removal disables the original signal date and reruns the full engine. Ablation is diagnostic and runs only on train/validation data; it is not used to retune OOS candidates.

Decision fields are reported separately as `hard_gate_pass`, `stress_stability_pass`, `statistically_robust`, `economically_competitive`, and `deployment_ready`. Historical candidates remain `deployment_ready=False` until forward paper trading is completed. A statistically stable candidate with insufficient benchmark-relative economic value remains a preliminary research candidate.

Each research run replaces the current derived CSV outputs so legacy non-nested OOS rows cannot remain mixed with valid evidence. Reports include the leaderboard, top-candidate summary, robustness heatmap, walk-forward returns, family comparison, and parameter-stability chart. No machine learning or open-ended "search until good" loop is used.

## Tests

```powershell
python -m pytest -q
python -m pip check
```
