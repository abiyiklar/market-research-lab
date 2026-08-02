# Market Research Lab

Python starter platform for BIST and financial market research data collection.

## Project Layout

File or Folder | Purpose
---------|----------
`src/bist_research/` | Python BIST research collector package
`tests/` | pytest coverage for the collector
`data/raw/` | yfinance raw downloads, ignored by git
`data/processed/` | cleaned datasets and reports, ignored by git
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

## Tests

```powershell
python -m pytest
```
