# Dalio Five Forces Tracker

A clean, extensible daily tracking system for Ray Dalio's five forces:

1. Debt / Money / Economic Cycle
2. Internal Order / Disorder
3. Great Power Conflict
4. Acts of Nature
5. Technology

The goal of v0.1 is not to predict macro markets perfectly. The goal is to convert heterogeneous public data into normalized, auditable force scores, track changes over time, and generate a daily market-relevant report with clear data-quality notes.

## Core Design

The project does not collapse everything into one black-box macro score. It builds separate latent-force style indices:

- `debt_money_score`
- `internal_disorder_score`
- `geopolitical_conflict_score`
- `nature_shock_score`
- `tech_productivity_score`
- `tech_fragility_score`

It then computes:

```python
macro_fragility_score = weighted_average([
    debt_money_score,
    internal_disorder_score,
    geopolitical_conflict_score,
    nature_shock_score,
])

productivity_upside_score = weighted_average([
    tech_productivity_score,
])

market_setup_score = productivity_upside_score - macro_fragility_score
net_tech_setup = tech_productivity_score - tech_fragility_score
```

Technology is split because the productivity impulse and the market-fragility impulse can both be true at the same time.

## Indicator Mapping

Indicator definitions live in `config/indicators.yml`. Each indicator includes:

- `force`
- `component`
- `source`
- `frequency`
- `ticker_or_series_id`
- `transform`
- `direction`
- `weight`
- `market_pricing_flag`
- `required`
- `notes`

The first milestone includes:

- FRED macro and policy-risk series, including rates, inflation, unemployment, credit spreads, dollar pressure, M2, and the US daily EPU index.
- Market proxies through `yfinance` with Stooq fallback, including VIX, SPY, QQQ, SMH, GLD, UUP, ITA, DBA, KIE, and XLU.
- GPR connector for the Caldara-Iacoviello geopolitical risk spreadsheet when the endpoint and format are available.
- GDELT connector for news-intensity proxies.
- Nature placeholder connector with explicit stale-data handling.
- Technology market proxies for productivity and fragility.

## Scoring

For each raw indicator, the system:

1. Aligns observations to a daily calendar.
2. Forward-fills only within a configurable limit by frequency.
3. Stores `last_updated_date`.
4. Stores `data_staleness_days`.
5. Applies the configured transform and direction so higher always means more of that force.
6. Computes latest level, 1m change, 3m change, rolling percentiles, z-score, robust z-score, and a 0-100 score.

The preferred score is based on robust z-score:

```python
robust_z = (x - rolling_median) / rolling_MAD
```

The robust z-score is converted to a 0-100 normal-CDF score. If robust history is not yet available, rolling percentiles are used as fallbacks.

## Confidence

Force confidence is based on:

- number of available indicators
- missing required or optional inputs
- data freshness
- source quality
- whether a force is driven mostly by slower real-world data or market/news proxies

If a source fails, the daily run continues with lower confidence and an explicit warning.

## Daily Run

Install dependencies with Python 3.11+:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment template if you want local overrides:

```bash
copy .env.example .env
```

Run the daily tracker:

```bash
python run_daily.py
```

Outputs are written to:

- `data/reports/YYYY-MM-DD_dalio_five_forces_report.md`
- `data/processed/latest_scores.csv`
- `data/processed/latest_scores.json`
- `data/processed/indicator_panel.parquet`
- `data/processed/force_scores.parquet`

If Parquet dependencies are missing, the runner logs a warning and writes CSV fallbacks for the affected files.

## Adding an Indicator

Add a new entry to `config/indicators.yml`:

```yaml
new_indicator_id:
  name: Human-readable name
  force: geopolitical_conflict
  component: real_world
  source: fred
  frequency: daily
  ticker_or_series_id: SOME_SERIES_ID
  transform: level
  direction: positive
  weight: 1.0
  market_pricing_flag: false
  required: false
  notes: Why this maps to the force.
```

If the source is new, add a connector under `src/data_sources/` with:

```python
fetch()
clean()
save_raw()
save_processed()
```

## Interpreting the Report

The report separates:

- real-world force level
- 1m / 3m momentum
- market pricing
- underpriced or overpriced signal
- confidence

High force scores do not automatically mean "sell risk." For example, high tech productivity and high macro fragility can coexist. The report uses cautious wording such as "may suggest" and "risk management implication."

## Backtest Skeleton

The `src/backtest/event_study.py` module defines event dates from force scores and calculates 1m, 3m, and 6m forward outcomes for assets such as SPY, QQQ, TLT, GLD, XLE, ITA/XAR, HYG, UUP, and VIX.

The event study uses only data available at or before the event date to prevent lookahead bias in event detection.

## Current Limitations

- Some public sources change file formats or rate-limit requests.
- GDELT news intensity is a proxy for attention, not an official incident count.
- Nature data is intentionally underbuilt in v0.1. Official disaster and climate datasets should be added before using the nature force as more than a watchlist.
- Many technology fundamentals are market proxies in v0.1. Future versions should add capex, productivity, semiconductor sales, AI-index, patent, and startup-investment data.
- FRED and market data revisions are not versioned in this MVP.

## Data-Source Caveats

- FRED graph CSV downloads can work without an API key, but some use cases may benefit from a FRED API key.
- Market data uses `yfinance` first and Stooq fallback where possible. Free market data can contain adjustments, ticker changes, or delayed corrections.
- GPR data is fetched from an external spreadsheet endpoint and may fail if its shape changes.
- GDELT DOC API queries are phrase-based and should be validated before production use.

## Disclaimer

This project is for research, monitoring, and education. It is not personalized financial advice, investment advice, or a recommendation to buy or sell any security.
