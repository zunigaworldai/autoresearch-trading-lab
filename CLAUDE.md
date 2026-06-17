# CLAUDE.md — Trading Station AI / Options Desk

## Project scope

This is the **Trading Station AI / Options Desk** research lab. The goal is disciplined, systematic discovery and validation of short-to-medium-term trading strategies for U.S. equities and ETFs using free OHLCV data.

## Current phase

**Free OHLCV research lab.** No live trading code exists. No futures. No crypto order book data. No paid data dependencies. All research uses existing Alpaca OHLCV data only. Future strategies will use additional providers and Python.

## Instruments

AAPL, MSFT, NVDA, QQQ, SPY.

## Official timeframes

| Timeframe | Role |
|-----------|------|
| 1m | Execution/slippage proxy |
| 5m | Entry |
| 15m | Main intraday setup |
| 30m / 1h | Structure/context |
| 1D | Regime/context |
| 2m / 3m | Robustness checks only |

## Validation requirements

Every strategy candidate must pass ALL of the following before promotion:

- Train / validation / out-of-sample splits
- Walk-forward analysis
- Transaction cost stress: x1, x2, x3
- Monte Carlo simulation
- Yearly consistency (no negative full years)
- Regime stability
- Single-trade dependency check

**Never optimize parameters only to improve results.** If a parameter change does not have a structural rationale, discard it.

## What must never be committed

- `.env`, API keys, broker credentials, OAuth tokens
- Raw market data (`data/raw/`)
- Processed data (`data/processed/`, `data/processed_standard/`)
- Logs (`logs/`)
- Large outputs, parquet, CSV, DB, SQLite, HDF5 datasets
- Any file containing secrets or credentials

## Workflow discipline

- One experiment per branch, one hypothesis per commit.
- Always report files changed and validation commands run.
- Do not modify live execution, broker, or webhook code without human approval.
- Do not add dependencies without approval.
- Do not promote strategies from short samples or single-metric optimization.
- Keep failed experiments visible and documented.

## Key files

- `AGENTS.md` — agent roles and onboarding
- `docs/DATA_POLICY.md` — data handling rules
- `docs/OPTIONS_DESK_CONTEXT.md` — trading desk context
- `docs/TRADING_RESEARCH_PROTOCOL.md` — research process and gate criteria
- `.github/copilot-instructions.md` — agent behavior guidelines
- `config/instruments.yaml` — instrument configuration
- `config/trading_runtime.yaml` — runtime settings
