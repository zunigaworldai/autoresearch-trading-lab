# AUTORESEARCH PROGRAM — Trading Lab

## Mission

This repo uses Karpathy AutoResearch as the research-loop model, adapted to systematic trading.

The agent must:

1. Read research outputs.
2. Audit what failed.
3. Propose one small experiment.
4. Modify only research/strategy code in an experiment branch.
5. Run backtests and diagnostics.
6. Keep only changes that improve robust performance.
7. Reject changes that only improve short samples.
8. Never touch live-money execution without explicit human approval.

## Production gates

No strategy can move to paper/live unless it passes:

- No negative years in 5Y validation.
- Strong monthly consistency.
- Weekly losses small and recoverable.
- Positive expectancy.
- Profit Factor > 1.
- Controlled max drawdown.
- Survives cost_2x and cost_3x stress.
- Walk-forward validation.
- No single-trade or single-year dependency.

## Current state

The 5-month discovery sample found promising NVDA/QQQ candidates.

The 5Y validation rejected the current strategy family for production.

Main rescue area:

- NVDA 5m
- A_only / A_C
- 09:30–10:30
- break_even_1r
- Needs regime filters and exit redesign.

## Allowed next experiments

- Regime filters: gap, volatility, trend, ATR, SMA20/SMA50, day-of-week.
- Indicator filters: RSI, MACD, Bollinger Bands, moving averages.
- Exit redesign: ATR trailing, partial profits, improved break-even, time stop.
- Monthly/weekly consistency analyzer.
- New strategy families if rescue fails.

## Forbidden actions

- Do not connect real money.
- Do not modify broker/live executor automatically.
- Do not commit raw market data.
- Do not promote a strategy from a short sample.
- Do not hide failed experiments.

## Research loop

Run:

```bash
bash scripts/research/run_autoresearch_audit.sh
