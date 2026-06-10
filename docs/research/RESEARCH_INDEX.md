# AutoResearch Trading Lab — Research Index

## Project Status

This file is the master research index for the strategy lab.  
It tracks which strategy families are blocked, which are watchlist candidates, and which are eligible for further validation.

Current deployment status:

```text
NO paper trading
NO live trading
NO real money
```

Only one strategy currently qualifies as an advanced watchlist candidate:

```text
Opening Failed Breakout Reversal V2 / robust_entry_core / NVDA 5m
```

---

## Watchlist Candidates

### 1. Opening Failed Breakout Reversal V2 — robust_entry_core

- **Status:** watch walk-forward candidate
- **Instrument:** NVDA
- **Timeframe:** 5m RTH
- **Base family:** Opening Failed Breakout / ORB Trap Reversal
- **Base variant:** `both_or15_fail_rr15_extreme`
- **Base window:** `afternoon_1300_1500`
- **V2 filter:** `robust_entry_core`
- **Entry windows:**
  - `13:10-13:25`
  - `14:20-14:40`
- **Exit policy:** `partial_50_at_1r_be`
- **Cost stress:** survived `cost_3x`
- **Karpathy marker:** `opening_failed_breakout_reversal_v2_watch_walk_forward_candidate`
- **Blocking marker:** `opening_failed_breakout_reversal_v2_not_paper_live_low_oos_density`
- **Dossier:** `docs/research/ORB_TRAP_V2_PROMOTION_DOSSIER.md`

#### Key Metrics, Cost 3x

```text
Trades:        19
PF:            2.9721
Expectancy:    0.007481
Max DD:        0.0240
YearsNeg:      1
OOS 2024 PF:   1.5133
OOS 2025 PF:   2.3010
MC p05 return: 0.0312
MC p95 DD:     0.0556
```

#### Decision

```text
NO paper
NO live
NO real money
```

Reason:

```text
Promising edge, but low OOS density.
2024 only had 1 test trade and it was negative.
2026 had 0 trades.
```

Next required work:

```text
Forward observation / paper-simulation only after enough new signals exist.
Single-trade dependency audit.
Additional OOS density.
Slippage beyond cost_3x.
Cross-symbol falsification.
```

---

## Blocked Strategy Families

### Opening Range Breakout V1 / Fast

- **Status:** blocked
- **Reason:** no zero-negative-year candidate found.
- **Karpathy marker:** `opening_range_breakout_v1_no_zero_negative_year_candidate`

---

### VWAP Mean Reversion

- **Status:** blocked / diagnostic only
- **Reason:** did not produce a robust positive-edge candidate across required filters.
- **Deployment:** no paper / no live.

---

### RSI Bollinger Mean Reversion

- **Status:** blocked / diagnostic only
- **Reason:** no robust candidate survived the full selection process.
- **Deployment:** no paper / no live.

---

### Inverse Signal Audit

- **Status:** diagnostic only
- **Reason:** used for falsification/inversion checks, not as a deployable strategy.
- **Deployment:** no paper / no live.

---

### EMA Pullback Trend Scalping

- **Status:** blocked
- **Reason:** no zero-negative-year candidate; multi-symbol results rejected or low sample.
- **Karpathy marker:** `ema_pullback_trend_scalping_v1_no_zero_negative_year_candidate`
- **Deployment:** no paper / no live.

---

### Gap Continuation / Fade

- **Status:** blocked
- **Reason:** AAPL watch rows failed cost_2x/cost_3x stress.
- **Karpathy marker:** `gap_continuation_fade_v1`
- **Deployment:** no paper / no live.

---

### Liquidity Sweep Reversal V1

- **Status:** blocked
- **Reason:** AAPL/NVDA watch rows failed cost_2x/cost_3x stress.
- **Karpathy marker:** `liquidity_sweep_reversal_v1_failed_cost_stress`
- **Deployment:** no paper / no live.

---

### Opening Failed Breakout Reversal V1

- **Status:** blocked for paper/live
- **Reason:** one NVDA setup survived cost_3x, but remained unstable due to negative years and low frequency.
- **Karpathy markers:**
  - `opening_failed_breakout_reversal_v1`
  - `opening_failed_breakout_reversal_v1_no_zero_negative_year_candidate`
  - `opening_failed_breakout_reversal_v1_watch_cost_survivor`
- **Deployment:** no paper / no live.

---

## Current Research Decision Tree

```text
If strategy has no cost_3x edge:
    BLOCKED

If strategy has cost_3x edge but negative years:
    WATCHLIST ONLY

If strategy has cost_3x edge + positive OOS but low trade density:
    WATCH WALK-FORWARD CANDIDATE

If strategy has enough OOS trades + stable cost_3x + MC positive:
    candidate_review

If candidate_review passes forward observation:
    paper trading candidate

No strategy currently reaches paper trading candidate status.
```

---

## Next Research Options

### Option A — Continue validating ORB Trap V2

Recommended if the goal is to mature the best current candidate.

Next steps:

```text
1. Single-trade dependency audit.
2. Trade omission Monte Carlo.
3. Slippage stress beyond cost_3x.
4. Build forward-observation tracker.
5. Wait for new signals in 2026+.
```

### Option B — Start a new family

Recommended if the goal is to find another uncorrelated source of edge.

Potential next families:

```text
1. Relative Strength / Sector Momentum Intraday
2. Previous Day Trend Continuation with Volatility Filter
3. VWAP Trend Day Continuation
4. Range Compression -> Expansion
5. Multi-symbol Regime Filter Layer
```

### Option C — Build Portfolio Layer

Recommended after at least two independent watchlist candidates exist.

Current status:

```text
Not enough validated candidates for portfolio construction.
```

---

## Current Best Candidate

```text
Opening Failed Breakout Reversal V2
NVDA 5m
robust_entry_core
watch walk-forward candidate
```

Still blocked for:

```text
paper/live/real money
```
