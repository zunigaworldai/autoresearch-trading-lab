# ORB Trap V2 Promotion Dossier — NVDA 5m

## Current Status

**Status:** WATCH WALK-FORWARD CANDIDATE  
**Approval:** NOT approved for paper trading, live trading, or real money.

This candidate is the strongest strategy candidate found so far in the current AutoResearch Trading Lab process, but it remains blocked because the out-of-sample density is still too low.

---

## Strategy Identity

- **Family:** Opening Failed Breakout / ORB Trap Reversal
- **Instrument:** NVDA
- **Timeframe:** 5m RTH
- **Base variant:** `both_or15_fail_rr15_extreme`
- **Base window:** `afternoon_1300_1500`
- **Survivor variant:** `robust_entry_core`
- **Exit policy:** `partial_50_at_1r_be`
- **Cost assumption:** Survives `cost_3x`
- **Karpathy marker:** `opening_failed_breakout_reversal_v2_watch_walk_forward_candidate`
- **Blocking marker:** `opening_failed_breakout_reversal_v2_not_paper_live_low_oos_density`

---

## Exact Trading Logic

### Base Setup

1. Build a 15-minute opening range.
2. Detect a breakout above/below the opening range.
3. Wait for failed continuation and return back inside the range.
4. Enter reversal:
   - Failed upside breakout -> short.
   - Failed downside breakdown -> long.
5. Stop is based on the failed breakout extreme.
6. Take partial at 1R and move remaining position to breakeven.
7. Force end-of-day exit.

### V2 Entry Filter

Only allow trades in these entry windows:

```text
13:10-13:25
OR
14:20-14:40
```

This is the `robust_entry_core` filter.

---

## Backtest Evidence

### Full Sample, Cost 3x

```text
Trades:        19
Profit Factor: 2.9721
Expectancy:    0.007481
Max DD:        0.0240
Years Neg:     1
MC p05 return: 0.0312
MC p95 DD:     0.0556
```

### OOS 2024+

```text
OOS 2024 trades: 6
OOS 2024 PF:     1.5133
OOS 2024 Exp:    0.003135
```

### OOS 2025+

```text
OOS 2025 trades: 5
OOS 2025 PF:     2.3010
OOS 2025 Exp:    0.006271
```

---

## Walk-Forward Segments, Cost 3x

| Split | Train Trades | Train PF | Test Trades | Test PF | Test Expectancy | Decision |
|---|---:|---:|---:|---:|---:|---|
| train_2022_test_2023 | 8 | 4.6748 | 5 | 3.7806 | 0.004265 | pass |
| train_2022_2023_test_2024 | 13 | 4.4812 | 1 | 0.0000 | -0.012545 | test_tiny_sample |
| train_2022_2024_test_2025 | 14 | 3.3093 | 5 | 2.3010 | 0.006271 | pass |
| train_2022_2025_test_2026 | 19 | 2.9721 | 0 | 0.0000 | 0.000000 | test_tiny_sample |

---

## Trade Density

Confirmed trade count by year for `robust_entry_core cost_3x`:

```text
2021: 0 trades
2022: 8 trades
2023: 5 trades
2024: 1 trade
2025: 5 trades
2026: 0 trades
```

Dataset contains bars for 2021 and 2026, so the absence of trades in those years is due to no valid setup triggers, not missing data.

---

## Why It Is Not Approved Yet

The candidate is promising, but still blocked because:

1. Low total trade count: only 19 trades.
2. Low OOS density: 2024 has only 1 test trade and it was negative.
3. No 2026 trades: current-year validation is absent.
4. Single-symbol dependency: currently only validated on NVDA.
5. Potential time-window overfitting: the entry windows were derived after inspecting the survivor.
6. No forward/paper evidence: all evidence is historical.

---

## Current Decision

```text
NO paper
NO live
NO real money
```

Approved label:

```text
watch_walk_forward_candidate
```

Not approved label:

```text
not_paper_live_low_oos_density
```

---

## Required Next Tests

Before promotion to paper trading, this candidate must pass:

1. Forward paper simulation with identical execution rules.
2. Additional OOS period once enough 2026+ trades exist.
3. Sensitivity around entry windows:
   - 13:05-13:30
   - 13:10-13:25
   - 13:15-13:30
   - 14:15-14:45
   - 14:20-14:40
   - 14:25-14:40
4. Single-trade dependency audit.
5. Slippage stress beyond cost_3x.
6. Monte Carlo by trade sequencing and trade omission.
7. Cross-symbol falsification test.
8. Portfolio correlation check if combined with other candidates.

---

## Promotion Rules

### Can move to paper only if:

```text
- At least 30 total trades
- At least 10 OOS trades after the final design lock
- OOS PF > 1.2
- OOS expectancy positive
- Monte Carlo p05 return >= 0
- No hidden single-trade dependency
- Cost_3x remains positive
```

### Must remain blocked if:

```text
- 2026 remains zero-trade / tiny-sample
- OOS expectancy turns negative
- PF collapses under cost stress
- edge depends on one or two trades
- signal frequency is too low to validate
```

---

## Operational Note

This is the first candidate in the current research chain to survive:

```text
cost_3x
OOS 2023
OOS 2025
Monte Carlo p05 positive
```

But it is not enough for deployment. It should be treated as a promising lab candidate requiring forward evidence.
