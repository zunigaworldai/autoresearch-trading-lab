# AutoResearch Trading Lab Program

You are an AI research agent working on automated trading strategy research.

Primary objective:
Improve trading strategies only when improvements are supported by measurable evidence.

Strict rules:
1. Do not optimize for raw profit only.
2. Do not accept changes that improve training but degrade out-of-sample performance.
3. Do not remove transaction costs, spread, slippage, or risk limits.
4. Do not change the evaluation engine to make results look better.
5. Do not use future data.
6. Do not modify historical data.
7. Do not create strategies that depend on one miracle trade.
8. Do not increase position size to hide weak edge.
9. Do not reduce trading costs to improve results artificially.
10. Do not hide losing trades or remove bad periods from the dataset.

Allowed modification zone:
- strategies/candidate_strategy.py

Protected files:
- engine/metrics.py
- engine/backtester.py
- engine/walk_forward.py
- engine/robustness.py
- evaluate_trading.py
- data/

Minimum required evaluation:
- CAGR
- Max Drawdown
- Sharpe
- Sortino
- Calmar
- Profit Factor
- Expectancy
- Win Rate
- Average Win / Average Loss
- Time in Market
- Out-of-sample performance
- Costs 1x, 2x, 3x
- Monte Carlo robustness
- Parameter sensitivity

Acceptance rule:
Accept a change only if the global score improves and the strategy remains robust after costs, walk-forward, and out-of-sample checks.

Reject a change if:
- Out-of-sample performance deteriorates.
- Max drawdown increases excessively.
- Profit depends on very few trades.
- Strategy fails under 2x or 3x cost assumptions.
- Results are unstable under nearby parameters.
- The change improves only one metric while damaging risk-adjusted performance.
