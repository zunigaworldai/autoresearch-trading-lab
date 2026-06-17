# Options Desk Context

This file captures the trading desk environment and the decision-making context for the research team.

## Trading desk mission

- Discover and refine short-to-medium-term systematic strategies for U.S. equity and ETF instruments.
- Emphasize robust, repeatable performance through consistent risk management and stress testing.
- Focus on patterns in opening-range behavior, VWAP, liquidity sweeps, mean reversion, continuation, and regime filters.

## In-scope instruments

Current research is primarily centered on:

- `AAPL`
- `MSFT`
- `NVDA`
- `QQQ`
- `SPY`

These symbols appear in the repository data and most research reports.

## Timeframes and signals

The current strategy universe includes intraday timeframes such as:

- 2-minute
- 3-minute
- 5-minute

Common research signals and structural elements:

- Opening range breakouts and failed breakouts
- VWAP reclaim and mean reversion patterns
- Gap continuation and gap fade behavior
- Liquidity sweep and reversal setups
- Trend vs. mean reversion regime filters
- Volatility and ATR-based timing

## Risk and execution context

The desk operates with conservative risk controls:

- Include transaction costs, slippage, and spread in every simulation.
- Test under stressed costs: `cost_2x` and `cost_3x`.
- Avoid leverage or execution assumptions that cannot be replicated in backtesting.
- Use conservative exits: fixed stops, break-even rules, partial profits, and time stops.

## Performance context

The desk evaluates candidates by more than just returns:

- Monthly consistency and stability
- Positive expectancy
- Controlled maximum drawdown
- No negative full years in validation periods
- Avoid overdependence on single years, single trades, or narrow market conditions
- Preference for stable performance across regimes

## Decision-making guidelines

- Prefer strategies that survive robust validation over those that optimize a single metric.
- Seek simplicity and clarity in entry/exit logic.
- Treat each change as an experiment: one hypothesis, one modification, one evaluation.
- Capture failures openly and use them to refine future hypotheses.

## Desk boundaries

- This repo is for research and strategy development, not live trading deployment.
- Do not make production or broker-level changes without explicit human approval.
- Do not expose raw data or sensitive execution details outside the approved research archive.
