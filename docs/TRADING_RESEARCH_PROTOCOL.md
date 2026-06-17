# Trading Research Protocol

This document defines the research process used by the trading lab. It is the standard operating procedure for new experiments, validation, and decision-making.

## Research objective

Discover and refine systematic trading strategies that deliver robust performance in backtests and stress tests while preserving risk control.

## Research process

1. Review the current state.
   - Read `README.md`, `program_trading.md`, and `AUTORESEARCH_PROGRAM_TRADING.md`.
   - Inspect the latest research reports under `docs/research/`.
   - Confirm the active problem statement and candidate instruments.

2. Create an experiment branch.
   - Use a dedicated branch: `autoresearch/<tag>`.
   - Ensure the branch is new and does not overwrite prior work.

3. Define a single experiment.
   - Propose one focused change or hypothesis.
   - Keep the modification localized to a single strategy or component.
   - Avoid broad rewrites or multiple unrelated changes in the same run.

4. Implement the experiment.
   - Modify `strategies/` or the relevant backtest script only.
   - Do not change live production or execution code without approval.
   - Do not introduce new dependencies without approval.

5. Run the evaluation.
   - Execute backtests and diagnostics using the repository scripts.
   - Include transaction costs, slippage, and runtime risk controls.
   - Perform stress tests under higher cost scenarios.

6. Assess the results.
   - Compare performance to the baseline using robust metrics.
   - Check for regressions in risk, drawdown, consistency, and stress resilience.
   - Document outcomes clearly.

7. Decide whether to keep or discard.
   - Keep: the change improves robust performance and preserves risk characteristics.
   - Discard: the change degrades or does not improve meaningful metrics.
   - Crash: if the experiment fails due to a bug or invalid assumption, document and revert.

## Gate criteria for production readiness

A candidate is only considered for paper/live after it passes all of these criteria:

- No negative years in the 5-year validation window.
- Strong monthly consistency across validation.
- Weekly losses are small and recoverable.
- Positive expectancy overall.
- Profit Factor > 1.
- Controlled and acceptable maximum drawdown.
- Survives cost stress tests: `cost_2x` and `cost_3x`.
- Walk-forward validation where applicable.
- No single-trade or single-year dependency.

## Documentation and transparency

- Record every experiment clearly.
- Keep failed experiments visible and explain why they failed.
- Do not hide or remove research failures.
- Use audit files and reports to summarize the hypothesis and outcome.

## Experiment discipline

- One experiment per branch, commit, or run.
- Do not mix hypothesis changes with housekeeping changes.
- Revert to the last clean state if you cannot recover a failed experiment.
- Keep the experiment loop moving, but do not bypass the human gate for production changes.

## Review and approval

- The final decision for production or paper deployment is made by the human owner.
- Research agents may recommend candidates, but they do not promote them automatically.
- Explicit human approval is required before any live-money execution or deployment change.
