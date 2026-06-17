# GitHub Copilot Instructions

This repository is an AI-driven trading research lab built on the Karpathy AutoResearch concept.

## Purpose

- Support systematic trading research in a disciplined, reproducible way.
- Keep the focus on strategy research and robust backtesting, not live execution.
- Preserve the existing experiment workflow, branch discipline, and data policy.

## Key rules for code and documentation

- Read the repository context before changing anything: `README.md`, `program_trading.md`, `AUTORESEARCH_PROGRAM_TRADING.md`, and the `docs/research/` dossier files.
- The main research scope is `strategies/`, `scripts/backtest/`, and `scripts/research/`.
- Do not modify `live_monitor/`, `webhook_receiver.py`, or any live broker/execution-related code unless explicitly instructed by the human.
- Do not add new dependencies or modify `pyproject.toml` without approval.
- Do not commit raw market data or private data sources. Follow the repo `docs/DATA_POLICY.md` guidelines.

## Recommended workflow

1. Confirm the user’s objective and scope.
2. Propose one small, focused experiment at a time.
3. Work on a dedicated experiment branch: `autoresearch/<tag>`.
4. Make one change, commit it, run the review/backtest flow, inspect results.
5. Keep only changes that improve robust performance.
6. Record outcomes in existing audit files or research reports.

## Behavior for autonomous agents

- Use the existing gate rules from `AUTORESEARCH_PROGRAM_TRADING.md`.
- Prefer safer, conservative research changes over risky or unvalidated ones.
- If a change degrades performance, discard it and reset to the previous validated state.
- If the repository contains a command or script for experiment execution, use it rather than inventing new tooling.

## What to check first

- `config/instruments.yaml` and `config/trading_runtime.yaml` for current market/instrument settings.
- `strategies/` for active strategy implementations.
- `engine/` and `scripts/backtest/` for evaluation and simulation logic.
- `docs/research/` for existing hypothesis and findings.

## When to ask the human

Ask the human if:

- the requested change affects production execution or live-money deployment.
- the change requires a new dependency or new data source.
- the experiment is beyond the current repo’s research scope.
- the data source or licensing status is unclear.

## Summary

This repo is about disciplined, agent-driven trading research. Keep changes small, transparent, and well-documented. Follow the gate rules, do not touch live execution, and respect data policy.
