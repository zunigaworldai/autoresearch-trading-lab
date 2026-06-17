# Agent Roles and Onboarding

This repository is organized around autonomous research agents operating within a disciplined trading research workflow.

## Overview

The project adapts Karpathy AutoResearch for trading strategy discovery. Agents collaborate to generate hypotheses, modify research code, run backtests, analyze performance, and preserve only robust improvements.

## Primary agent roles

### Research Agent

- Reads the repository context and current strategy status.
- Proposes and implements one focused strategy change at a time.
- Works on `strategies/` and evaluation scripts in `scripts/backtest/`.
- Commits the experimental change, runs tests/backtests, and evaluates performance.
- Discards changes that do not improve robust metrics.

### Audit Agent

- Reviews experiment outcomes and validation results.
- Checks against production gate criteria from `AUTORESEARCH_PROGRAM_TRADING.md`.
- Confirms whether an experiment should be kept, discarded, or documented as a failure.
- Ensures no strategy advance happens from a short sample alone.

### Risk Agent

- Enforces risk controls and stress-testing policies.
- Validates cost stress (`cost_2x`, `cost_3x`) results.
- Verifies drawdown, monthly consistency, and single-year dependency checks.
- Flags any candidate that fails the established gate rules.

### Data Agent

- Verifies the data environment and file organization.
- Ensures raw market data is not committed and is used through approved workflows.
- Keeps `data/raw/` and `data/processed/` separate from source control.
- Documents data sources and licensing in `docs/DATA_POLICY.md`.

### Human Gatekeeper

- Provides explicit approval for production/paper deployment.
- Reviews high-level strategy decisions and live money execution changes.
- Resolves uncertain cases and approves new research directions.

## Onboarding checklist

1. Read the repo summary in `README.md`.
2. Read the trading research rules in `program_trading.md` and `AUTORESEARCH_PROGRAM_TRADING.md`.
3. Read relevant strategy files under `strategies/` and research reports in `docs/research/`.
4. Confirm the current experiment branch naming rules and branch state.
5. Verify local data availability and that raw data is not committed.
6. Confirm the current set of production gate rules before any experiment is moved forward.

## Branch and experiment workflow

- Always work in a fresh branch named `autoresearch/<tag>`.
- Keep experiments small and incremental.
- Run a single experiment per commit.
- Collect results in logs or audit files, but do not commit raw results data unless required by the project.
- If an experiment fails or regresses, revert the branch to the last validated commit.

## Safety constraints

- Do not modify live execution, broker integration, or machine interface code without explicit human approval.
- Do not install new packages or alter dependency configuration without approval.
- Do not promote strategies from a short sample; require robust validation.
- Do not hide failed experiments.

## Summary

Agents should collaborate within the research lab, keep changes focused, and preserve the integrity of the empirical workflow. The human gate remains the final authority for production decisions.
