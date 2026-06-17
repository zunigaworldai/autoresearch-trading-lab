# Data Policy

This repository uses market data to support trading strategy research. This policy defines how data may be used, stored, and shared.

## Data categories

- `data/raw/` — raw market data. This is source data that should remain local and should not be committed to git.
- `data/processed/` — preprocessed data derived from raw sources. It may be used by research scripts, but should still be treated as local research assets.
- `outputs/` — experiment results, reports, and diagnostics. These files can be shared in summaries but should not contain raw or proprietary data.
- `logs/` — runtime logs and alerts. Logs may include details of research runs, but should not leak sensitive execution or data source credentials.

## Data usage rules

- Do not commit raw market data to the repository.
- Do not commit any proprietary or licensed data without explicit permission.
- Use only approved data sources and document the source, licensing, and refresh process.
- Keep data file paths and naming conventions consistent with `config/instruments.yaml` and `config/trading_runtime.yaml`.

## Data governance

- Verify that any external data sources are permitted for research use.
- If new data is added, document it clearly in the repository and confirm licensing.
- Avoid any PII or sensitive trader information in logs, reports, or results.
- For public sharing, aggregate or summarize results instead of exposing raw datasets.

## Data integrity

- Validate data quality before using it in experiments.
- Check for missing values, timestamp gaps, and inconsistent instrument symbols.
- Use the repository’s existing preprocessing scripts where available.
- Maintain a clear separation between raw ingestion and processed research data.

## Data sharing and publication

- Share strategy results, metrics, and research summaries only.
- Do not share raw tick-level or minute-level data publicly unless the license allows it.
- Do not export or publish data that is not authorized by the source or the human owner.

## Data handling for experiments

- Use local copies of data for backtests and diagnostics.
- Do not rely on hidden or undocumented external data sources.
- If an experiment requires new data, get explicit approval and document the source.

## Summary

This project’s data policy is simple: use market data for research only, preserve privacy and licensing, avoid committing raw data, and document all sources clearly. When in doubt, ask the human owner.
