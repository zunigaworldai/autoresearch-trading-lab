from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.trade_simulator import run_trade_simulation
from strategies.opening_failed_breakout_reversal import DEFAULT_PARAMS, generate_signals


def load_runner():
    runner_path = ROOT / "scripts/research/test_opening_failed_breakout_reversal.py"
    spec = importlib.util.spec_from_file_location("ofbr_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner: {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def main() -> None:
    runner = load_runner()

    symbol = "NVDA"
    timeframe = "5m"
    csv_path = ROOT / "data/processed/NVDA/NVDA_5m_RTH_2021-06-01_2026-05-29.csv"
    variant_name = "both_or15_fail_rr15_extreme"
    window_label = "afternoon_1300_1500"
    exit_policy = "partial_50_at_1r_be"

    entry_start, entry_end = runner.ENTRY_WINDOWS[window_label]
    variant = runner.VARIANTS[variant_name]
    params = {**DEFAULT_PARAMS, **variant, "entry_start": entry_start, "entry_end": entry_end}

    df = runner.load_ohlcv(csv_path)
    signals = generate_signals(df, params)

    out_dir = ROOT / "outputs/research/opening_failed_breakout_reversal/survivor_trades"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = []

    for cost_label, cost_per_side in runner.COST_SCENARIOS.items():
        _, _, trades = run_trade_simulation(
            signals=signals,
            cost_per_side=cost_per_side,
            force_eod_exit=True,
            exit_policy=exit_policy,
        )

        trades = trades.copy()
        trades["symbol"] = symbol
        trades["timeframe"] = timeframe
        trades["strategy_family"] = "opening_failed_breakout_reversal"
        trades["variant"] = variant_name
        trades["window_label"] = window_label
        trades["cost_label"] = cost_label
        trades["cost_per_side"] = cost_per_side
        trades["exit_policy"] = exit_policy

        out_file = out_dir / f"{symbol}_{variant_name}_{window_label}_{cost_label}_trades.csv"
        trades.to_csv(out_file, index=False)
        combined.append(trades)
        print(f"Wrote {out_file} rows={len(trades)}")

    combined_df = pd.concat(combined, ignore_index=True)
    combined_file = out_dir / f"{symbol}_{variant_name}_{window_label}_cost_1x_2x_3x_combined_trades.csv"
    combined_df.to_csv(combined_file, index=False)

    print(f"\nWrote combined file: {combined_file}")
    print("\nColumns:")
    print(list(combined_df.columns))
    print("\nPreview:")
    print(combined_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
