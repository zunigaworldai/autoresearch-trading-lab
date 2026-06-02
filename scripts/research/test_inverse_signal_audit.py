from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.trade_simulator import VALID_EXIT_POLICIES, run_trade_simulation


FAMILY_RUNNERS = {
    "vwap_mean_reversion": ROOT / "scripts/research/test_vwap_mean_reversion.py",
    "rsi_bollinger_mean_reversion": ROOT / "scripts/research/test_rsi_bollinger_mean_reversion.py",
}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if hasattr(value, "item"):
        try:
            return clean(value.item())
        except Exception:
            pass
    return value


def load_runner(path: Path):
    if not path.exists():
        raise SystemExit(f"ERROR: runner not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: could not load runner module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invert_signals(signals: pd.DataFrame) -> pd.DataFrame:
    required = {"entry_signal", "exit_stop", "exit_tp"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Missing required signal columns: {sorted(missing)}")

    out = signals.copy()
    mask = out["entry_signal"] != 0

    old_stop = out.loc[mask, "exit_stop"].copy()
    old_tp = out.loc[mask, "exit_tp"].copy()

    out.loc[mask, "entry_signal"] = -out.loc[mask, "entry_signal"]
    out.loc[mask, "exit_stop"] = old_tp
    out.loc[mask, "exit_tp"] = old_stop

    if "setup" in out.columns:
        out.loc[mask, "setup"] = out.loc[mask, "setup"].astype(str) + "_INVERSE"
    else:
        out["setup"] = ""
        out.loc[mask, "setup"] = "INVERSE"

    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.exit_policy not in VALID_EXIT_POLICIES:
        raise SystemExit(f"ERROR: invalid exit policy {args.exit_policy}")

    runner = load_runner(FAMILY_RUNNERS[args.family])

    unknown_variants = sorted(set(args.variants) - set(runner.VARIANTS))
    if unknown_variants:
        raise SystemExit(f"ERROR: unknown variants {unknown_variants}. Valid: {sorted(runner.VARIANTS)}")

    unknown_windows = sorted(set(args.entry_windows) - set(runner.ENTRY_WINDOWS))
    if unknown_windows:
        raise SystemExit(f"ERROR: unknown windows {unknown_windows}. Valid: {sorted(runner.ENTRY_WINDOWS)}")

    unknown_costs = sorted(set(args.cost_labels) - set(runner.COST_SCENARIOS))
    if unknown_costs:
        raise SystemExit(f"ERROR: unknown cost labels {unknown_costs}. Valid: {sorted(runner.COST_SCENARIOS)}")

    df = runner.load_ohlcv(Path(args.csv))
    dataset_start = df["timestamp"].min()
    dataset_end = df["timestamp"].max()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for variant_name in args.variants:
        variant = runner.VARIANTS[variant_name]
        for window_label in args.entry_windows:
            entry_start, entry_end = runner.ENTRY_WINDOWS[window_label]
            params = {
                **runner.DEFAULT_PARAMS,
                **variant,
                "entry_start": entry_start,
                "entry_end": entry_end,
            }

            normal_signals = runner.generate_signals(df, params)
            inverse_signals = invert_signals(normal_signals)

            for cost_label in args.cost_labels:
                cost_per_side = runner.COST_SCENARIOS[cost_label]
                _, _, trades = run_trade_simulation(
                    signals=inverse_signals,
                    cost_per_side=cost_per_side,
                    force_eod_exit=True,
                    exit_policy=args.exit_policy,
                )

                row = runner.evaluate_trades(
                    symbol=args.symbol.upper(),
                    timeframe=args.timeframe,
                    variant=f"inverse_{variant_name}",
                    window_label=window_label,
                    entry_start=entry_start,
                    entry_end=entry_end,
                    exit_policy=args.exit_policy,
                    cost_label=cost_label,
                    cost_per_side=cost_per_side,
                    trades=trades,
                    dataset_start=dataset_start,
                    dataset_end=dataset_end,
                    min_trades=args.min_trades,
                )
                row["source_family"] = args.family
                row["audit_type"] = "inverse_signal"
                rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["cost_label", "strategy_score", "pf", "expectancy", "trades"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    safe = f"{args.symbol.upper()}_{args.timeframe}_{args.exit_policy}_{args.family}_inverse_signal_audit"
    csv_path = output_dir / f"{safe}.csv"
    report_path = output_dir / f"{safe}.json"

    result.to_csv(csv_path, index=False)

    cost_1x = result[result["cost_label"] == "cost_1x"].copy()
    report = {
        "ok": True,
        "symbol": args.symbol.upper(),
        "timeframe": args.timeframe,
        "family": args.family,
        "audit_type": "inverse_signal",
        "exit_policy": args.exit_policy,
        "csv": args.csv,
        "variants": args.variants,
        "entry_windows": [
            {"label": label, "start": runner.ENTRY_WINDOWS[label][0], "end": runner.ENTRY_WINDOWS[label][1]}
            for label in args.entry_windows
        ],
        "cost_labels": args.cost_labels,
        "rows": int(len(result)),
        "decision_counts_cost_1x": cost_1x["decision"].value_counts().to_dict(),
        "top_cost_1x": cost_1x.head(args.top_n).to_dict("records"),
        "output_files": {
            "csv": str(csv_path),
            "report": str(report_path),
        },
        "warning": "Diagnostic only. Inverse-signal results are not production candidates without fresh hypothesis, OOS, walk-forward, cost stress, and portfolio checks.",
    }

    report_path.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse signal audit for failed mean-reversion strategies.")
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_RUNNERS))
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--exit-policy", default="fixed", choices=sorted(VALID_EXIT_POLICIES))
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--entry-windows", nargs="+", required=True)
    parser.add_argument("--cost-labels", nargs="+", default=["cost_1x"])
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output-dir", default="outputs/research/inverse_signal_audit")
    args = parser.parse_args()

    print(json.dumps(clean(run(args)), indent=2))


if __name__ == "__main__":
    main()
