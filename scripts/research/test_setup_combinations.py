from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.metrics import score_report
from engine.trade_simulator import VALID_EXIT_POLICIES, run_trade_simulation
from strategies.zw_vwap_vol_keltner import DEFAULT_PARAMS, generate_signals


COST_SCENARIOS = {
    "cost_1x": 0.0005,
    "cost_2x": 0.0010,
    "cost_3x": 0.0015,
}

DEFAULT_WINDOWS = [
    ("09:30", "10:30", "open_0930_1030"),
    ("09:30", "11:00", "open_0930_1100"),
    ("09:30", "11:30", "open_0930_1130"),
    ("10:00", "11:30", "mid_1000_1130"),
    ("09:30", "16:00", "rth_full"),
]

SETUP_COMBINATIONS = [
    ("A_only", True, False, False),
    ("B_only", False, True, False),
    ("C_only", False, False, True),
    ("A_B", True, True, False),
    ("A_C", True, False, True),
    ("B_C", False, True, True),
    ("A_B_C", True, True, True),
]


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if hasattr(value, "item"):
        try:
            return clean_for_json(value.item())
        except Exception:
            pass
    return value


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: CSV missing columns: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["datetime"] = df["timestamp"]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def parse_windows(raw_windows: list[str] | None) -> list[tuple[str, str, str]]:
    if not raw_windows:
        return DEFAULT_WINDOWS

    windows = []
    for raw in raw_windows:
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 3:
            raise SystemExit(
                "ERROR: --window must use format label,start,end. "
                "Example: --window open_0930_1130,09:30,11:30"
            )
        label, start, end = parts
        windows.append((start, end, label))
    return windows


def summarize_trades(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "total_trades": 0,
            "setup_counts": {},
            "tp_hits": 0,
            "stop_hits": 0,
            "be_exits": 0,
            "eod_exits": 0,
            "partial_taken_count": 0,
            "be_triggered_count": 0,
            "avg_net_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }

    partial_taken_count = 0
    if "partial_taken" in trades.columns:
        partial_taken_count = int(trades["partial_taken"].astype(bool).sum())

    be_triggered_count = 0
    if "be_triggered" in trades.columns:
        be_triggered_count = int(trades["be_triggered"].astype(bool).sum())

    return {
        "total_trades": int(len(trades)),
        "setup_counts": trades["setup"].value_counts().to_dict() if "setup" in trades.columns else {},
        "tp_hits": int((trades["exit_reason"] == "TP").sum()),
        "stop_hits": int((trades["exit_reason"] == "STOP").sum()),
        "be_exits": int((trades["exit_reason"] == "BE").sum()),
        "eod_exits": int((trades["exit_reason"] == "EOD").sum()),
        "partial_taken_count": partial_taken_count,
        "be_triggered_count": be_triggered_count,
        "avg_net_return": float(trades["net_return"].mean()) if "net_return" in trades.columns else 0.0,
        "best_trade": float(trades["net_return"].max()) if "net_return" in trades.columns else 0.0,
        "worst_trade": float(trades["net_return"].min()) if "net_return" in trades.columns else 0.0,
    }


def build_params(enable_a: bool, enable_b: bool, enable_c: bool, session_start: str, session_end: str) -> dict:
    params = DEFAULT_PARAMS.copy()
    params["use_rth"] = True
    params["session_start"] = session_start
    params["session_end"] = session_end
    params["enable_a"] = enable_a
    params["enable_b"] = enable_b
    params["enable_c"] = enable_c
    return params


def run_setup_window_cost(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    strategy: str,
    exit_policy: str,
    setup_label: str,
    enable_a: bool,
    enable_b: bool,
    enable_c: bool,
    window_label: str,
    session_start: str,
    session_end: str,
    cost_label: str,
    cost_per_side: float,
    output_dir: Path,
) -> dict[str, Any]:
    params = build_params(enable_a, enable_b, enable_c, session_start, session_end)
    signals = generate_signals(df, params)

    signal_count = int((signals["entry_signal"] != 0).sum()) if "entry_signal" in signals.columns else 0

    equity, trade_returns, trades = run_trade_simulation(
        signals=signals,
        cost_per_side=cost_per_side,
        force_eod_exit=True,
        exit_policy=exit_policy,
    )

    report = score_report(
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        trade_returns=trade_returns,
    )

    trade_summary = summarize_trades(trades)

    safe_window = window_label.replace(" ", "_").replace(":", "")
    safe_setup = setup_label.replace(" ", "_")
    base = f"{symbol}_{timeframe}_{strategy}_{exit_policy}_{safe_setup}_{safe_window}_{cost_label}"

    trades_path = output_dir / f"{base}_trades.csv"
    equity_path = output_dir / f"{base}_equity.csv"

    trades.to_csv(trades_path, index=False)
    equity.to_frame().to_csv(equity_path, index=True)

    row = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": strategy,
        "exit_policy": exit_policy,
        "setup_combo": setup_label,
        "enable_a": enable_a,
        "enable_b": enable_b,
        "enable_c": enable_c,
        "window_label": window_label,
        "session_start": session_start,
        "session_end": session_end,
        "cost_label": cost_label,
        "cost_per_side": cost_per_side,
        "signal_count": signal_count,
        "trades_path": str(trades_path),
        "equity_path": str(equity_path),
    }

    row.update(report)
    row.update(trade_summary)
    return row


def run_experiment(
    symbol: str,
    timeframe: str,
    csv_path: Path,
    exit_policy: str,
    output_dir: Path,
    windows: list[tuple[str, str, str]],
    min_trades_for_candidate: int,
) -> dict[str, Any]:
    if exit_policy not in VALID_EXIT_POLICIES:
        raise SystemExit(
            f"ERROR: invalid exit_policy={exit_policy}. "
            f"Valid options: {sorted(VALID_EXIT_POLICIES)}"
        )

    df = load_ohlcv(csv_path)
    strategy = "zw_vwap_vol_keltner"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for setup_label, enable_a, enable_b, enable_c in SETUP_COMBINATIONS:
        for session_start, session_end, window_label in windows:
            for cost_label, cost_per_side in COST_SCENARIOS.items():
                rows.append(
                    run_setup_window_cost(
                        df=df,
                        symbol=symbol,
                        timeframe=timeframe,
                        strategy=strategy,
                        exit_policy=exit_policy,
                        setup_label=setup_label,
                        enable_a=enable_a,
                        enable_b=enable_b,
                        enable_c=enable_c,
                        window_label=window_label,
                        session_start=session_start,
                        session_end=session_end,
                        cost_label=cost_label,
                        cost_per_side=cost_per_side,
                        output_dir=output_dir,
                    )
                )

    matrix = pd.DataFrame(rows)
    matrix = matrix.sort_values(
        ["cost_label", "global_score", "profit_factor", "expectancy"],
        ascending=[True, False, False, False],
    )

    matrix_path = output_dir / f"{symbol}_{timeframe}_{exit_policy}_setup_combo_matrix.csv"
    report_path = output_dir / f"{symbol}_{timeframe}_{exit_policy}_setup_combo_report.json"
    matrix.to_csv(matrix_path, index=False)

    cost_1x = matrix[matrix["cost_label"] == "cost_1x"].copy()
    best_cost_1x = (
        cost_1x.sort_values(["global_score", "profit_factor", "expectancy"], ascending=False)
        .head(15)
        .to_dict("records")
    )

    frequency_candidates = cost_1x[
        (cost_1x["total_trades"] >= min_trades_for_candidate)
        & (cost_1x["profit_factor"] > 1.0)
        & (cost_1x["expectancy"] > 0)
    ].copy()

    frequency_candidates = (
        frequency_candidates.sort_values(
            ["profit_factor", "expectancy", "total_trades", "global_score"],
            ascending=[False, False, False, False],
        )
        .head(20)
        .to_dict("records")
    )

    best_overall = (
        matrix.sort_values(["global_score", "profit_factor", "expectancy"], ascending=False)
        .head(20)
        .to_dict("records")
    )

    report = {
        "ok": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "exit_policy": exit_policy,
        "csv_path": str(csv_path),
        "rows_analyzed": int(len(matrix)),
        "setups_analyzed": [
            {"label": label, "enable_a": a, "enable_b": b, "enable_c": c}
            for label, a, b, c in SETUP_COMBINATIONS
        ],
        "windows_analyzed": [
            {"label": label, "session_start": start, "session_end": end}
            for start, end, label in windows
        ],
        "min_trades_for_candidate": min_trades_for_candidate,
        "best_cost_1x": best_cost_1x,
        "frequency_candidates_cost_1x": frequency_candidates,
        "best_overall": best_overall,
        "output_files": {
            "matrix": str(matrix_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(json.dumps(clean_for_json(report), indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run setup-combination experiments")
    parser.add_argument("--symbol", required=True, help="Symbol, example: SPY")
    parser.add_argument("--timeframe", required=True, help="Timeframe, example: 5m")
    parser.add_argument("--csv", required=True, help="Path to processed OHLCV CSV")
    parser.add_argument(
        "--exit-policy",
        default="break_even_1r",
        choices=sorted(VALID_EXIT_POLICIES),
        help="Exit policy to test",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/research/setup_combos",
        help="Output directory for experiment files",
    )
    parser.add_argument(
        "--window",
        action="append",
        default=None,
        help="Optional custom window in format label,start,end. Can be repeated.",
    )
    parser.add_argument(
        "--min-trades-for-candidate",
        type=int,
        default=20,
        help="Minimum trades required to classify as frequency candidate.",
    )

    args = parser.parse_args()
    windows = parse_windows(args.window)

    report = run_experiment(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        csv_path=Path(args.csv),
        exit_policy=args.exit_policy,
        output_dir=Path(args.output_dir),
        windows=windows,
        min_trades_for_candidate=args.min_trades_for_candidate,
    )

    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()
