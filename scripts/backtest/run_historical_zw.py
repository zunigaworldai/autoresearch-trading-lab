from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.metrics import score_report
from engine.trade_simulator import run_trade_simulation, VALID_EXIT_POLICIES
from strategies.zw_vwap_vol_keltner import DEFAULT_PARAMS, generate_signals
from scripts.data.validate_ohlcv import validate_file


COST_SCENARIOS = {
    "cost_1x": 0.0005,
    "cost_2x": 0.0010,
    "cost_3x": 0.0015,
}


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["datetime"] = df["timestamp"]

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def policy_suffix(exit_policy: str) -> str:
    if exit_policy == "fixed":
        return ""

    return f"_{exit_policy}"


def enrich_report(
    report: dict,
    trades: pd.DataFrame,
    symbol: str,
    timeframe: str,
    cost_label: str,
    cost_per_side: float,
    csv_path: Path,
    exit_policy: str,
) -> dict:
    report = dict(report)

    report["symbol"] = symbol
    report["timeframe"] = timeframe
    report["strategy"] = "zw_vwap_vol_keltner"
    report["exit_policy"] = exit_policy
    report["cost_label"] = cost_label
    report["cost_per_side"] = cost_per_side
    report["csv_path"] = str(csv_path)
    report["total_trades"] = int(len(trades))

    if len(trades):
        report["setup_counts"] = trades["setup"].value_counts().to_dict()
        report["tp_hits"] = int((trades["exit_reason"] == "TP").sum())
        report["stop_hits"] = int((trades["exit_reason"] == "STOP").sum())
        report["be_exits"] = int((trades["exit_reason"] == "BE").sum())
        report["eod_exits"] = int((trades["exit_reason"] == "EOD").sum())
        report["be_triggered_count"] = int(trades.get("be_triggered", pd.Series(dtype=bool)).sum())
        report["avg_net_return"] = float(trades["net_return"].mean())
        report["best_trade"] = float(trades["net_return"].max())
        report["worst_trade"] = float(trades["net_return"].min())
    else:
        report["setup_counts"] = {}
        report["tp_hits"] = 0
        report["stop_hits"] = 0
        report["be_exits"] = 0
        report["eod_exits"] = 0
        report["be_triggered_count"] = 0
        report["avg_net_return"] = 0.0
        report["best_trade"] = 0.0
        report["worst_trade"] = 0.0

    return report


def run_backtest_for_cost(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    csv_path: Path,
    cost_label: str,
    cost_per_side: float,
    params: dict,
    output_dir: Path,
    exit_policy: str,
) -> dict:
    signals = generate_signals(df, params)

    equity, trade_returns, trades = run_trade_simulation(
        signals,
        cost_per_side=cost_per_side,
        force_eod_exit=True,
        exit_policy=exit_policy,
    )

    report = score_report(
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        trade_returns=trade_returns,
    )

    report = enrich_report(
        report=report,
        trades=trades,
        symbol=symbol,
        timeframe=timeframe,
        cost_label=cost_label,
        cost_per_side=cost_per_side,
        csv_path=csv_path,
        exit_policy=exit_policy,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = policy_suffix(exit_policy)

    trades_path = output_dir / f"{symbol}_{timeframe}_zw_vwap_vol_keltner{suffix}_{cost_label}_trades.csv"
    equity_path = output_dir / f"{symbol}_{timeframe}_zw_vwap_vol_keltner{suffix}_{cost_label}_equity.csv"

    trades.to_csv(trades_path, index=False)
    equity.to_frame().to_csv(equity_path, index=True)

    report["trades_path"] = str(trades_path)
    report["equity_path"] = str(equity_path)

    return report


def run_historical_backtest(
    symbol: str,
    timeframe: str,
    csv_path: Path,
    start: str | None,
    end: str | None,
    output_root: Path,
    exit_policy: str,
) -> dict:
    if exit_policy not in VALID_EXIT_POLICIES:
        raise SystemExit(
            f"ERROR: invalid exit_policy={exit_policy}. Valid options: {sorted(VALID_EXIT_POLICIES)}"
        )

    validation = validate_file(csv_path, start=start, end=end)

    df = load_ohlcv(csv_path)

    params = DEFAULT_PARAMS.copy()
    params["use_rth"] = True

    output_dir = output_root / symbol

    results = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy": "zw_vwap_vol_keltner",
        "exit_policy": exit_policy,
        "csv_path": str(csv_path),
        "validation": validation,
        "cost_results": {},
    }

    for cost_label, cost_per_side in COST_SCENARIOS.items():
        results["cost_results"][cost_label] = run_backtest_for_cost(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            csv_path=csv_path,
            cost_label=cost_label,
            cost_per_side=cost_per_side,
            params=params,
            output_dir=output_dir,
            exit_policy=exit_policy,
        )

    suffix = policy_suffix(exit_policy)
    result_path = output_dir / f"{symbol}_{timeframe}_zw_vwap_vol_keltner{suffix}_result.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(results, indent=2))

    results["result_path"] = str(result_path)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical ZW VWAP VOL KELTNER backtest")
    parser.add_argument("--symbol", required=True, help="Symbol, example: SPY")
    parser.add_argument("--timeframe", required=True, help="Timeframe, example: 3m")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--start", default=None, help="Expected start date, example: 2026-01-01")
    parser.add_argument("--end", default=None, help="Expected end date, example: 2026-05-27")
    parser.add_argument("--output-root", default="outputs/backtests", help="Output root folder")
    parser.add_argument(
        "--exit-policy",
        default="fixed",
        choices=sorted(VALID_EXIT_POLICIES),
        help="Exit policy: fixed or break_even_1r",
    )

    args = parser.parse_args()

    results = run_historical_backtest(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        csv_path=Path(args.csv),
        start=args.start,
        end=args.end,
        output_root=Path(args.output_root),
        exit_policy=args.exit_policy,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()