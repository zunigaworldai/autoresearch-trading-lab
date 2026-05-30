from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]
DEFAULT_INPUT_DIR = Path("outputs/research/setup_combos")
DEFAULT_OUTPUT_DIR = Path("outputs/research/rankings")


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def trading_days_from_csv(csv_path: str) -> int:
    path = Path(csv_path)
    if not path.exists():
        return 100

    try:
        df = pd.read_csv(path, usecols=["timestamp"])
    except Exception:
        return 100

    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return 100

    return int(ts.dt.date.nunique())


def normalize_profit_factor(pf: float) -> float:
    return max(0.0, min(pf, 3.0)) / 3.0


def normalize_expectancy(exp: float) -> float:
    return max(0.0, min(exp, 0.0030)) / 0.0030


def normalize_drawdown(dd: float) -> float:
    return max(0.0, 1.0 - min(max(dd, 0.0), 0.08) / 0.08)


def normalize_frequency(trades_per_month: float) -> float:
    return max(0.0, min(trades_per_month, 20.0)) / 20.0


def production_score(row: dict[str, Any]) -> float:
    pf = safe_float(row.get("profit_factor"))
    exp = safe_float(row.get("expectancy"))
    dd = safe_float(row.get("max_drawdown"))
    tpm = safe_float(row.get("trades_per_month"))
    gs = safe_float(row.get("global_score"))

    return float(
        0.32 * normalize_profit_factor(pf)
        + 0.28 * normalize_expectancy(exp)
        + 0.20 * normalize_drawdown(dd)
        + 0.15 * normalize_frequency(tpm)
        + 0.05 * max(0.0, min(gs, 1.5)) / 1.5
    )


def decision(row: dict[str, Any], min_trades: int, min_pf: float, min_exp: float) -> str:
    trades = safe_int(row.get("total_trades"))
    pf = safe_float(row.get("profit_factor"))
    exp = safe_float(row.get("expectancy"))
    dd = safe_float(row.get("max_drawdown"))

    if trades <= 0:
        return "ignore_no_trades"
    if trades < 5:
        return "ignore_tiny_sample"
    if pf <= 1.0 or exp <= 0:
        return "reject_no_edge"
    if dd >= 0.08:
        return "watch_high_drawdown"
    if trades < min_trades:
        return "watch_low_frequency"
    if pf >= min_pf and exp >= min_exp:
        return "candidate_validate_oos"
    return "watch_edge_moderate"


def flatten_report(
    report_path: Path,
    min_trades: int,
    min_pf: float,
    min_exp: float,
) -> list[dict[str, Any]]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    symbol = str(payload.get("symbol", "")).upper()
    timeframe = str(payload.get("timeframe", ""))
    exit_policy = str(payload.get("exit_policy", ""))
    csv_path = str(payload.get("csv_path", ""))
    trading_days = trading_days_from_csv(csv_path)

    matrix_path = Path(payload.get("output_files", {}).get("matrix", ""))
    if matrix_path.exists():
        source_rows = pd.read_csv(matrix_path).to_dict("records")
    else:
        source_rows = payload.get("best_cost_1x", [])

    rows = []
    for item in source_rows:
        total_trades = safe_int(item.get("total_trades", item.get("trades", 0)))
        trades_per_day = total_trades / trading_days if trading_days else 0.0
        trades_per_week = trades_per_day * 5.0
        trades_per_month = trades_per_day * 21.0

        out = {
            "source_report": str(report_path),
            "symbol": symbol,
            "timeframe": timeframe,
            "exit_policy": exit_policy,
            "setup_combo": str(item.get("setup_combo", "")),
            "window_label": str(item.get("window_label", "")),
            "session_start": str(item.get("session_start", "")),
            "session_end": str(item.get("session_end", "")),
            "cost_label": str(item.get("cost_label", "")),
            "cost_per_side": safe_float(item.get("cost_per_side")),
            "trading_days": trading_days,
            "trades_per_day": trades_per_day,
            "trades_per_week": trades_per_week,
            "trades_per_month": trades_per_month,
            "profit_factor": safe_float(item.get("profit_factor")),
            "expectancy": safe_float(item.get("expectancy")),
            "global_score": safe_float(item.get("global_score")),
            "max_drawdown": safe_float(item.get("max_drawdown")),
            "win_rate": safe_float(item.get("win_rate")),
            "total_trades": total_trades,
            "tp_hits": safe_int(item.get("tp_hits")),
            "stop_hits": safe_int(item.get("stop_hits")),
            "be_exits": safe_int(item.get("be_exits")),
            "eod_exits": safe_int(item.get("eod_exits")),
            "be_triggered_count": safe_int(item.get("be_triggered_count")),
        }

        out["is_frequency_candidate"] = bool(
            total_trades >= min_trades
            and out["profit_factor"] > 1.0
            and out["expectancy"] > 0
        )
        out["production_score"] = production_score(out)
        out["decision"] = decision(out, min_trades=min_trades, min_pf=min_pf, min_exp=min_exp)
        rows.append(out)

    return rows


def load_rows(
    symbols: list[str],
    timeframe: str,
    exit_policy: str,
    input_dir: Path,
    min_trades: int,
    min_pf: float,
    min_exp: float,
) -> pd.DataFrame:
    rows = []

    for symbol in symbols:
        path = input_dir / f"{symbol}_{timeframe}_{exit_policy}_setup_combo_report.json"
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        rows.extend(flatten_report(path, min_trades=min_trades, min_pf=min_pf, min_exp=min_exp))

    if not rows:
        raise SystemExit("ERROR: no rows loaded.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["production_score", "profit_factor", "expectancy", "total_trades", "max_drawdown"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    return df


def build_report(df: pd.DataFrame, symbols: list[str], timeframe: str, exit_policy: str) -> dict[str, Any]:
    cost_1x = df[df["cost_label"] == "cost_1x"].copy()
    candidates = cost_1x[cost_1x["decision"] == "candidate_validate_oos"].copy()

    top_all = (
        cost_1x.sort_values(
            ["production_score", "profit_factor", "expectancy", "total_trades"],
            ascending=[False, False, False, False],
        )
        .head(30)
        .to_dict("records")
    )

    top_candidates = (
        candidates.sort_values(
            ["production_score", "profit_factor", "expectancy", "total_trades"],
            ascending=[False, False, False, False],
        )
        .head(30)
        .to_dict("records")
    )

    by_symbol = {}
    for symbol, group in cost_1x.groupby("symbol"):
        by_symbol[symbol] = (
            group.sort_values(
                ["production_score", "profit_factor", "expectancy", "total_trades"],
                ascending=[False, False, False, False],
            )
            .head(5)
            .to_dict("records")
        )

    recommendations = []
    if top_candidates:
        best = top_candidates[0]
        recommendations.append(
            f"Top validation candidate: {best['symbol']} {best['timeframe']} "
            f"{best['setup_combo']} {best['window_label']} PF={best['profit_factor']:.4f}, "
            f"expectancy={best['expectancy']:.6f}, trades={best['total_trades']}, "
            f"trades/month={best['trades_per_month']:.2f}."
        )
    else:
        recommendations.append("No setup meets current validation threshold.")

    recommendations.append("Next step: run IS/OOS split Jan-Mar vs Apr-May for candidates.")
    recommendations.append("Then stress test cost_2x/cost_3x before paper/live.")

    return {
        "ok": True,
        "symbols": symbols,
        "timeframe": timeframe,
        "exit_policy": exit_policy,
        "rows_total": int(len(df)),
        "cost_1x_rows": int(len(cost_1x)),
        "decision_counts_cost_1x": cost_1x["decision"].value_counts().to_dict(),
        "top_candidates_cost_1x": top_candidates,
        "top_all_cost_1x": top_all,
        "best_by_symbol_cost_1x": by_symbol,
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank multi-symbol setup-combo results.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--exit-policy", default="break_even_1r")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-pf", type=float, default=1.25)
    parser.add_argument("--min-expectancy", type=float, default=0.00025)

    args = parser.parse_args()
    symbols = [s.upper() for s in args.symbols]

    df = load_rows(
        symbols=symbols,
        timeframe=args.timeframe,
        exit_policy=args.exit_policy,
        input_dir=Path(args.input_dir),
        min_trades=args.min_trades,
        min_pf=args.min_pf,
        min_exp=args.min_expectancy,
    )

    report = build_report(df, symbols=symbols, timeframe=args.timeframe, exit_policy=args.exit_policy)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / f"multi_symbol_{args.timeframe}_{args.exit_policy}_setup_ranking.csv"
    report_path = output_dir / f"multi_symbol_{args.timeframe}_{args.exit_policy}_setup_ranking_report.json"

    df.to_csv(matrix_path, index=False)
    report["output_files"] = {
        "matrix": str(matrix_path),
        "report": str(report_path),
    }
    report_path.write_text(json.dumps(clean_for_json(report), indent=2), encoding="utf-8")

    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()
