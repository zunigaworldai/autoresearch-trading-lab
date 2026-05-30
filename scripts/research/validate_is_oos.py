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

DEFAULT_RANKING = Path("outputs/research/rankings/multi_symbol_5m_break_even_1r_setup_ranking.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/validation")
DEFAULT_DATA_DIR = Path("data/processed")

PERIODS = {
    "IS": ("2026-01-01", "2026-03-31"),
    "OOS": ("2026-04-01", "2026-05-31"),
}


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


def setup_flags(setup_combo: str) -> tuple[bool, bool, bool]:
    setup_combo = setup_combo.strip()

    mapping = {
        "A_only": (True, False, False),
        "B_only": (False, True, False),
        "C_only": (False, False, True),
        "A_B": (True, True, False),
        "A_C": (True, False, True),
        "B_C": (False, True, True),
        "A_B_C": (True, True, True),
    }

    if setup_combo not in mapping:
        raise ValueError(f"Unknown setup_combo: {setup_combo}")

    return mapping[setup_combo]


def find_csv(symbol: str, timeframe: str, data_dir: Path) -> Path:
    symbol_dir = data_dir / symbol
    pattern = f"{symbol}_{timeframe}_RTH_2026-01-01_2026-05-*.csv"
    matches = sorted(symbol_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No processed CSV found for {symbol} {timeframe}: {symbol_dir}/{pattern}")

    # Prefer latest end-date file by sorted name.
    return matches[-1]


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: CSV missing columns {sorted(missing)}: {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["datetime"] = df["timestamp"]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    return df


def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)

    out = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
    out = out.reset_index(drop=True)

    return out


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
            "largest_win_share": 0.0,
        }

    partial_taken_count = 0
    if "partial_taken" in trades.columns:
        partial_taken_count = int(trades["partial_taken"].astype(bool).sum())

    be_triggered_count = 0
    if "be_triggered" in trades.columns:
        be_triggered_count = int(trades["be_triggered"].astype(bool).sum())

    net = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    gross_profit = float(net[net > 0].sum()) if len(net) else 0.0
    best_trade = float(net.max()) if len(net) else 0.0
    largest_win_share = best_trade / gross_profit if gross_profit > 0 else 0.0

    return {
        "total_trades": int(len(trades)),
        "setup_counts": trades["setup"].value_counts().to_dict() if "setup" in trades.columns else {},
        "tp_hits": int((trades["exit_reason"] == "TP").sum()),
        "stop_hits": int((trades["exit_reason"] == "STOP").sum()),
        "be_exits": int((trades["exit_reason"] == "BE").sum()),
        "eod_exits": int((trades["exit_reason"] == "EOD").sum()),
        "partial_taken_count": partial_taken_count,
        "be_triggered_count": be_triggered_count,
        "avg_net_return": float(net.mean()) if len(net) else 0.0,
        "best_trade": best_trade,
        "worst_trade": float(net.min()) if len(net) else 0.0,
        "largest_win_share": float(largest_win_share),
    }


def build_params(candidate: dict[str, Any]) -> dict:
    enable_a, enable_b, enable_c = setup_flags(str(candidate["setup_combo"]))

    params = DEFAULT_PARAMS.copy()
    params["use_rth"] = True
    params["session_start"] = str(candidate["session_start"])
    params["session_end"] = str(candidate["session_end"])
    params["enable_a"] = enable_a
    params["enable_b"] = enable_b
    params["enable_c"] = enable_c

    return params


def run_candidate_period_cost(
    candidate: dict[str, Any],
    df: pd.DataFrame,
    period_label: str,
    period_start: str,
    period_end: str,
    cost_label: str,
    cost_per_side: float,
    output_dir: Path,
) -> dict[str, Any]:
    symbol = str(candidate["symbol"])
    timeframe = str(candidate["timeframe"])
    exit_policy = str(candidate["exit_policy"])

    if exit_policy not in VALID_EXIT_POLICIES:
        raise SystemExit(f"Invalid exit_policy in candidate: {exit_policy}")

    params = build_params(candidate)

    period_df = filter_period(df, period_start, period_end)

    if period_df.empty:
        raise SystemExit(f"No data for {symbol} {timeframe} {period_label} {period_start} to {period_end}")

    signals = generate_signals(period_df, params)

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

    safe_name = (
        f"{symbol}_{timeframe}_{exit_policy}_{candidate['setup_combo']}_"
        f"{candidate['window_label']}_{period_label}_{cost_label}"
    ).replace(":", "").replace(" ", "_")

    trades_path = output_dir / f"{safe_name}_trades.csv"
    equity_path = output_dir / f"{safe_name}_equity.csv"

    trades.to_csv(trades_path, index=False)
    equity.to_frame().to_csv(equity_path, index=True)

    row = {
        "symbol": symbol,
        "timeframe": timeframe,
        "exit_policy": exit_policy,
        "setup_combo": str(candidate["setup_combo"]),
        "window_label": str(candidate["window_label"]),
        "session_start": str(candidate["session_start"]),
        "session_end": str(candidate["session_end"]),
        "period": period_label,
        "period_start": period_start,
        "period_end": period_end,
        "cost_label": cost_label,
        "cost_per_side": cost_per_side,
        "signal_count": signal_count,
        "bars": int(len(period_df)),
        "trading_days": int(period_df["timestamp"].dt.date.nunique()),
        "trades_path": str(trades_path),
        "equity_path": str(equity_path),
    }

    row.update(report)
    row.update(trade_summary)

    row["trades_per_day"] = (
        row["total_trades"] / row["trading_days"] if row["trading_days"] else 0.0
    )
    row["trades_per_month"] = row["trades_per_day"] * 21.0

    return row


def load_candidates(
    ranking_path: Path,
    max_candidates: int,
    symbols: list[str] | None,
    decisions: list[str],
) -> list[dict[str, Any]]:
    if not ranking_path.exists():
        raise SystemExit(f"ERROR: ranking file not found: {ranking_path}")

    df = pd.read_csv(ranking_path)

    if symbols:
        wanted = {s.upper() for s in symbols}
        df = df[df["symbol"].astype(str).str.upper().isin(wanted)].copy()

    df = df[df["cost_label"] == "cost_1x"].copy()
    df = df[df["decision"].isin(decisions)].copy()

    df = df.sort_values(
        ["production_score", "profit_factor", "expectancy", "total_trades"],
        ascending=[False, False, False, False],
    )

    df = df.head(max_candidates)

    return df.to_dict("records")


def aggregate_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    key = {
        "symbol": rows[0]["symbol"],
        "timeframe": rows[0]["timeframe"],
        "exit_policy": rows[0]["exit_policy"],
        "setup_combo": rows[0]["setup_combo"],
        "window_label": rows[0]["window_label"],
        "session_start": rows[0]["session_start"],
        "session_end": rows[0]["session_end"],
    }

    by_period_cost = {
        f"{row['period']}_{row['cost_label']}": row
        for row in rows
    }

    is_1x = by_period_cost.get("IS_cost_1x", {})
    oos_1x = by_period_cost.get("OOS_cost_1x", {})
    oos_2x = by_period_cost.get("OOS_cost_2x", {})
    oos_3x = by_period_cost.get("OOS_cost_3x", {})

    pass_is = (
        safe_float(is_1x.get("profit_factor")) > 1.0
        and safe_float(is_1x.get("expectancy")) > 0
        and safe_int(is_1x.get("total_trades")) >= 5
    )

    pass_oos = (
        safe_float(oos_1x.get("profit_factor")) > 1.0
        and safe_float(oos_1x.get("expectancy")) > 0
        and safe_int(oos_1x.get("total_trades")) >= 5
        and safe_float(oos_1x.get("largest_win_share")) <= 0.60
    )

    pass_oos_2x = (
        safe_float(oos_2x.get("profit_factor")) > 1.0
        and safe_float(oos_2x.get("expectancy")) > 0
    )

    pass_oos_3x = (
        safe_float(oos_3x.get("profit_factor")) > 1.0
        and safe_float(oos_3x.get("expectancy")) > 0
    )

    if pass_is and pass_oos and pass_oos_2x:
        decision = "candidate_paper_watch"
    elif pass_is and pass_oos:
        decision = "watch_cost_sensitive"
    elif pass_is and not pass_oos:
        decision = "fail_oos"
    else:
        decision = "fail_is"

    return {
        **key,
        "IS_trades": safe_int(is_1x.get("total_trades")),
        "IS_pf": safe_float(is_1x.get("profit_factor")),
        "IS_expectancy": safe_float(is_1x.get("expectancy")),
        "IS_dd": safe_float(is_1x.get("max_drawdown")),
        "IS_score": safe_float(is_1x.get("global_score")),
        "OOS_trades": safe_int(oos_1x.get("total_trades")),
        "OOS_trades_per_month": safe_float(oos_1x.get("trades_per_month")),
        "OOS_pf": safe_float(oos_1x.get("profit_factor")),
        "OOS_expectancy": safe_float(oos_1x.get("expectancy")),
        "OOS_dd": safe_float(oos_1x.get("max_drawdown")),
        "OOS_score": safe_float(oos_1x.get("global_score")),
        "OOS_largest_win_share": safe_float(oos_1x.get("largest_win_share")),
        "OOS_2x_pf": safe_float(oos_2x.get("profit_factor")),
        "OOS_2x_expectancy": safe_float(oos_2x.get("expectancy")),
        "OOS_3x_pf": safe_float(oos_3x.get("profit_factor")),
        "OOS_3x_expectancy": safe_float(oos_3x.get("expectancy")),
        "pass_is": pass_is,
        "pass_oos": pass_oos,
        "pass_oos_2x": pass_oos_2x,
        "pass_oos_3x": pass_oos_3x,
        "decision": decision,
    }


def run_validation(
    ranking_path: Path,
    data_dir: Path,
    output_dir: Path,
    max_candidates: int,
    symbols: list[str] | None,
    decisions: list[str],
) -> dict[str, Any]:
    candidates = load_candidates(
        ranking_path=ranking_path,
        max_candidates=max_candidates,
        symbols=symbols,
        decisions=decisions,
    )

    if not candidates:
        raise SystemExit("ERROR: no candidates selected for IS/OOS validation.")

    output_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    aggregate_rows = []

    for candidate in candidates:
        symbol = str(candidate["symbol"]).upper()
        timeframe = str(candidate["timeframe"])

        key = (symbol, timeframe)

        if key not in cache:
            csv_path = find_csv(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
            cache[key] = load_ohlcv(csv_path)

        df = cache[key]
        candidate_rows = []

        for period_label, (period_start, period_end) in PERIODS.items():
            for cost_label, cost_per_side in COST_SCENARIOS.items():
                row = run_candidate_period_cost(
                    candidate=candidate,
                    df=df,
                    period_label=period_label,
                    period_start=period_start,
                    period_end=period_end,
                    cost_label=cost_label,
                    cost_per_side=cost_per_side,
                    output_dir=output_dir,
                )
                rows.append(row)
                candidate_rows.append(row)

        aggregate_rows.append(aggregate_candidate(candidate_rows))

    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(aggregate_rows)

    detail_path = output_dir / "is_oos_validation_detail.csv"
    summary_path = output_dir / "is_oos_validation_summary.csv"
    report_path = output_dir / "is_oos_validation_report.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    summary_sorted = summary.sort_values(
        ["decision", "OOS_pf", "OOS_expectancy", "OOS_trades"],
        ascending=[True, False, False, False],
    )

    report = {
        "ok": True,
        "ranking_path": str(ranking_path),
        "candidates_tested": int(len(candidates)),
        "periods": PERIODS,
        "decision_counts": summary["decision"].value_counts().to_dict(),
        "top_summary": summary_sorted.head(30).to_dict("records"),
        "output_files": {
            "detail": str(detail_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "rules": {
            "pass_is": "IS cost_1x PF>1, expectancy>0, trades>=5",
            "pass_oos": "OOS cost_1x PF>1, expectancy>0, trades>=5, largest_win_share<=0.60",
            "candidate_paper_watch": "pass IS, pass OOS, and pass OOS cost_2x",
            "watch_cost_sensitive": "pass IS and OOS cost_1x but fail OOS cost_2x",
        },
    }

    report_path.write_text(json.dumps(clean_for_json(report), indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ranked setup candidates with IS/OOS split.")
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument(
        "--decisions",
        nargs="+",
        default=["candidate_validate_oos"],
        help="Candidate decision labels to include from ranking file.",
    )

    args = parser.parse_args()

    report = run_validation(
        ranking_path=Path(args.ranking),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_candidates=args.max_candidates,
        symbols=args.symbols,
        decisions=args.decisions,
    )

    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()
