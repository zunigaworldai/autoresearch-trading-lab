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

DEFAULT_RANKING = Path("outputs/research/rankings_5y_diagnostic/multi_symbol_5m_break_even_1r_setup_ranking.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/research/yearly_diagnostics")
DEFAULT_DATA_DIR = Path("data/processed")

DEFAULT_DECISIONS = [
    "candidate_validate_oos",
    "watch_high_drawdown",
    "watch_edge_moderate",
    "watch_low_frequency",
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


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: CSV not found: {path}")

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


def audit_csv_days(path: Path) -> tuple[int, int]:
    try:
        df = pd.read_csv(path, usecols=["timestamp"])
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    except Exception:
        return (0, 0)
    if ts.empty:
        return (0, 0)
    return (int(ts.dt.date.nunique()), int(len(ts)))


def find_best_processed_csv(symbol: str, timeframe: str, data_dir: Path) -> Path:
    symbol = symbol.upper()
    symbol_dir = data_dir / symbol
    files = sorted(symbol_dir.glob(f"{symbol}_{timeframe}_RTH_*.csv"))
    if not files:
        raise FileNotFoundError(f"No processed RTH CSV found for {symbol} {timeframe} in {symbol_dir}")
    ranked = []
    for path in files:
        days, rows = audit_csv_days(path)
        ranked.append((days, rows, path))
    ranked.sort(key=lambda x: (x[0], x[1], str(x[2])), reverse=True)
    return ranked[0][2]


def filter_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["timestamp"].dt.year == year].copy().reset_index(drop=True)


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


def summarize_trades(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "total_trades": 0,
            "tp_hits": 0,
            "stop_hits": 0,
            "be_exits": 0,
            "eod_exits": 0,
            "be_triggered_count": 0,
            "avg_net_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "largest_win_share": 0.0,
        }

    net = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    gross_profit = float(net[net > 0].sum()) if len(net) else 0.0
    best_trade = float(net.max()) if len(net) else 0.0
    largest_win_share = best_trade / gross_profit if gross_profit > 0 else 0.0

    be_triggered_count = 0
    if "be_triggered" in trades.columns:
        be_triggered_count = int(trades["be_triggered"].astype(bool).sum())

    return {
        "total_trades": int(len(trades)),
        "tp_hits": int((trades["exit_reason"] == "TP").sum()) if "exit_reason" in trades.columns else 0,
        "stop_hits": int((trades["exit_reason"] == "STOP").sum()) if "exit_reason" in trades.columns else 0,
        "be_exits": int((trades["exit_reason"] == "BE").sum()) if "exit_reason" in trades.columns else 0,
        "eod_exits": int((trades["exit_reason"] == "EOD").sum()) if "exit_reason" in trades.columns else 0,
        "be_triggered_count": be_triggered_count,
        "avg_net_return": float(net.mean()) if len(net) else 0.0,
        "best_trade": best_trade,
        "worst_trade": float(net.min()) if len(net) else 0.0,
        "largest_win_share": float(largest_win_share),
    }


def run_candidate_year_cost(candidate: dict[str, Any], df: pd.DataFrame, year: int, cost_label: str, cost_per_side: float) -> dict[str, Any]:
    symbol = str(candidate["symbol"]).upper()
    timeframe = str(candidate["timeframe"])
    exit_policy = str(candidate["exit_policy"])

    if exit_policy not in VALID_EXIT_POLICIES:
        raise SystemExit(f"Invalid exit_policy={exit_policy}. Valid: {sorted(VALID_EXIT_POLICIES)}")

    year_df = filter_year(df, year)
    base = {
        "symbol": symbol,
        "timeframe": timeframe,
        "exit_policy": exit_policy,
        "setup_combo": str(candidate["setup_combo"]),
        "window_label": str(candidate["window_label"]),
        "session_start": str(candidate["session_start"]),
        "session_end": str(candidate["session_end"]),
        "year": int(year),
        "cost_label": cost_label,
        "cost_per_side": float(cost_per_side),
        "bars": int(len(year_df)),
        "trading_days": int(year_df["timestamp"].dt.date.nunique()) if not year_df.empty else 0,
    }

    if year_df.empty:
        base.update({
            "signal_count": 0,
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "global_score": 0.0,
        })
        base.update(summarize_trades(pd.DataFrame()))
        base["trades_per_month"] = 0.0
        base["year_pass_edge"] = False
        return base

    params = build_params(candidate)
    signals = generate_signals(year_df, params)
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
    base["signal_count"] = signal_count
    base.update(report)
    base.update(trade_summary)
    base["trades_per_month"] = base["total_trades"] / base["trading_days"] * 21.0 if base["trading_days"] else 0.0
    base["year_pass_edge"] = (
        safe_int(base["total_trades"]) >= 5
        and safe_float(base["profit_factor"]) > 1.0
        and safe_float(base["expectancy"]) > 0
    )
    return base


def load_candidates(ranking_path: Path, max_candidates: int, decisions: list[str], include_reject_no_edge_top: int) -> list[dict[str, Any]]:
    if not ranking_path.exists():
        raise SystemExit(f"ERROR: ranking CSV not found: {ranking_path}")

    df = pd.read_csv(ranking_path)
    if "cost_label" in df.columns:
        df = df[df["cost_label"] == "cost_1x"].copy()
    if "production_score" not in df.columns:
        df["production_score"] = 0.0

    selected = df[df["decision"].isin(decisions)].copy()
    selected = selected.sort_values(
        ["production_score", "profit_factor", "expectancy", "total_trades"],
        ascending=[False, False, False, False],
    )

    if include_reject_no_edge_top > 0:
        reject = df[df["decision"] == "reject_no_edge"].copy()
        reject = reject.sort_values(
            ["production_score", "profit_factor", "expectancy", "total_trades"],
            ascending=[False, False, False, False],
        ).head(include_reject_no_edge_top)
        selected = pd.concat([selected, reject], ignore_index=True)

    selected = selected.drop_duplicates(
        ["symbol", "timeframe", "exit_policy", "setup_combo", "window_label", "session_start", "session_end"]
    )
    selected = selected.head(max_candidates)
    return selected.to_dict("records")


def aggregate_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cost_1x = [row for row in rows if row["cost_label"] == "cost_1x"]
    cost_2x = [row for row in rows if row["cost_label"] == "cost_2x"]
    cost_3x = [row for row in rows if row["cost_label"] == "cost_3x"]
    first = rows[0]

    valid_years = [row for row in cost_1x if row["trading_days"] > 0]
    positive_years = [row for row in valid_years if bool(row["year_pass_edge"])]
    negative_years = [
        row for row in valid_years
        if row["total_trades"] >= 5 and (row["profit_factor"] <= 1.0 or row["expectancy"] <= 0)
    ]

    total_trades = sum(safe_int(row["total_trades"]) for row in cost_1x)
    years_with_trades = sum(1 for row in valid_years if safe_int(row["total_trades"]) > 0)
    years_pass_edge = len(positive_years)

    pf_values = [safe_float(row["profit_factor"]) for row in valid_years if safe_int(row["total_trades"]) >= 5]
    exp_values = [safe_float(row["expectancy"]) for row in valid_years if safe_int(row["total_trades"]) >= 5]
    dd_values = [safe_float(row["max_drawdown"]) for row in valid_years]
    tpm_values = [safe_float(row["trades_per_month"]) for row in valid_years]

    stability_ratio = years_pass_edge / len(valid_years) if valid_years else 0.0

    pass_2x_years = sum(
        1 for row in cost_2x
        if safe_int(row["total_trades"]) >= 5 and safe_float(row["profit_factor"]) > 1.0 and safe_float(row["expectancy"]) > 0
    )
    pass_3x_years = sum(
        1 for row in cost_3x
        if safe_int(row["total_trades"]) >= 5 and safe_float(row["profit_factor"]) > 1.0 and safe_float(row["expectancy"]) > 0
    )

    max_dd = float(max(dd_values)) if dd_values else 0.0
    min_pf = float(min(pf_values)) if pf_values else 0.0

    if years_pass_edge >= 4 and total_trades >= 100 and min_pf > 1.0 and max_dd <= 0.12:
        final_decision = "candidate_walk_forward"
    elif years_pass_edge >= 3 and total_trades >= 100 and max_dd <= 0.20:
        final_decision = "watch_regime_filter"
    elif max_dd > 0.20:
        final_decision = "reject_high_yearly_drawdown"
    elif total_trades < 50:
        final_decision = "watch_low_frequency"
    elif years_pass_edge < 3:
        final_decision = "reject_yearly_edge_inconsistent"
    else:
        final_decision = "reject_yearly_unstable"

    return {
        "symbol": first["symbol"],
        "timeframe": first["timeframe"],
        "exit_policy": first["exit_policy"],
        "setup_combo": first["setup_combo"],
        "window_label": first["window_label"],
        "session_start": first["session_start"],
        "session_end": first["session_end"],
        "years_total": len(valid_years),
        "years_with_trades": years_with_trades,
        "years_pass_edge_1x": years_pass_edge,
        "years_negative_1x": len(negative_years),
        "years_pass_2x": pass_2x_years,
        "years_pass_3x": pass_3x_years,
        "total_trades_1x": int(total_trades),
        "avg_trades_per_month": float(sum(tpm_values) / len(tpm_values)) if tpm_values else 0.0,
        "median_pf_1x": float(pd.Series(pf_values).median()) if pf_values else 0.0,
        "min_pf_1x": min_pf,
        "avg_expectancy_1x": float(sum(exp_values) / len(exp_values)) if exp_values else 0.0,
        "median_expectancy_1x": float(pd.Series(exp_values).median()) if exp_values else 0.0,
        "max_yearly_dd_1x": max_dd,
        "stability_ratio_1x": float(stability_ratio),
        "decision": final_decision,
    }


def run_yearly_diagnostics(ranking_path: Path, data_dir: Path, output_dir: Path, max_candidates: int, decisions: list[str], include_reject_no_edge_top: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(
        ranking_path=ranking_path,
        max_candidates=max_candidates,
        decisions=decisions,
        include_reject_no_edge_top=include_reject_no_edge_top,
    )
    if not candidates:
        raise SystemExit("ERROR: no candidates selected for yearly diagnostics.")

    cache: dict[tuple[str, str], pd.DataFrame] = {}
    detail_rows = []
    summary_rows = []

    for i, candidate in enumerate(candidates, start=1):
        symbol = str(candidate["symbol"]).upper()
        timeframe = str(candidate["timeframe"])
        key = (symbol, timeframe)
        if key not in cache:
            csv_path = find_best_processed_csv(symbol, timeframe, data_dir)
            cache[key] = load_ohlcv(csv_path)

        df = cache[key]
        years = sorted(df["timestamp"].dt.year.dropna().unique().tolist())
        print(
            f"[{i}/{len(candidates)}] {symbol} {timeframe} {candidate['setup_combo']} {candidate['window_label']} years={years}",
            flush=True,
            file=sys.stderr,
        )

        candidate_rows = []
        for year in years:
            for cost_label, cost_per_side in COST_SCENARIOS.items():
                row = run_candidate_year_cost(
                    candidate=candidate,
                    df=df,
                    year=int(year),
                    cost_label=cost_label,
                    cost_per_side=cost_per_side,
                )
                detail_rows.append(row)
                candidate_rows.append(row)
        summary_rows.append(aggregate_candidate(candidate_rows))

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)

    detail_path = output_dir / "yearly_setup_performance_detail.csv"
    summary_path = output_dir / "yearly_setup_performance_summary.csv"
    report_path = output_dir / "yearly_setup_performance_report.json"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    summary_sorted = summary.sort_values(
        ["decision", "stability_ratio_1x", "median_pf_1x", "avg_expectancy_1x", "total_trades_1x", "max_yearly_dd_1x"],
        ascending=[True, False, False, False, False, True],
    )

    report = {
        "ok": True,
        "ranking_path": str(ranking_path),
        "candidates_tested": int(len(candidates)),
        "decision_counts": summary["decision"].value_counts().to_dict(),
        "top_summary": summary_sorted.head(40).to_dict("records"),
        "output_files": {
            "detail": str(detail_path),
            "summary": str(summary_path),
            "report": str(report_path),
        },
        "rules": {
            "candidate_walk_forward": ">=4 positive years, >=100 trades, min yearly PF>1, max yearly DD<=12%",
            "watch_regime_filter": ">=3 positive years, >=100 trades, max yearly DD<=20%",
            "reject_high_yearly_drawdown": "max yearly DD>20%",
            "reject_yearly_edge_inconsistent": "<3 positive edge years",
        },
    }
    report_path.write_text(json.dumps(clean_for_json(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze yearly setup performance for 5Y diagnostics.")
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--decisions", nargs="+", default=DEFAULT_DECISIONS)
    parser.add_argument("--include-reject-no-edge-top", type=int, default=5)
    args = parser.parse_args()

    report = run_yearly_diagnostics(
        ranking_path=Path(args.ranking),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        max_candidates=args.max_candidates,
        decisions=args.decisions,
        include_reject_no_edge_top=args.include_reject_no_edge_top,
    )
    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()
