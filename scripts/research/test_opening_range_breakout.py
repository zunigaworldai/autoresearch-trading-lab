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

from engine.trade_simulator import VALID_EXIT_POLICIES, run_trade_simulation
from strategies.opening_range_breakout import DEFAULT_PARAMS, generate_signals


COST_SCENARIOS = {
    "cost_1x": 0.0005,
    "cost_2x": 0.0010,
    "cost_3x": 0.0015,
}

DEFAULT_OR_MINUTES = [5, 10, 15, 30]
DEFAULT_ENTRY_WINDOWS = [
    ("entry_0940_1030", "09:40", "10:30"),
    ("entry_0945_1030", "09:45", "10:30"),
    ("entry_0945_1100", "09:45", "11:00"),
    ("entry_1000_1130", "10:00", "11:30"),
    ("entry_1030_1200", "10:30", "12:00"),
    ("entry_1100_1300", "11:00", "13:00"),
    ("entry_1400_1600", "14:00", "16:00"),
    ("entry_0945_1600", "09:45", "16:00"),
]

DEFAULT_VARIANTS = [
    {
        "variant": "both_or_stop_rr2",
        "direction": "both",
        "stop_mode": "or_opposite",
        "min_rr": 2.0,
        "require_volume_confirm": False,
        "require_trend_filter": False,
    },
    {
        "variant": "long_or_stop_rr2",
        "direction": "long",
        "stop_mode": "or_opposite",
        "min_rr": 2.0,
        "require_volume_confirm": False,
        "require_trend_filter": False,
    },
    {
        "variant": "short_or_stop_rr2",
        "direction": "short",
        "stop_mode": "or_opposite",
        "min_rr": 2.0,
        "require_volume_confirm": False,
        "require_trend_filter": False,
    },
    {
        "variant": "both_atr_stop_rr2",
        "direction": "both",
        "stop_mode": "atr",
        "atr_stop_mult": 1.0,
        "min_rr": 2.0,
        "require_volume_confirm": False,
        "require_trend_filter": False,
    },
    {
        "variant": "both_or_stop_rr15_vol",
        "direction": "both",
        "stop_mode": "or_opposite",
        "min_rr": 1.5,
        "require_volume_confirm": True,
        "vol_mult": 1.2,
        "require_trend_filter": False,
    },
    {
        "variant": "both_or_stop_rr2_trend",
        "direction": "both",
        "stop_mode": "or_opposite",
        "min_rr": 2.0,
        "require_volume_confirm": False,
        "require_trend_filter": True,
        "trend_len": 50,
    },
]


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


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"ERROR: CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"ERROR: missing columns {sorted(missing)} in {path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["datetime"] = df["timestamp"]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).sort_values("timestamp").reset_index(drop=True)


def profit_factor(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


def compound_return(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return 0.0
    return float((1.0 + returns).prod() - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if returns.empty:
        return 0.0
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(abs(dd.min()))


def trade_dates(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype="datetime64[ns]")
    for col in ["entry_time", "entry_timestamp", "timestamp", "datetime", "date"]:
        if col in trades.columns:
            return pd.to_datetime(trades[col], errors="coerce")
    return pd.Series(dtype="datetime64[ns]")


def period_stats(trades: pd.DataFrame, period: str) -> pd.DataFrame:
    if trades.empty or "date" not in trades.columns:
        return pd.DataFrame(columns=["period", "trades", "pf", "expectancy", "return", "max_drawdown"])

    rows = []
    for key, group in trades.groupby(trades["date"].dt.to_period(period)):
        r = pd.to_numeric(group["net_return"], errors="coerce").dropna()
        rows.append(
            {
                "period": str(key),
                "trades": int(len(r)),
                "pf": profit_factor(r),
                "expectancy": float(r.mean()) if len(r) else 0.0,
                "return": compound_return(r),
                "max_drawdown": max_drawdown(r),
            }
        )
    return pd.DataFrame(rows)


def classify(row: dict[str, Any], min_trades: int) -> str:
    if row["trades"] < min_trades:
        return "ignore_low_sample"
    if row["pf"] <= 1.0 or row["expectancy"] <= 0:
        return "reject_no_edge"
    if row["years_negative"] > 0:
        return "watch_negative_year"
    if row["negative_month_pct_active"] > 0.45:
        return "watch_monthly_unstable"
    if row["worst_month_return"] < -0.06:
        return "watch_worst_month"
    if row["max_drawdown"] > 0.20:
        return "watch_high_drawdown"
    return "candidate_review"


def score(row: dict[str, Any]) -> float:
    s = 0.0
    s += min(max(row["pf"], 0.0), 2.5) / 2.5 * 0.20
    s += min(max(row["expectancy"], 0.0), 0.006) / 0.006 * 0.20
    s += max(0.0, 1.0 - min(row["max_drawdown"], 0.25) / 0.25) * 0.15
    s += max(0.0, 1.0 - min(row["negative_month_pct_active"], 1.0)) * 0.15
    s += max(0.0, 1.0 - min(row["years_negative"], 3) / 3) * 0.10
    s += min(row["trades_per_calendar_month"] / 10.0, 1.0) * 0.10
    s += min(row["active_month_pct"], 1.0) * 0.10
    if row["pf"] <= 1.0 or row["expectancy"] <= 0:
        s -= 0.30
    if row["years_negative"] > 0:
        s -= 0.15
    if row["trades"] < 30:
        s -= 0.15
    return float(s)


def evaluate_trades(
    symbol: str,
    timeframe: str,
    variant: str,
    or_minutes: int,
    entry_window_label: str,
    entry_start: str,
    entry_end: str,
    exit_policy: str,
    cost_label: str,
    cost_per_side: float,
    trades: pd.DataFrame,
    dataset_start: pd.Timestamp,
    dataset_end: pd.Timestamp,
    min_trades: int,
) -> dict[str, Any]:
    out = trades.copy()
    if not out.empty:
        out["date"] = trade_dates(out).dt.normalize()
        out["net_return"] = pd.to_numeric(out["net_return"], errors="coerce")
        out = out.dropna(subset=["date", "net_return"]).copy()

    returns = out["net_return"] if not out.empty and "net_return" in out.columns else pd.Series(dtype=float)

    yearly = period_stats(out, "Y")
    monthly = period_stats(out, "M")
    weekly = period_stats(out, "W")

    total_months = int(len(pd.period_range(dataset_start, dataset_end, freq="M")))
    active_months = int(out["date"].dt.to_period("M").nunique()) if not out.empty else 0
    total_weeks = int(len(pd.period_range(dataset_start, dataset_end, freq="W")))
    active_weeks = int(out["date"].dt.to_period("W").nunique()) if not out.empty else 0

    years_negative = int(len(yearly[yearly["return"] < 0])) if not yearly.empty else 0
    months_negative = int(len(monthly[monthly["return"] < 0])) if not monthly.empty else 0
    weeks_negative = int(len(weekly[weekly["return"] < 0])) if not weekly.empty else 0

    row = {
        "symbol": symbol,
        "timeframe": timeframe,
        "strategy_family": "opening_range_breakout",
        "variant": variant,
        "or_minutes": or_minutes,
        "entry_window_label": entry_window_label,
        "entry_start": entry_start,
        "entry_end": entry_end,
        "exit_policy": exit_policy,
        "cost_label": cost_label,
        "cost_per_side": cost_per_side,
        "trades": int(len(out)),
        "pf": profit_factor(returns),
        "expectancy": float(returns.mean()) if len(returns) else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "total_return": compound_return(returns),
        "max_drawdown": max_drawdown(returns),
        "calendar_months": total_months,
        "active_months": active_months,
        "active_month_pct": float(active_months / total_months) if total_months else 0.0,
        "trades_per_calendar_month": float(len(out) / total_months) if total_months else 0.0,
        "trades_per_active_month": float(len(out) / active_months) if active_months else 0.0,
        "calendar_weeks": total_weeks,
        "active_weeks": active_weeks,
        "trades_per_calendar_week": float(len(out) / total_weeks) if total_weeks else 0.0,
        "years_total": int(len(yearly)),
        "years_negative": years_negative,
        "worst_year_return": float(yearly["return"].min()) if not yearly.empty else 0.0,
        "months_total_active": int(len(monthly)),
        "months_negative": months_negative,
        "negative_month_pct_active": float(months_negative / len(monthly)) if len(monthly) else 0.0,
        "worst_month_return": float(monthly["return"].min()) if not monthly.empty else 0.0,
        "weeks_negative": weeks_negative,
        "negative_week_pct_active": float(weeks_negative / len(weekly)) if len(weekly) else 0.0,
        "worst_week_return": float(weekly["return"].min()) if not weekly.empty else 0.0,
    }
    row["decision"] = classify(row, min_trades)
    row["strategy_score"] = score(row)
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.exit_policy not in VALID_EXIT_POLICIES:
        raise SystemExit(f"ERROR: invalid exit policy {args.exit_policy}")

    df = load_ohlcv(Path(args.csv))
    dataset_start = df["timestamp"].min()
    dataset_end = df["timestamp"].max()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for variant in DEFAULT_VARIANTS:
        for or_minutes in args.or_minutes:
            for window_label, entry_start, entry_end in DEFAULT_ENTRY_WINDOWS:
                params = {
                    **DEFAULT_PARAMS,
                    **variant,
                    "or_minutes": int(or_minutes),
                    "entry_start": entry_start,
                    "entry_end": entry_end,
                }

                signals = generate_signals(df, params)

                for cost_label, cost_per_side in COST_SCENARIOS.items():
                    _, _, trades = run_trade_simulation(
                        signals=signals,
                        cost_per_side=cost_per_side,
                        force_eod_exit=True,
                        exit_policy=args.exit_policy,
                    )
                    rows.append(
                        evaluate_trades(
                            symbol=args.symbol.upper(),
                            timeframe=args.timeframe,
                            variant=variant["variant"],
                            or_minutes=int(or_minutes),
                            entry_window_label=window_label,
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
                    )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["cost_label", "strategy_score", "pf", "expectancy", "trades"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)

    safe = f"{args.symbol.upper()}_{args.timeframe}_{args.exit_policy}_opening_range_breakout"
    csv_path = output_dir / f"{safe}.csv"
    report_path = output_dir / f"{safe}.json"

    result.to_csv(csv_path, index=False)

    cost_1x = result[result["cost_label"] == "cost_1x"].copy()
    report = {
        "ok": True,
        "symbol": args.symbol.upper(),
        "timeframe": args.timeframe,
        "strategy_family": "opening_range_breakout",
        "exit_policy": args.exit_policy,
        "csv": args.csv,
        "or_minutes": args.or_minutes,
        "variants": [v["variant"] for v in DEFAULT_VARIANTS],
        "entry_windows": [
            {"label": x[0], "start": x[1], "end": x[2]}
            for x in DEFAULT_ENTRY_WINDOWS
        ],
        "rows": int(len(result)),
        "decision_counts_cost_1x": cost_1x["decision"].value_counts().to_dict(),
        "top_cost_1x": cost_1x.head(args.top_n).to_dict("records"),
        "output_files": {
            "csv": str(csv_path),
            "report": str(report_path),
        },
        "warning": "Diagnostic only. Must pass OOS, walk-forward, cost stress, and portfolio analysis before paper/live.",
    }

    report_path.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Opening Range Breakout strategy on processed OHLCV data.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--exit-policy", default="partial_50_at_1r_be", choices=sorted(VALID_EXIT_POLICIES))
    parser.add_argument("--or-minutes", nargs="+", type=int, default=DEFAULT_OR_MINUTES)
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--output-dir", default="outputs/research/opening_range_breakout")
    args = parser.parse_args()

    print(json.dumps(clean(run(args)), indent=2))


if __name__ == "__main__":
    main()
