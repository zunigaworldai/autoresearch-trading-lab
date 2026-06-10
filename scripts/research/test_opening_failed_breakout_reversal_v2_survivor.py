from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.trade_simulator import VALID_EXIT_POLICIES, run_trade_simulation
from strategies.opening_failed_breakout_reversal import DEFAULT_PARAMS, generate_signals


COST_SCENARIOS = {
    "cost_1x": 0.0005,
    "cost_2x": 0.0010,
    "cost_3x": 0.0015,
}


V2_VARIANTS = {
    # Control: same survivor setup already found.
    "v2_base_survivor": {
        "filter_kind": "none",
        "calendar_diagnostic": False,
    },

    # Defensible pre-trade filters from survivor diagnostics.
    "v2_gap_abs_gt_1pct": {
        "filter_kind": "gap_abs_gt_1pct",
        "calendar_diagnostic": False,
    },
    "v2_entry_1310_1325_or_1420_1440": {
        "filter_kind": "entry_combo",
        "calendar_diagnostic": False,
    },
    "v2_ema_fast_above_slow": {
        "filter_kind": "ema_fast_above_slow",
        "calendar_diagnostic": False,
    },
    "v2_gap_abs_gt_1pct_and_entry_combo": {
        "filter_kind": "gap_entry_combo",
        "calendar_diagnostic": False,
    },
    "v2_gap_abs_gt_1pct_and_ema": {
        "filter_kind": "gap_ema",
        "calendar_diagnostic": False,
    },
    "v2_entry_combo_and_ema": {
        "filter_kind": "entry_combo_ema",
        "calendar_diagnostic": False,
    },
    "v2_gap_entry_combo_and_ema": {
        "filter_kind": "gap_entry_combo_ema",
        "calendar_diagnostic": False,
    },

    # Calendar diagnostics only. Do not promote without OOS/walk-forward.
    "v2_months_1_3_5_7_11_DIAGNOSTIC_ONLY": {
        "filter_kind": "months_1_3_5_7_11",
        "calendar_diagnostic": True,
    },
    "v2_gap_and_months_DIAGNOSTIC_ONLY": {
        "filter_kind": "gap_months",
        "calendar_diagnostic": True,
    },
    "v2_entry_combo_and_months_DIAGNOSTIC_ONLY": {
        "filter_kind": "entry_combo_months",
        "calendar_diagnostic": True,
    },
}


SURVIVOR_BASE_PARAMS = {
    "variant_name": "both_or15_fail_rr15_extreme",
    "window_label": "afternoon_1300_1500",
    "entry_start": "13:00",
    "entry_end": "15:00",
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


def load_runner():
    runner_path = ROOT / "scripts/research/test_opening_failed_breakout_reversal.py"
    spec = importlib.util.spec_from_file_location("ofbr_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner: {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


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
    return float(abs((equity / peak - 1.0).min()))


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
    if row["calendar_diagnostic"]:
        if row["trades"] < min_trades:
            return "diagnostic_low_sample_calendar_filter"
        return "diagnostic_calendar_filter_only"
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
    s += min(max(row["pf"], 0.0), 3.0) / 3.0 * 0.20
    s += min(max(row["expectancy"], 0.0), 0.010) / 0.010 * 0.20
    s += max(0.0, 1.0 - min(row["max_drawdown"], 0.25) / 0.25) * 0.15
    s += max(0.0, 1.0 - min(row["negative_month_pct_active"], 1.0)) * 0.15
    s += max(0.0, 1.0 - min(row["years_negative"], 3) / 3) * 0.10
    s += min(row["trades_per_calendar_month"] / 5.0, 1.0) * 0.10
    s += min(row["active_month_pct"], 1.0) * 0.10
    if row["pf"] <= 1.0 or row["expectancy"] <= 0:
        s -= 0.30
    if row["years_negative"] > 0:
        s -= 0.15
    if row["trades"] < 20:
        s -= 0.15
    if row["calendar_diagnostic"]:
        s -= 0.25
    return float(s)


def add_pretrade_features(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    out["dt"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["month"] = out["dt"].dt.month
    out["entry_hhmm"] = out["dt"].dt.strftime("%H:%M")
    out["gap_abs"] = out["gap_pct"].abs()
    out["ema_fast_above_slow"] = out["ema_fast"] > out["ema_slow"]
    out["close_above_vwap"] = out["close"] > out["vwap"]
    out["volume_ratio"] = out["volume"] / out["vol_sma"].replace(0, np.nan)
    out["or_range_atr"] = out["or_range"] / out["atr"].replace(0, np.nan)
    return out


def build_filter(signals: pd.DataFrame, filter_kind: str) -> pd.Series:
    dt = pd.to_datetime(signals["datetime"], errors="coerce")
    hhmm = dt.dt.strftime("%H:%M")

    gap = signals["gap_abs"] > 0.01
    entry_combo = hhmm.between("13:10", "13:25") | hhmm.between("14:20", "14:40")
    ema = signals["ema_fast_above_slow"]
    months = dt.dt.month.isin([1, 3, 5, 7, 11])

    if filter_kind == "none":
        return pd.Series(True, index=signals.index)
    if filter_kind == "gap_abs_gt_1pct":
        return gap
    if filter_kind == "entry_combo":
        return entry_combo
    if filter_kind == "ema_fast_above_slow":
        return ema
    if filter_kind == "gap_entry_combo":
        return gap & entry_combo
    if filter_kind == "gap_ema":
        return gap & ema
    if filter_kind == "entry_combo_ema":
        return entry_combo & ema
    if filter_kind == "gap_entry_combo_ema":
        return gap & entry_combo & ema
    if filter_kind == "months_1_3_5_7_11":
        return months
    if filter_kind == "gap_months":
        return gap & months
    if filter_kind == "entry_combo_months":
        return entry_combo & months

    raise ValueError(f"Unknown filter_kind: {filter_kind}")


def apply_signal_filter(signals: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    out = signals.copy()
    mask = mask.fillna(False)
    signal_cols = [
        "entry_signal",
        "long_signal",
        "short_signal",
        "long_signal_raw",
        "short_signal_raw",
    ]
    for col in signal_cols:
        if col in out.columns:
            if col == "entry_signal":
                out.loc[~mask, col] = 0
            else:
                out.loc[~mask, col] = False
    out.loc[~mask, "exit_stop"] = np.nan
    out.loc[~mask, "exit_tp"] = np.nan
    if "setup" in out.columns:
        out.loc[~mask, "setup"] = ""
    return out


def evaluate(
    symbol: str,
    timeframe: str,
    variant_label: str,
    filter_kind: str,
    calendar_diagnostic: bool,
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
        "strategy_family": "opening_failed_breakout_reversal_v2_survivor",
        "base_strategy_family": "opening_failed_breakout_reversal",
        "base_variant": SURVIVOR_BASE_PARAMS["variant_name"],
        "base_window_label": SURVIVOR_BASE_PARAMS["window_label"],
        "variant": variant_label,
        "filter_kind": filter_kind,
        "calendar_diagnostic": bool(calendar_diagnostic),
        "exit_policy": "partial_50_at_1r_be",
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
    row["decision"] = classify(row, min_trades=min_trades)
    row["strategy_score"] = score(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="ORB Trap survivor v2 diagnostic runner.")
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument(
        "--csv",
        default="data/processed/NVDA/NVDA_5m_RTH_2021-06-01_2026-05-29.csv",
    )
    parser.add_argument("--exit-policy", default="partial_50_at_1r_be", choices=sorted(VALID_EXIT_POLICIES))
    parser.add_argument("--min-trades", type=int, default=15)
    parser.add_argument("--output-dir", default="outputs/research/opening_failed_breakout_reversal_v2_survivor")
    args = parser.parse_args()

    if args.exit_policy != "partial_50_at_1r_be":
        raise SystemExit("This v2 survivor diagnostic is currently locked to partial_50_at_1r_be.")

    runner = load_runner()
    variant = runner.VARIANTS[SURVIVOR_BASE_PARAMS["variant_name"]]
    entry_start, entry_end = runner.ENTRY_WINDOWS[SURVIVOR_BASE_PARAMS["window_label"]]

    params = {
        **DEFAULT_PARAMS,
        **variant,
        "entry_start": entry_start,
        "entry_end": entry_end,
    }

    df = runner.load_ohlcv(ROOT / args.csv)
    dataset_start = df["timestamp"].min()
    dataset_end = df["timestamp"].max()

    base_signals = generate_signals(df, params)
    base_signals = add_pretrade_features(base_signals)

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_trades = []

    for variant_label, spec in V2_VARIANTS.items():
        mask = build_filter(base_signals, spec["filter_kind"])
        filtered_signals = apply_signal_filter(base_signals, mask)

        for cost_label, cost_per_side in COST_SCENARIOS.items():
            _, _, trades = run_trade_simulation(
                signals=filtered_signals,
                cost_per_side=cost_per_side,
                force_eod_exit=True,
                exit_policy=args.exit_policy,
            )

            trades = trades.copy()
            trades["symbol"] = args.symbol.upper()
            trades["timeframe"] = args.timeframe
            trades["strategy_family"] = "opening_failed_breakout_reversal_v2_survivor"
            trades["base_variant"] = SURVIVOR_BASE_PARAMS["variant_name"]
            trades["base_window_label"] = SURVIVOR_BASE_PARAMS["window_label"]
            trades["variant"] = variant_label
            trades["filter_kind"] = spec["filter_kind"]
            trades["calendar_diagnostic"] = bool(spec["calendar_diagnostic"])
            trades["cost_label"] = cost_label
            trades["cost_per_side"] = cost_per_side
            trades["exit_policy"] = args.exit_policy
            all_trades.append(trades)

            rows.append(
                evaluate(
                    symbol=args.symbol.upper(),
                    timeframe=args.timeframe,
                    variant_label=variant_label,
                    filter_kind=spec["filter_kind"],
                    calendar_diagnostic=bool(spec["calendar_diagnostic"]),
                    cost_label=cost_label,
                    cost_per_side=cost_per_side,
                    trades=trades,
                    dataset_start=dataset_start,
                    dataset_end=dataset_end,
                    min_trades=args.min_trades,
                )
            )

    result = pd.DataFrame(rows).sort_values(
        ["calendar_diagnostic", "cost_label", "strategy_score", "pf", "expectancy"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    safe = f"{args.symbol.upper()}_{args.timeframe}_{args.exit_policy}_opening_failed_breakout_reversal_v2_survivor"
    csv_path = output_dir / f"{safe}.csv"
    trades_path = output_dir / f"{safe}_trades.csv"
    report_path = output_dir / f"{safe}.json"

    result.to_csv(csv_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    cost3 = result[result["cost_label"] == "cost_3x"].copy()
    report = {
        "ok": True,
        "symbol": args.symbol.upper(),
        "timeframe": args.timeframe,
        "strategy_family": "opening_failed_breakout_reversal_v2_survivor",
        "base_strategy_family": "opening_failed_breakout_reversal",
        "base_variant": SURVIVOR_BASE_PARAMS["variant_name"],
        "base_window_label": SURVIVOR_BASE_PARAMS["window_label"],
        "exit_policy": args.exit_policy,
        "warning": "Diagnostic only. Calendar filters are marked diagnostic only and must not be promoted without OOS/walk-forward.",
        "rows": int(len(result)),
        "decision_counts_cost_3x": cost3["decision"].value_counts().to_dict(),
        "top_cost_3x": cost3.head(30).to_dict("records"),
        "output_files": {
            "csv": str(csv_path),
            "trades": str(trades_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")

    print(json.dumps(clean(report), indent=2))


if __name__ == "__main__":
    main()
