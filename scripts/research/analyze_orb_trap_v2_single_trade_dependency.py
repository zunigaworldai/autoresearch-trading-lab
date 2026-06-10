from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRADES_FILE = ROOT / "outputs/research/opening_failed_breakout_reversal_v2_walk_forward/NVDA_5m_orb_trap_v2_walk_forward_trades.csv"
OUT_DIR = ROOT / "outputs/research/opening_failed_breakout_reversal_v2_single_trade_dependency"


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


def profit_factor(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    gains = r[r > 0].sum()
    losses = r[r < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


def compound_return(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return 0.0
    return float((1.0 + r).prod() - 1.0)


def max_drawdown(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if r.empty:
        return 0.0
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    return float(abs((eq / peak - 1.0).min()))


def summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "trades": 0,
            "pf": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "years_negative": 0,
            "months_negative": 0,
            "months_total_active": 0,
        }

    r = pd.to_numeric(df["net_return"], errors="coerce").dropna()
    yearly = df.groupby(df["entry_dt"].dt.to_period("Y"))["net_return"].apply(compound_return)
    monthly = df.groupby(df["entry_dt"].dt.to_period("M"))["net_return"].apply(compound_return)

    return {
        "trades": int(len(r)),
        "pf": profit_factor(r),
        "expectancy": float(r.mean()) if len(r) else 0.0,
        "win_rate": float((r > 0).mean()) if len(r) else 0.0,
        "total_return": compound_return(r),
        "max_drawdown": max_drawdown(r),
        "years_negative": int((yearly < 0).sum()) if len(yearly) else 0,
        "months_negative": int((monthly < 0).sum()) if len(monthly) else 0,
        "months_total_active": int(len(monthly)),
    }


def classify(report: dict[str, Any]) -> str:
    base = report["baseline"]
    loo = report["leave_one_out"]
    remove_best = report["remove_best_n"]

    if base["trades"] < 30:
        base_flag = "low_trade_count"
    else:
        base_flag = "enough_trades"

    if loo["min_pf_after_removing_one"] <= 1.0 or loo["min_expectancy_after_removing_one"] <= 0:
        return f"fail_single_trade_dependency_{base_flag}"

    if remove_best["remove_top_1"]["pf"] <= 1.0 or remove_best["remove_top_1"]["expectancy"] <= 0:
        return f"fail_top_trade_dependency_{base_flag}"

    if remove_best["remove_top_2"]["pf"] <= 1.0 or remove_best["remove_top_2"]["expectancy"] <= 0:
        return f"watch_top_2_trade_dependency_{base_flag}"

    if loo["max_pf_drop_pct"] > 0.50:
        return f"watch_large_pf_drop_{base_flag}"

    return f"pass_single_trade_dependency_{base_flag}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRADES_FILE)
    df = df[
        (df["cost_label"] == "cost_3x") &
        (df["variant"] == "robust_entry_core")
    ].copy()

    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
    df = df.dropna(subset=["entry_dt", "net_return"]).sort_values("entry_dt").reset_index(drop=True)
    df["trade_id"] = np.arange(1, len(df) + 1)

    baseline = summary(df)
    base_pf = baseline["pf"]
    base_exp = baseline["expectancy"]
    base_total = baseline["total_return"]

    loo_rows = []
    for _, trade in df.iterrows():
        reduced = df[df["trade_id"] != trade["trade_id"]].copy()
        s = summary(reduced)
        loo_rows.append({
            "removed_trade_id": int(trade["trade_id"]),
            "removed_entry_time": str(trade["entry_dt"]),
            "removed_side": trade.get("side", ""),
            "removed_exit_reason": trade.get("exit_reason", ""),
            "removed_net_return": float(trade["net_return"]),
            "trades_after": s["trades"],
            "pf_after": s["pf"],
            "expectancy_after": s["expectancy"],
            "total_return_after": s["total_return"],
            "max_drawdown_after": s["max_drawdown"],
            "pf_drop": float(base_pf - s["pf"]),
            "pf_drop_pct": float((base_pf - s["pf"]) / base_pf) if base_pf else 0.0,
            "expectancy_drop": float(base_exp - s["expectancy"]),
            "total_return_drop": float(base_total - s["total_return"]),
        })

    loo_df = pd.DataFrame(loo_rows).sort_values(["pf_after", "expectancy_after"], ascending=[True, True])

    best_sorted = df.sort_values("net_return", ascending=False).copy()
    worst_sorted = df.sort_values("net_return", ascending=True).copy()

    remove_best = {}
    for n in [1, 2, 3, 5]:
        remove_ids = set(best_sorted.head(n)["trade_id"])
        reduced = df[~df["trade_id"].isin(remove_ids)].copy()
        remove_best[f"remove_top_{n}"] = summary(reduced)

    remove_worst = {}
    for n in [1, 2, 3, 5]:
        remove_ids = set(worst_sorted.head(n)["trade_id"])
        reduced = df[~df["trade_id"].isin(remove_ids)].copy()
        remove_worst[f"remove_worst_{n}"] = summary(reduced)

    contribution = df.copy()
    contribution["abs_return"] = contribution["net_return"].abs()
    contribution["positive_return"] = contribution["net_return"].clip(lower=0.0)
    total_positive = contribution["positive_return"].sum()
    contribution["pct_of_positive_pnl"] = np.where(
        total_positive != 0,
        contribution["positive_return"] / total_positive,
        0.0,
    )
    contribution = contribution.sort_values("net_return", ascending=False)

    loo_summary = {
        "min_pf_after_removing_one": float(loo_df["pf_after"].min()) if not loo_df.empty else 0.0,
        "min_expectancy_after_removing_one": float(loo_df["expectancy_after"].min()) if not loo_df.empty else 0.0,
        "min_total_return_after_removing_one": float(loo_df["total_return_after"].min()) if not loo_df.empty else 0.0,
        "max_pf_drop": float(loo_df["pf_drop"].max()) if not loo_df.empty else 0.0,
        "max_pf_drop_pct": float(loo_df["pf_drop_pct"].max()) if not loo_df.empty else 0.0,
        "worst_removed_trade": loo_df.head(1).to_dict("records")[0] if not loo_df.empty else {},
    }

    report = {
        "ok": True,
        "strategy_family": "opening_failed_breakout_reversal_v2_single_trade_dependency",
        "symbol": "NVDA",
        "timeframe": "5m",
        "variant": "robust_entry_core",
        "cost_label": "cost_3x",
        "warning": "Dependency diagnostic only. No paper/live without explicit approval.",
        "baseline": baseline,
        "leave_one_out": loo_summary,
        "remove_best_n": remove_best,
        "remove_worst_n": remove_worst,
        "top_positive_contributors": contribution.head(10)[[
            "trade_id", "entry_time", "exit_time", "side", "exit_reason",
            "net_return", "pct_of_positive_pnl"
        ]].to_dict("records"),
    }
    report["decision"] = classify(report)

    trades_out = OUT_DIR / "NVDA_5m_orb_trap_v2_single_trade_dependency_trades.csv"
    loo_out = OUT_DIR / "NVDA_5m_orb_trap_v2_leave_one_out.csv"
    contribution_out = OUT_DIR / "NVDA_5m_orb_trap_v2_trade_contribution.csv"
    report_out = OUT_DIR / "NVDA_5m_orb_trap_v2_single_trade_dependency.json"

    df.to_csv(trades_out, index=False)
    loo_df.to_csv(loo_out, index=False)
    contribution.to_csv(contribution_out, index=False)
    report_out.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")

    print(json.dumps(clean(report), indent=2))


if __name__ == "__main__":
    main()
