from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRADES_FILE = ROOT / "outputs/research/opening_failed_breakout_reversal_v2_walk_forward/NVDA_5m_orb_trap_v2_walk_forward_trades.csv"
OUT_DIR = ROOT / "outputs/research/opening_failed_breakout_reversal_v2_tail_stress"


COST_SCENARIOS = {
    "cost_3x": 0.0015,
    "cost_4x": 0.0020,
    "cost_5x": 0.0025,
    "cost_6x": 0.0030,
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
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    return float(abs((equity / peak - 1.0).min()))


def summary(df: pd.DataFrame, return_col: str = "stress_net_return") -> dict[str, Any]:
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

    r = pd.to_numeric(df[return_col], errors="coerce").dropna()
    yearly = df.groupby(df["entry_dt"].dt.to_period("Y"))[return_col].apply(compound_return)
    monthly = df.groupby(df["entry_dt"].dt.to_period("M"))[return_col].apply(compound_return)

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


def monte_carlo_trade_omission(
    df: pd.DataFrame,
    return_col: str,
    omission_pcts: list[float],
    n: int = 3000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    returns = pd.to_numeric(df[return_col], errors="coerce").dropna().to_numpy()
    rows = []

    if len(returns) == 0:
        return rows

    for pct in omission_pcts:
        keep_n = max(1, int(round(len(returns) * (1.0 - pct))))
        totals = []
        pfs = []
        exps = []
        dds = []

        for _ in range(n):
            idx = rng.choice(np.arange(len(returns)), size=keep_n, replace=False)
            sample = pd.Series(returns[idx])
            totals.append(compound_return(sample))
            pfs.append(profit_factor(sample))
            exps.append(float(sample.mean()))
            dds.append(max_drawdown(sample))

        rows.append(
            {
                "omission_pct": pct,
                "keep_trades": keep_n,
                "mc_n": n,
                "p05_return": float(np.percentile(totals, 5)),
                "p50_return": float(np.percentile(totals, 50)),
                "p95_return": float(np.percentile(totals, 95)),
                "p05_pf": float(np.percentile(pfs, 5)),
                "p50_pf": float(np.percentile(pfs, 50)),
                "p05_expectancy": float(np.percentile(exps, 5)),
                "p50_expectancy": float(np.percentile(exps, 50)),
                "p95_maxdd": float(np.percentile(dds, 95)),
                "fail_rate_return_le_0": float(np.mean(np.array(totals) <= 0.0)),
                "fail_rate_pf_le_1": float(np.mean(np.array(pfs) <= 1.0)),
            }
        )

    return rows


def classify_slippage(row: dict[str, Any]) -> str:
    if row["pf"] <= 1.0 or row["expectancy"] <= 0:
        return "fail_slippage_stress"
    if row["years_negative"] > 1:
        return "watch_yearly_unstable"
    if row["cost_label"] in {"cost_5x", "cost_6x"} and row["pf"] > 1.0:
        return "pass_extreme_cost_watch_low_count"
    return "pass_cost_stress_low_count"


def classify_omission(row: dict[str, Any]) -> str:
    if row["fail_rate_return_le_0"] > 0.20 or row["fail_rate_pf_le_1"] > 0.20:
        return "fail_trade_omission_mc"
    if row["p05_return"] <= 0 or row["p05_pf"] <= 1.0 or row["p05_expectancy"] <= 0:
        return "watch_trade_omission_tail_risk"
    return "pass_trade_omission_mc"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRADES_FILE)
    df = df[
        (df["variant"] == "robust_entry_core") &
        (df["cost_label"] == "cost_3x")
    ].copy()

    df["entry_dt"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["gross_return"] = pd.to_numeric(df["gross_return"], errors="coerce")
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
    df = df.dropna(subset=["entry_dt", "gross_return", "net_return"]).sort_values("entry_dt").reset_index(drop=True)
    df["trade_id"] = np.arange(1, len(df) + 1)

    slippage_rows = []
    all_stress_trades = []

    for cost_label, cost_per_side in COST_SCENARIOS.items():
        stress = df.copy()
        stress["stress_cost_label"] = cost_label
        stress["stress_cost_per_side"] = cost_per_side
        stress["stress_roundtrip_cost"] = 2.0 * cost_per_side
        stress["stress_net_return"] = stress["gross_return"] - stress["stress_roundtrip_cost"]

        row = {
            "symbol": "NVDA",
            "timeframe": "5m",
            "variant": "robust_entry_core",
            "cost_label": cost_label,
            "cost_per_side": cost_per_side,
            "roundtrip_cost": 2.0 * cost_per_side,
        }
        row.update(summary(stress, "stress_net_return"))
        row["decision"] = classify_slippage(row)
        slippage_rows.append(row)
        all_stress_trades.append(stress)

    slippage_df = pd.DataFrame(slippage_rows)
    stress_trades_df = pd.concat(all_stress_trades, ignore_index=True)

    omission_rows = []
    for cost_label in ["cost_3x", "cost_4x", "cost_5x", "cost_6x"]:
        stress = stress_trades_df[stress_trades_df["stress_cost_label"] == cost_label].copy()
        rows = monte_carlo_trade_omission(
            stress,
            return_col="stress_net_return",
            omission_pcts=[0.10, 0.20, 0.30, 0.40],
            n=3000,
            seed=42,
        )
        for row in rows:
            row["cost_label"] = cost_label
            row["decision"] = classify_omission(row)
            omission_rows.append(row)

    omission_df = pd.DataFrame(omission_rows)

    slippage_path = OUT_DIR / "NVDA_5m_orb_trap_v2_slippage_beyond_cost3.csv"
    omission_path = OUT_DIR / "NVDA_5m_orb_trap_v2_trade_omission_mc.csv"
    stress_trades_path = OUT_DIR / "NVDA_5m_orb_trap_v2_tail_stress_trades.csv"
    report_path = OUT_DIR / "NVDA_5m_orb_trap_v2_tail_stress.json"

    slippage_df.to_csv(slippage_path, index=False)
    omission_df.to_csv(omission_path, index=False)
    stress_trades_df.to_csv(stress_trades_path, index=False)

    report = {
        "ok": True,
        "strategy_family": "opening_failed_breakout_reversal_v2_tail_stress",
        "symbol": "NVDA",
        "timeframe": "5m",
        "variant": "robust_entry_core",
        "warning": "Tail stress diagnostic only. No paper/live without explicit approval.",
        "slippage_beyond_cost3": slippage_df.to_dict("records"),
        "trade_omission_mc": omission_df.to_dict("records"),
        "slippage_decision_counts": slippage_df["decision"].value_counts().to_dict(),
        "omission_decision_counts": omission_df["decision"].value_counts().to_dict(),
        "output_files": {
            "slippage": str(slippage_path),
            "trade_omission_mc": str(omission_path),
            "stress_trades": str(stress_trades_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    print(json.dumps(clean(report), indent=2))


if __name__ == "__main__":
    main()
