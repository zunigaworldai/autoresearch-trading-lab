from __future__ import annotations

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

from engine.trade_simulator import run_trade_simulation
from strategies.opening_failed_breakout_reversal import DEFAULT_PARAMS, generate_signals


COST_SCENARIOS = {
    "cost_1x": 0.0005,
    "cost_2x": 0.0010,
    "cost_3x": 0.0015,
}

BASE_VARIANT_NAME = "both_or15_fail_rr15_extreme"
BASE_WINDOW_LABEL = "afternoon_1300_1500"
EXIT_POLICY = "partial_50_at_1r_be"


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


def summarize_trades(trades: pd.DataFrame, label_prefix: str = "") -> dict[str, Any]:
    out = trades.copy()
    if out.empty:
        return {
            f"{label_prefix}trades": 0,
            f"{label_prefix}pf": 0.0,
            f"{label_prefix}expectancy": 0.0,
            f"{label_prefix}win_rate": 0.0,
            f"{label_prefix}total_return": 0.0,
            f"{label_prefix}max_drawdown": 0.0,
            f"{label_prefix}years_total": 0,
            f"{label_prefix}years_negative": 0,
            f"{label_prefix}months_total_active": 0,
            f"{label_prefix}months_negative": 0,
            f"{label_prefix}worst_month_return": 0.0,
        }

    out["date"] = trade_dates(out).dt.normalize()
    out["net_return"] = pd.to_numeric(out["net_return"], errors="coerce")
    out = out.dropna(subset=["date", "net_return"]).copy()
    returns = out["net_return"]

    yearly_returns = out.groupby(out["date"].dt.to_period("Y"))["net_return"].apply(compound_return)
    monthly_returns = out.groupby(out["date"].dt.to_period("M"))["net_return"].apply(compound_return)

    return {
        f"{label_prefix}trades": int(len(out)),
        f"{label_prefix}pf": profit_factor(returns),
        f"{label_prefix}expectancy": float(returns.mean()) if len(returns) else 0.0,
        f"{label_prefix}win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        f"{label_prefix}total_return": compound_return(returns),
        f"{label_prefix}max_drawdown": max_drawdown(returns),
        f"{label_prefix}years_total": int(len(yearly_returns)),
        f"{label_prefix}years_negative": int((yearly_returns < 0).sum()),
        f"{label_prefix}months_total_active": int(len(monthly_returns)),
        f"{label_prefix}months_negative": int((monthly_returns < 0).sum()),
        f"{label_prefix}worst_month_return": float(monthly_returns.min()) if len(monthly_returns) else 0.0,
    }


def score(row: dict[str, Any]) -> float:
    s = 0.0
    s += min(max(row["pf"], 0.0), 3.0) / 3.0 * 0.20
    s += min(max(row["expectancy"], 0.0), 0.010) / 0.010 * 0.20
    s += max(0.0, 1.0 - min(row["max_drawdown"], 0.20) / 0.20) * 0.15
    s += max(0.0, 1.0 - min(row["years_negative"], 3) / 3) * 0.15
    s += min(row["trades"] / 25.0, 1.0) * 0.10
    s += min(row["oos_2024_trades"] / 8.0, 1.0) * 0.10
    s += min(row["oos_2025_trades"] / 5.0, 1.0) * 0.10

    if row["pf"] <= 1.0 or row["expectancy"] <= 0:
        s -= 0.30
    if row["years_negative"] > 1:
        s -= 0.20
    if row["oos_2024_pf"] <= 1.0 or row["oos_2024_expectancy"] <= 0:
        s -= 0.20
    if row["oos_2025_pf"] <= 1.0 or row["oos_2025_expectancy"] <= 0:
        s -= 0.20
    if row["trades"] < 15:
        s -= 0.20
    return float(s)


def classify(row: dict[str, Any]) -> str:
    if row["trades"] < 15:
        return "reject_low_sample"
    if row["pf"] <= 1.0 or row["expectancy"] <= 0:
        return "reject_no_edge"
    if row["years_negative"] > 1:
        return "watch_yearly_unstable"
    if row["oos_2024_trades"] < 5 or row["oos_2025_trades"] < 3:
        return "watch_oos_tiny_sample"
    if row["oos_2024_pf"] <= 1.0 or row["oos_2024_expectancy"] <= 0:
        return "reject_oos_2024_failed"
    if row["oos_2025_pf"] <= 1.0 or row["oos_2025_expectancy"] <= 0:
        return "reject_oos_2025_failed"
    if row["max_drawdown"] > 0.08:
        return "watch_drawdown"
    return "candidate_review_needs_walk_forward"


def add_pretrade_features(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    dt = pd.to_datetime(out["datetime"], errors="coerce")
    out["month"] = dt.dt.month
    out["entry_hhmm"] = dt.dt.strftime("%H:%M")
    out["gap_abs"] = out["gap_pct"].abs()
    out["ema_fast_above_slow"] = out["ema_fast"] > out["ema_slow"]
    out["close_above_vwap"] = out["close"] > out["vwap"]
    out["volume_ratio"] = out["volume"] / out["vol_sma"].replace(0, np.nan)
    out["or_range_atr"] = out["or_range"] / out["atr"].replace(0, np.nan)
    return out


def between_hhmm(signals: pd.DataFrame, start: str, end: str) -> pd.Series:
    hhmm = pd.to_datetime(signals["datetime"], errors="coerce").dt.strftime("%H:%M")
    return hhmm.between(start, end)


def make_variant_masks(signals: pd.DataFrame) -> dict[str, pd.Series]:
    dt = pd.to_datetime(signals["datetime"], errors="coerce")
    hhmm = dt.dt.strftime("%H:%M")

    entry_core = hhmm.between("13:10", "13:25") | hhmm.between("14:20", "14:40")
    entry_wide = hhmm.between("13:05", "13:30") | hhmm.between("14:15", "14:45")
    entry_shift = hhmm.between("13:15", "13:30") | hhmm.between("14:25", "14:40")
    entry_early = hhmm.between("13:10", "13:25")
    entry_late = hhmm.between("14:20", "14:40")
    gap_0075 = signals["gap_abs"] > 0.0075
    gap_0100 = signals["gap_abs"] > 0.0100
    gap_0125 = signals["gap_abs"] > 0.0125
    gap_0150 = signals["gap_abs"] > 0.0150
    ema = signals["ema_fast_above_slow"]
    above_vwap = signals["close_above_vwap"]

    return {
        "robust_base_survivor": pd.Series(True, index=signals.index),
        "robust_entry_core": entry_core,
        "robust_entry_wide": entry_wide,
        "robust_entry_shift": entry_shift,
        "robust_entry_early_only": entry_early,
        "robust_entry_late_only": entry_late,
        "robust_gap_0075": gap_0075,
        "robust_gap_0100": gap_0100,
        "robust_gap_0125": gap_0125,
        "robust_gap_0150": gap_0150,
        "robust_ema": ema,
        "robust_above_vwap": above_vwap,
        "robust_entry_core_gap_0075": entry_core & gap_0075,
        "robust_entry_core_gap_0100": entry_core & gap_0100,
        "robust_entry_core_gap_0125": entry_core & gap_0125,
        "robust_entry_core_gap_0150": entry_core & gap_0150,
        "robust_entry_wide_gap_0075": entry_wide & gap_0075,
        "robust_entry_wide_gap_0100": entry_wide & gap_0100,
        "robust_entry_wide_gap_0125": entry_wide & gap_0125,
        "robust_entry_core_ema": entry_core & ema,
        "robust_gap_0100_ema": gap_0100 & ema,
        "robust_entry_core_gap_0100_ema": entry_core & gap_0100 & ema,
        "robust_entry_core_above_vwap": entry_core & above_vwap,
        "robust_gap_0100_above_vwap": gap_0100 & above_vwap,
    }


def apply_signal_filter(signals: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    out = signals.copy()
    mask = mask.fillna(False)

    if "entry_signal" in out.columns:
        out.loc[~mask, "entry_signal"] = 0
    for col in ["long_signal", "short_signal", "long_signal_raw", "short_signal_raw"]:
        if col in out.columns:
            out.loc[~mask, col] = False

    out.loc[~mask, "exit_stop"] = np.nan
    out.loc[~mask, "exit_tp"] = np.nan
    if "setup" in out.columns:
        out.loc[~mask, "setup"] = ""
    return out


def monte_carlo_summary(returns: pd.Series, n: int = 1000, seed: int = 42) -> dict[str, Any]:
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy()
    if len(r) == 0:
        return {
            "mc_n": n,
            "mc_median_return": 0.0,
            "mc_p05_return": 0.0,
            "mc_p95_return": 0.0,
            "mc_median_maxdd": 0.0,
            "mc_p95_maxdd": 0.0,
        }

    rng = np.random.default_rng(seed)
    totals = []
    dds = []
    for _ in range(n):
        sample = rng.choice(r, size=len(r), replace=True)
        s = pd.Series(sample)
        totals.append(compound_return(s))
        dds.append(max_drawdown(s))

    return {
        "mc_n": n,
        "mc_median_return": float(np.percentile(totals, 50)),
        "mc_p05_return": float(np.percentile(totals, 5)),
        "mc_p95_return": float(np.percentile(totals, 95)),
        "mc_median_maxdd": float(np.percentile(dds, 50)),
        "mc_p95_maxdd": float(np.percentile(dds, 95)),
    }


def main() -> None:
    runner = load_runner()

    csv_path = ROOT / "data/processed/NVDA/NVDA_5m_RTH_2021-06-01_2026-05-29.csv"
    ohlcv = runner.load_ohlcv(csv_path)

    base_variant = runner.VARIANTS[BASE_VARIANT_NAME]
    entry_start, entry_end = runner.ENTRY_WINDOWS[BASE_WINDOW_LABEL]
    params = {**DEFAULT_PARAMS, **base_variant, "entry_start": entry_start, "entry_end": entry_end}

    base_signals = generate_signals(ohlcv, params)
    base_signals = add_pretrade_features(base_signals)

    masks = make_variant_masks(base_signals)

    out_dir = ROOT / "outputs/research/opening_failed_breakout_reversal_v2_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_trades = []

    for variant_name, mask in masks.items():
        filtered = apply_signal_filter(base_signals, mask)

        for cost_label, cost_per_side in COST_SCENARIOS.items():
            _, _, trades = run_trade_simulation(
                signals=filtered,
                cost_per_side=cost_per_side,
                force_eod_exit=True,
                exit_policy=EXIT_POLICY,
            )

            trades = trades.copy()
            trades["symbol"] = "NVDA"
            trades["timeframe"] = "5m"
            trades["strategy_family"] = "opening_failed_breakout_reversal_v2_robustness"
            trades["base_variant"] = BASE_VARIANT_NAME
            trades["base_window_label"] = BASE_WINDOW_LABEL
            trades["variant"] = variant_name
            trades["cost_label"] = cost_label
            trades["cost_per_side"] = cost_per_side
            trades["exit_policy"] = EXIT_POLICY
            all_trades.append(trades)

            trades["entry_dt"] = trade_dates(trades)
            trades["net_return"] = pd.to_numeric(trades["net_return"], errors="coerce")
            trades = trades.dropna(subset=["entry_dt", "net_return"]).copy()

            oos_2024 = trades[trades["entry_dt"] >= pd.Timestamp("2024-01-01")].copy()
            oos_2025 = trades[trades["entry_dt"] >= pd.Timestamp("2025-01-01")].copy()

            row = {
                "symbol": "NVDA",
                "timeframe": "5m",
                "strategy_family": "opening_failed_breakout_reversal_v2_robustness",
                "base_variant": BASE_VARIANT_NAME,
                "base_window_label": BASE_WINDOW_LABEL,
                "variant": variant_name,
                "cost_label": cost_label,
                "cost_per_side": cost_per_side,
            }
            row.update(summarize_trades(trades, ""))
            row.update(summarize_trades(oos_2024, "oos_2024_"))
            row.update(summarize_trades(oos_2025, "oos_2025_"))

            if cost_label == "cost_3x":
                row.update(monte_carlo_summary(trades["net_return"], n=1000, seed=42))
            else:
                row.update({
                    "mc_n": 0,
                    "mc_median_return": 0.0,
                    "mc_p05_return": 0.0,
                    "mc_p95_return": 0.0,
                    "mc_median_maxdd": 0.0,
                    "mc_p95_maxdd": 0.0,
                })

            row["decision"] = classify(row)
            row["strategy_score"] = score(row)
            rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["cost_label", "strategy_score", "pf", "expectancy", "trades"],
        ascending=[True, False, False, False, False],
    )
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    summary_path = out_dir / "NVDA_5m_orb_trap_v2_robustness.csv"
    trades_path = out_dir / "NVDA_5m_orb_trap_v2_robustness_trades.csv"
    report_path = out_dir / "NVDA_5m_orb_trap_v2_robustness.json"

    result.to_csv(summary_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    cost3 = result[result["cost_label"] == "cost_3x"].copy()
    report = {
        "ok": True,
        "strategy_family": "opening_failed_breakout_reversal_v2_robustness",
        "symbol": "NVDA",
        "timeframe": "5m",
        "base_variant": BASE_VARIANT_NAME,
        "base_window_label": BASE_WINDOW_LABEL,
        "exit_policy": EXIT_POLICY,
        "warning": "Robustness diagnostic only. No paper/live until walk-forward/OOS review is explicitly approved.",
        "rows": int(len(result)),
        "decision_counts_cost_3x": cost3["decision"].value_counts().to_dict(),
        "top_cost_3x": cost3.head(30).to_dict("records"),
        "output_files": {
            "summary": str(summary_path),
            "trades": str(trades_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    print(json.dumps(clean(report), indent=2))


if __name__ == "__main__":
    main()
