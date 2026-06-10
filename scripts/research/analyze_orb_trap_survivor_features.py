from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strategies.opening_failed_breakout_reversal import DEFAULT_PARAMS, generate_signals


def load_runner():
    runner_path = ROOT / "scripts/research/test_opening_failed_breakout_reversal.py"
    spec = importlib.util.spec_from_file_location("ofbr_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner: {runner_path}")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def pf(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    gains = s[s > 0].sum()
    losses = s[s < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


def compound_return(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return 0.0
    return float((1.0 + s).prod() - 1.0)


def max_drawdown(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if s.empty:
        return 0.0
    eq = (1.0 + s).cumprod()
    peak = eq.cummax()
    return float(abs((eq / peak - 1.0).min()))


def summarize(df: pd.DataFrame, label: str) -> dict:
    r = pd.to_numeric(df["net_return"], errors="coerce").dropna()
    if df.empty or r.empty:
        return {
            "label": label,
            "trades": 0,
            "total_return": 0.0,
            "expectancy": 0.0,
            "pf": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "years_negative": 0,
        }

    yearly = df.groupby("year")["net_return"].apply(compound_return)
    return {
        "label": label,
        "trades": int(len(r)),
        "total_return": compound_return(r),
        "expectancy": float(r.mean()),
        "pf": pf(r),
        "win_rate": float((r > 0).mean()),
        "max_drawdown": max_drawdown(r),
        "years_negative": int((yearly < 0).sum()),
    }


def print_summary_table(rows: list[dict], title: str) -> None:
    out = pd.DataFrame(rows)
    if out.empty:
        print(f"\n=== {title} ===")
        print("No rows")
        return

    out = out.sort_values(["pf", "expectancy", "trades"], ascending=[False, False, False])
    print(f"\n=== {title} ===")
    print(out.to_string(index=False, formatters={
        "total_return": "{:.4f}".format,
        "expectancy": "{:.6f}".format,
        "pf": "{:.4f}".format,
        "win_rate": "{:.2%}".format,
        "max_drawdown": "{:.4f}".format,
    }))


def group_stats(df: pd.DataFrame, cols: list[str], title: str) -> None:
    rows = []
    for key, g in df.groupby(cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        label = " | ".join(f"{c}={v}" for c, v in zip(cols, key))
        rows.append(summarize(g, label))
    print_summary_table(rows, title)


def main() -> None:
    runner = load_runner()

    symbol = "NVDA"
    variant_name = "both_or15_fail_rr15_extreme"
    window_label = "afternoon_1300_1500"
    entry_start, entry_end = runner.ENTRY_WINDOWS[window_label]
    variant = runner.VARIANTS[variant_name]

    csv_path = ROOT / "data/processed/NVDA/NVDA_5m_RTH_2021-06-01_2026-05-29.csv"
    params = {**DEFAULT_PARAMS, **variant, "entry_start": entry_start, "entry_end": entry_end}

    ohlcv = runner.load_ohlcv(csv_path)
    signals = generate_signals(ohlcv, params).copy()
    signals["entry_dt"] = pd.to_datetime(signals["datetime"], errors="coerce")

    combined_file = ROOT / "outputs/research/opening_failed_breakout_reversal/survivor_trades/NVDA_both_or15_fail_rr15_extreme_afternoon_1300_1500_cost_1x_2x_3x_combined_trades.csv"
    trades = pd.read_csv(combined_file)
    trades = trades[trades["cost_label"] == "cost_3x"].copy()
    trades["entry_dt"] = pd.to_datetime(trades["entry_time"], errors="coerce")
    trades["year"] = trades["entry_dt"].dt.year
    trades["month"] = trades["entry_dt"].dt.month
    trades["entry_hhmm"] = trades["entry_dt"].dt.strftime("%H:%M")
    trades["net_return"] = pd.to_numeric(trades["net_return"], errors="coerce")

    feature_cols = [
        "entry_dt", "close", "volume", "atr", "atr_pct", "vwap", "ema_fast", "ema_slow",
        "vol_sma", "or_high", "or_low", "or_range", "or_direction", "gap_pct",
        "prev_day_return",
    ]
    merged = trades.merge(signals[feature_cols], on="entry_dt", how="left")

    merged["or_range_atr"] = merged["or_range"] / merged["atr"].replace(0, np.nan)
    merged["volume_ratio"] = merged["volume"] / merged["vol_sma"].replace(0, np.nan)
    merged["vwap_dist_pct"] = merged["close"] / merged["vwap"].replace(0, np.nan) - 1.0
    merged["ema_dist_pct"] = merged["ema_fast"] / merged["ema_slow"].replace(0, np.nan) - 1.0
    merged["gap_abs"] = merged["gap_pct"].abs()
    merged["prev_day_abs_return"] = merged["prev_day_return"].abs()
    merged["long_side"] = merged["side"].eq("LONG")
    merged["short_side"] = merged["side"].eq("SHORT")
    merged["close_above_vwap"] = merged["close"] > merged["vwap"]
    merged["ema_fast_above_slow"] = merged["ema_fast"] > merged["ema_slow"]

    out_file = ROOT / "outputs/research/opening_failed_breakout_reversal/survivor_trades/NVDA_orb_trap_survivor_cost3_entry_features.csv"
    merged.to_csv(out_file, index=False)

    print(f"\nWrote feature file: {out_file}")
    print(f"Rows: {len(merged)}")
    print("\nColumns:")
    print(list(merged.columns))

    base = summarize(merged, "BASE cost_3x survivor")
    print_summary_table([base], "BASELINE COST_3X")

    group_stats(merged, ["side"], "BY SIDE")
    group_stats(merged, ["exit_reason"], "BY EXIT REASON")
    group_stats(merged, ["month"], "BY MONTH")
    group_stats(merged, ["entry_hhmm"], "BY ENTRY TIME")
    group_stats(merged, ["close_above_vwap"], "BY CLOSE ABOVE VWAP")
    group_stats(merged, ["ema_fast_above_slow"], "BY EMA FAST ABOVE SLOW")
    group_stats(merged, ["or_direction"], "BY OPENING RANGE DIRECTION")

    candidate_filters = {
        "long_only": merged["side"].eq("LONG"),
        "short_only": merged["side"].eq("SHORT"),
        "exclude_months_2_6_9_10_12": ~merged["month"].isin([2, 6, 9, 10, 12]),
        "only_months_1_3_5_7_11": merged["month"].isin([1, 3, 5, 7, 11]),
        "entry_1310_1325_or_1420_1440": merged["entry_hhmm"].between("13:10", "13:25") | merged["entry_hhmm"].between("14:20", "14:40"),
        "entry_1310_1325": merged["entry_hhmm"].between("13:10", "13:25"),
        "entry_after_1415": merged["entry_hhmm"] >= "14:15",
        "close_above_vwap_only": merged["close_above_vwap"],
        "close_below_vwap_only": ~merged["close_above_vwap"],
        "ema_fast_above_slow_only": merged["ema_fast_above_slow"],
        "ema_fast_below_slow_only": ~merged["ema_fast_above_slow"],
        "long_and_close_above_vwap": merged["side"].eq("LONG") & merged["close_above_vwap"],
        "short_and_close_below_vwap": merged["side"].eq("SHORT") & (~merged["close_above_vwap"]),
        "trend_aligned_side": (merged["side"].eq("LONG") & merged["ema_fast_above_slow"]) | (merged["side"].eq("SHORT") & (~merged["ema_fast_above_slow"])),
        "countertrend_side": (merged["side"].eq("LONG") & (~merged["ema_fast_above_slow"])) | (merged["side"].eq("SHORT") & merged["ema_fast_above_slow"]),
        "or_range_atr_le_1": merged["or_range_atr"] <= 1.0,
        "or_range_atr_gt_1": merged["or_range_atr"] > 1.0,
        "volume_ratio_ge_1_10": merged["volume_ratio"] >= 1.10,
        "volume_ratio_lt_1_10": merged["volume_ratio"] < 1.10,
        "gap_abs_le_1pct": merged["gap_abs"] <= 0.01,
        "gap_abs_gt_1pct": merged["gap_abs"] > 0.01,
        "prev_day_abs_return_le_3pct": merged["prev_day_abs_return"] <= 0.03,
        "prev_day_abs_return_gt_3pct": merged["prev_day_abs_return"] > 0.03,
    }

    rows = []
    for label, mask in candidate_filters.items():
        sub = merged[mask].copy()
        row = summarize(sub, label)
        rows.append(row)

    print_summary_table(rows, "CANDIDATE PRE-TRADE FILTERS COST_3X")

    robust_rows = [
        r for r in rows
        if r["trades"] >= 15 and r["pf"] > 1.1 and r["expectancy"] > 0 and r["years_negative"] <= 1
    ]
    print_summary_table(robust_rows, "FILTERS THAT PASS MINIMUM DIAGNOSTIC THRESHOLD")


if __name__ == "__main__":
    main()
