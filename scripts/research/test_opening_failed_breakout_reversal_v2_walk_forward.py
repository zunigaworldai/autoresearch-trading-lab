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


COST_SCENARIOS = {"cost_1x": 0.0005, "cost_2x": 0.0010, "cost_3x": 0.0015}
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
    eq = (1.0 + returns).cumprod()
    peak = eq.cummax()
    return float(abs((eq / peak - 1.0).min()))


def trade_dates(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype="datetime64[ns]")
    for col in ["entry_time", "entry_timestamp", "timestamp", "datetime", "date"]:
        if col in trades.columns:
            return pd.to_datetime(trades[col], errors="coerce")
    return pd.Series(dtype="datetime64[ns]")


def summarize(trades: pd.DataFrame, prefix: str = "") -> dict[str, Any]:
    out = trades.copy()
    if out.empty:
        return {
            f"{prefix}trades": 0,
            f"{prefix}pf": 0.0,
            f"{prefix}expectancy": 0.0,
            f"{prefix}win_rate": 0.0,
            f"{prefix}total_return": 0.0,
            f"{prefix}max_drawdown": 0.0,
            f"{prefix}years_negative": 0,
            f"{prefix}months_negative": 0,
            f"{prefix}months_total_active": 0,
        }

    out["entry_dt"] = trade_dates(out)
    out["net_return"] = pd.to_numeric(out["net_return"], errors="coerce")
    out = out.dropna(subset=["entry_dt", "net_return"]).copy()
    r = out["net_return"]

    yearly = out.groupby(out["entry_dt"].dt.to_period("Y"))["net_return"].apply(compound_return)
    monthly = out.groupby(out["entry_dt"].dt.to_period("M"))["net_return"].apply(compound_return)

    return {
        f"{prefix}trades": int(len(out)),
        f"{prefix}pf": profit_factor(r),
        f"{prefix}expectancy": float(r.mean()) if len(r) else 0.0,
        f"{prefix}win_rate": float((r > 0).mean()) if len(r) else 0.0,
        f"{prefix}total_return": compound_return(r),
        f"{prefix}max_drawdown": max_drawdown(r),
        f"{prefix}years_negative": int((yearly < 0).sum()) if len(yearly) else 0,
        f"{prefix}months_negative": int((monthly < 0).sum()) if len(monthly) else 0,
        f"{prefix}months_total_active": int(len(monthly)),
    }


def monte_carlo_summary(returns: pd.Series, n: int = 2000, seed: int = 42) -> dict[str, Any]:
    r = pd.to_numeric(returns, errors="coerce").dropna().to_numpy()
    if len(r) == 0:
        return {"mc_n": n, "mc_p05_return": 0.0, "mc_p50_return": 0.0, "mc_p95_return": 0.0, "mc_p95_maxdd": 0.0}

    rng = np.random.default_rng(seed)
    totals = []
    drawdowns = []
    for _ in range(n):
        sample = rng.choice(r, size=len(r), replace=True)
        s = pd.Series(sample)
        totals.append(compound_return(s))
        drawdowns.append(max_drawdown(s))

    return {
        "mc_n": n,
        "mc_p05_return": float(np.percentile(totals, 5)),
        "mc_p50_return": float(np.percentile(totals, 50)),
        "mc_p95_return": float(np.percentile(totals, 95)),
        "mc_p95_maxdd": float(np.percentile(drawdowns, 95)),
    }


def classify_segment(row: dict[str, Any]) -> str:
    if row["test_trades"] < 3:
        return "test_tiny_sample"
    if row["test_pf"] <= 1.0 or row["test_expectancy"] <= 0:
        return "test_failed"
    if row["train_trades"] < 8:
        return "train_tiny_sample"
    if row["train_pf"] <= 1.0 or row["train_expectancy"] <= 0:
        return "train_failed"
    return "pass"


def apply_entry_core(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    hhmm = pd.to_datetime(out["datetime"], errors="coerce").dt.strftime("%H:%M")
    mask = hhmm.between("13:10", "13:25") | hhmm.between("14:20", "14:40")

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


def main() -> None:
    runner = load_runner()

    csv_path = ROOT / "data/processed/NVDA/NVDA_5m_RTH_2021-06-01_2026-05-29.csv"
    ohlcv = runner.load_ohlcv(csv_path)

    base_variant = runner.VARIANTS[BASE_VARIANT_NAME]
    entry_start, entry_end = runner.ENTRY_WINDOWS[BASE_WINDOW_LABEL]
    params = {**DEFAULT_PARAMS, **base_variant, "entry_start": entry_start, "entry_end": entry_end}

    signals = generate_signals(ohlcv, params)
    signals = apply_entry_core(signals)

    out_dir = ROOT / "outputs/research/opening_failed_breakout_reversal_v2_walk_forward"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    all_segment_rows = []
    all_trades = []

    for cost_label, cost_per_side in COST_SCENARIOS.items():
        _, _, trades = run_trade_simulation(
            signals=signals,
            cost_per_side=cost_per_side,
            force_eod_exit=True,
            exit_policy=EXIT_POLICY,
        )

        trades = trades.copy()
        trades["symbol"] = "NVDA"
        trades["timeframe"] = "5m"
        trades["strategy_family"] = "opening_failed_breakout_reversal_v2_walk_forward"
        trades["base_variant"] = BASE_VARIANT_NAME
        trades["base_window_label"] = BASE_WINDOW_LABEL
        trades["variant"] = "robust_entry_core"
        trades["cost_label"] = cost_label
        trades["cost_per_side"] = cost_per_side
        trades["exit_policy"] = EXIT_POLICY
        trades["entry_dt"] = trade_dates(trades)
        trades["net_return"] = pd.to_numeric(trades["net_return"], errors="coerce")
        trades = trades.dropna(subset=["entry_dt", "net_return"]).copy()
        all_trades.append(trades)

        summary = {
            "symbol": "NVDA",
            "timeframe": "5m",
            "variant": "robust_entry_core",
            "cost_label": cost_label,
            "cost_per_side": cost_per_side,
        }
        summary.update(summarize(trades, ""))
        summary.update(summarize(trades[trades["entry_dt"] >= pd.Timestamp("2024-01-01")], "oos_2024_"))
        summary.update(summarize(trades[trades["entry_dt"] >= pd.Timestamp("2025-01-01")], "oos_2025_"))
        summary.update(monte_carlo_summary(trades["net_return"], n=2000, seed=42))
        all_summary_rows.append(summary)

        splits = [
            ("train_2022_test_2023", "2022-01-01", "2023-01-01", "2024-01-01"),
            ("train_2022_2023_test_2024", "2022-01-01", "2024-01-01", "2025-01-01"),
            ("train_2022_2024_test_2025", "2022-01-01", "2025-01-01", "2026-01-01"),
            ("train_2022_2025_test_2026", "2022-01-01", "2026-01-01", "2027-01-01"),
        ]

        for split_label, train_start, test_start, test_end in splits:
            train = trades[(trades["entry_dt"] >= pd.Timestamp(train_start)) & (trades["entry_dt"] < pd.Timestamp(test_start))].copy()
            test = trades[(trades["entry_dt"] >= pd.Timestamp(test_start)) & (trades["entry_dt"] < pd.Timestamp(test_end))].copy()

            row = {
                "split": split_label,
                "cost_label": cost_label,
                "cost_per_side": cost_per_side,
                "train_start": train_start,
                "test_start": test_start,
                "test_end": test_end,
            }
            row.update(summarize(train, "train_"))
            row.update(summarize(test, "test_"))
            row["decision"] = classify_segment(row)
            all_segment_rows.append(row)

    summary_df = pd.DataFrame(all_summary_rows)
    segments_df = pd.DataFrame(all_segment_rows)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    summary_path = out_dir / "NVDA_5m_orb_trap_v2_walk_forward_summary.csv"
    segments_path = out_dir / "NVDA_5m_orb_trap_v2_walk_forward_segments.csv"
    trades_path = out_dir / "NVDA_5m_orb_trap_v2_walk_forward_trades.csv"
    report_path = out_dir / "NVDA_5m_orb_trap_v2_walk_forward.json"

    summary_df.to_csv(summary_path, index=False)
    segments_df.to_csv(segments_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    cost3_summary = summary_df[summary_df["cost_label"] == "cost_3x"].to_dict("records")
    cost3_segments = segments_df[segments_df["cost_label"] == "cost_3x"].to_dict("records")

    report = {
        "ok": True,
        "strategy_family": "opening_failed_breakout_reversal_v2_walk_forward",
        "symbol": "NVDA",
        "timeframe": "5m",
        "base_variant": BASE_VARIANT_NAME,
        "base_window_label": BASE_WINDOW_LABEL,
        "variant": "robust_entry_core",
        "exit_policy": EXIT_POLICY,
        "warning": "Walk-forward diagnostic only. No paper/live without explicit approval after segment review.",
        "summary_cost_3x": cost3_summary,
        "segments_cost_3x": cost3_segments,
        "segment_decision_counts_cost_3x": pd.Series([r["decision"] for r in cost3_segments]).value_counts().to_dict(),
        "output_files": {
            "summary": str(summary_path),
            "segments": str(segments_path),
            "trades": str(trades_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(json.dumps(clean(report), indent=2), encoding="utf-8")
    print(json.dumps(clean(report), indent=2))


if __name__ == "__main__":
    main()
