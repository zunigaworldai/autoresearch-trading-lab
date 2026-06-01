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

SETUP_MAP = {
    "A_only": (True, False, False),
    "B_only": (False, True, False),
    "C_only": (False, False, True),
    "A_B": (True, True, False),
    "A_C": (True, False, True),
    "B_C": (False, True, True),
    "A_B_C": (True, True, True),
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


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["date"] = x["timestamp"].dt.date

    daily = (
        x.groupby("date")
        .agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
            day_volume=("volume", "sum"),
        )
        .reset_index()
    )

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["day_return"] = daily["day_close"].pct_change()
    daily["gap_pct"] = daily["day_open"] / daily["day_close"].shift(1) - 1.0
    daily["range_pct"] = (daily["day_high"] - daily["day_low"]) / daily["day_close"].shift(1)

    daily["sma20"] = daily["day_close"].rolling(20).mean()
    daily["sma50"] = daily["day_close"].rolling(50).mean()
    daily["trend_20"] = daily["day_close"] / daily["sma20"] - 1.0
    daily["trend_50"] = daily["day_close"] / daily["sma50"] - 1.0

    daily["vol20"] = daily["day_return"].rolling(20).std()
    daily["range20"] = daily["range_pct"].rolling(20).mean()

    tr1 = daily["day_high"] - daily["day_low"]
    tr2 = (daily["day_high"] - daily["day_close"].shift(1)).abs()
    tr3 = (daily["day_low"] - daily["day_close"].shift(1)).abs()
    daily["true_range"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    daily["atr20"] = daily["true_range"].rolling(20).mean()
    daily["atr20_pct"] = daily["atr20"] / daily["day_close"].shift(1)

    # Shift all regime features by 1 day where needed so they are known before today's trading.
    known_cols = [
        "day_return",
        "range_pct",
        "trend_20",
        "trend_50",
        "vol20",
        "range20",
        "atr20_pct",
    ]

    for col in known_cols:
        daily[f"prev_{col}"] = daily[col].shift(1)

    # gap_pct is known at today's open, so do not shift.
    daily["dow"] = daily["date"].dt.dayofweek

    keep = [
        "date",
        "gap_pct",
        "dow",
        "prev_day_return",
        "prev_range_pct",
        "prev_trend_20",
        "prev_trend_50",
        "prev_vol20",
        "prev_range20",
        "prev_atr20_pct",
    ]

    return daily[keep].copy()


def build_params(setup_combo: str, session_start: str, session_end: str) -> dict:
    if setup_combo not in SETUP_MAP:
        raise SystemExit(f"ERROR: unknown setup_combo={setup_combo}. Valid: {sorted(SETUP_MAP)}")

    enable_a, enable_b, enable_c = SETUP_MAP[setup_combo]

    params = DEFAULT_PARAMS.copy()
    params["use_rth"] = True
    params["session_start"] = session_start
    params["session_end"] = session_end
    params["enable_a"] = enable_a
    params["enable_b"] = enable_b
    params["enable_c"] = enable_c

    return params


def summarize_returns(trade_returns: pd.Series, equity: pd.Series, trades: pd.DataFrame) -> dict[str, Any]:
    report = score_report(
        equity=equity,
        returns=equity.pct_change().fillna(0.0),
        trade_returns=trade_returns,
    )

    if trades.empty:
        report.update(
            {
                "total_trades": 0,
                "tp_hits": 0,
                "stop_hits": 0,
                "be_exits": 0,
                "eod_exits": 0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "largest_win_share": 0.0,
            }
        )
        return report

    net = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    gross_profit = float(net[net > 0].sum()) if len(net) else 0.0
    best_trade = float(net.max()) if len(net) else 0.0

    report.update(
        {
            "total_trades": int(len(trades)),
            "tp_hits": int((trades["exit_reason"] == "TP").sum()) if "exit_reason" in trades.columns else 0,
            "stop_hits": int((trades["exit_reason"] == "STOP").sum()) if "exit_reason" in trades.columns else 0,
            "be_exits": int((trades["exit_reason"] == "BE").sum()) if "exit_reason" in trades.columns else 0,
            "eod_exits": int((trades["exit_reason"] == "EOD").sum()) if "exit_reason" in trades.columns else 0,
            "best_trade": best_trade,
            "worst_trade": float(net.min()) if len(net) else 0.0,
            "largest_win_share": best_trade / gross_profit if gross_profit > 0 else 0.0,
        }
    )

    return report


def run_baseline_trades(
    df: pd.DataFrame,
    setup_combo: str,
    session_start: str,
    session_end: str,
    exit_policy: str,
    cost_per_side: float,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    params = build_params(setup_combo, session_start, session_end)
    signals = generate_signals(df, params)

    equity, trade_returns, trades = run_trade_simulation(
        signals=signals,
        cost_per_side=cost_per_side,
        force_eod_exit=True,
        exit_policy=exit_policy,
    )

    return equity, trade_returns, trades


def attach_regime_features(trades: pd.DataFrame, daily_features: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()

    out = trades.copy()

    timestamp_col = None
    for candidate in ["entry_time", "entry_timestamp", "timestamp", "datetime"]:
        if candidate in out.columns:
            timestamp_col = candidate
            break

    if timestamp_col is None:
        # Fallback: no timestamps; cannot attach features.
        out["date"] = pd.NaT
        return out

    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out["date"] = out[timestamp_col].dt.normalize()

    features = daily_features.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()

    out = out.merge(features, on="date", how="left")
    return out


def pf_from_returns(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    gains = values[values > 0].sum()
    losses = values[values < 0].sum()
    if losses == 0:
        return 10.0 if gains > 0 else 0.0
    return float(gains / abs(losses))


def max_dd_from_returns(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if values.empty:
        return 0.0
    equity = (1.0 + values).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(abs(dd.min()))


def evaluate_filter(trades: pd.DataFrame, condition_name: str, mask: pd.Series) -> dict[str, Any]:
    kept = trades[mask.fillna(False)].copy()

    if kept.empty:
        return {
            "filter": condition_name,
            "trades": 0,
            "pf": 0.0,
            "expectancy": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "years": 0,
            "years_pass": 0,
            "largest_win_share": 0.0,
        }

    returns = pd.to_numeric(kept["net_return"], errors="coerce").dropna()
    years = kept["date"].dt.year.dropna().unique().tolist() if "date" in kept.columns else []

    years_pass = 0
    if "date" in kept.columns:
        for _, group in kept.groupby(kept["date"].dt.year):
            r = pd.to_numeric(group["net_return"], errors="coerce").dropna()
            if len(r) >= 5 and pf_from_returns(r) > 1.0 and float(r.mean()) > 0:
                years_pass += 1

    gross_profit = float(returns[returns > 0].sum()) if len(returns) else 0.0
    best_trade = float(returns.max()) if len(returns) else 0.0

    return {
        "filter": condition_name,
        "trades": int(len(returns)),
        "pf": pf_from_returns(returns),
        "expectancy": float(returns.mean()) if len(returns) else 0.0,
        "win_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "max_drawdown": max_dd_from_returns(returns),
        "years": int(len(years)),
        "years_pass": int(years_pass),
        "largest_win_share": best_trade / gross_profit if gross_profit > 0 else 0.0,
    }


def generate_filter_candidates(trades: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    candidates: list[tuple[str, pd.Series]] = []

    def add(name: str, mask: pd.Series) -> None:
        candidates.append((name, mask))

    numeric_features = [
        "gap_pct",
        "prev_day_return",
        "prev_range_pct",
        "prev_trend_20",
        "prev_trend_50",
        "prev_vol20",
        "prev_range20",
        "prev_atr20_pct",
    ]

    for col in numeric_features:
        if col not in trades.columns:
            continue

        s = pd.to_numeric(trades[col], errors="coerce")
        if s.dropna().nunique() < 5:
            continue

        qs = s.quantile([0.20, 0.33, 0.50, 0.67, 0.80]).dropna().to_dict()

        for q, val in qs.items():
            add(f"{col} >= q{int(q*100)} ({val:.6f})", s >= val)
            add(f"{col} <= q{int(q*100)} ({val:.6f})", s <= val)

        if "return" in col or "trend" in col or "gap" in col:
            add(f"{col} > 0", s > 0)
            add(f"{col} < 0", s < 0)

    if "dow" in trades.columns:
        for dow in sorted(pd.to_numeric(trades["dow"], errors="coerce").dropna().unique()):
            add(f"dow == {int(dow)}", pd.to_numeric(trades["dow"], errors="coerce") == dow)
            add(f"dow != {int(dow)}", pd.to_numeric(trades["dow"], errors="coerce") != dow)

    return candidates


def score_filter(row: dict[str, Any]) -> float:
    trades = safe_int(row["trades"])
    pf = safe_float(row["pf"])
    exp = safe_float(row["expectancy"])
    dd = safe_float(row["max_drawdown"])
    years_pass = safe_int(row["years_pass"])
    largest_win_share = safe_float(row["largest_win_share"])

    if trades < 30:
        return -999.0

    score = 0.0
    score += min(pf, 2.0) * 0.30
    score += min(max(exp, 0.0), 0.004) / 0.004 * 0.25
    score += min(years_pass / 6.0, 1.0) * 0.20
    score += max(0.0, 1.0 - min(dd, 0.20) / 0.20) * 0.15
    score += max(0.0, 1.0 - min(largest_win_share, 1.0)) * 0.10
    return float(score)


def run_regime_diagnostics(
    csv_path: Path,
    symbol: str,
    timeframe: str,
    setup_combo: str,
    session_start: str,
    session_end: str,
    exit_policy: str,
    output_dir: Path,
) -> dict[str, Any]:
    if exit_policy not in VALID_EXIT_POLICIES:
        raise SystemExit(f"ERROR: invalid exit_policy={exit_policy}. Valid: {sorted(VALID_EXIT_POLICIES)}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_ohlcv(csv_path)
    daily_features = build_daily_features(df)

    all_rows = []
    trades_output_rows = []

    for cost_label, cost_per_side in COST_SCENARIOS.items():
        equity, trade_returns, trades = run_baseline_trades(
            df=df,
            setup_combo=setup_combo,
            session_start=session_start,
            session_end=session_end,
            exit_policy=exit_policy,
            cost_per_side=cost_per_side,
        )

        trades_with_features = attach_regime_features(trades, daily_features)

        if cost_label == "cost_1x":
            trades_output_rows = trades_with_features.to_dict("records")

        baseline = summarize_returns(trade_returns, equity, trades)
        baseline_row = {
            "symbol": symbol,
            "timeframe": timeframe,
            "setup_combo": setup_combo,
            "session_start": session_start,
            "session_end": session_end,
            "exit_policy": exit_policy,
            "cost_label": cost_label,
            "filter": "BASELINE_NO_FILTER",
            "filter_score": score_filter(
                {
                    "trades": baseline["total_trades"],
                    "pf": baseline["profit_factor"],
                    "expectancy": baseline["expectancy"],
                    "max_drawdown": baseline["max_drawdown"],
                    "years_pass": 0,
                    "largest_win_share": baseline["largest_win_share"],
                }
            ),
        }
        baseline_row.update(baseline)
        baseline_row["years"] = int(trades_with_features["date"].dt.year.nunique()) if not trades_with_features.empty and "date" in trades_with_features.columns else 0
        baseline_row["years_pass"] = None
        all_rows.append(baseline_row)

        if cost_label != "cost_1x":
            continue

        filters = generate_filter_candidates(trades_with_features)
        for name, mask in filters:
            result = evaluate_filter(trades_with_features, name, mask)
            row = {
                "symbol": symbol,
                "timeframe": timeframe,
                "setup_combo": setup_combo,
                "session_start": session_start,
                "session_end": session_end,
                "exit_policy": exit_policy,
                "cost_label": cost_label,
                **result,
            }
            row["filter_score"] = score_filter(row)
            all_rows.append(row)

    results = pd.DataFrame(all_rows)

    results = results.sort_values(
        ["filter_score", "pf", "expectancy", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    safe = f"{symbol}_{timeframe}_{setup_combo}_{session_start.replace(':','')}_{session_end.replace(':','')}_{exit_policy}"

    results_path = output_dir / f"{safe}_regime_filter_results.csv"
    trades_path = output_dir / f"{safe}_trades_with_regime_features.csv"
    report_path = output_dir / f"{safe}_regime_filter_report.json"

    results.to_csv(results_path, index=False)
    pd.DataFrame(trades_output_rows).to_csv(trades_path, index=False)

    report = {
        "ok": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "setup_combo": setup_combo,
        "session_start": session_start,
        "session_end": session_end,
        "exit_policy": exit_policy,
        "csv_path": str(csv_path),
        "rows_tested": int(len(results)),
        "top_filters": results.head(25).to_dict("records"),
        "output_files": {
            "results": str(results_path),
            "trades_with_features": str(trades_path),
            "report": str(report_path),
        },
        "warning": "Filters are diagnostic only. They must be retested with walk-forward to avoid overfitting.",
    }

    report_path.write_text(json.dumps(clean_for_json(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose simple regime filters for one setup.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--setup-combo", default="A_only")
    parser.add_argument("--session-start", default="09:30")
    parser.add_argument("--session-end", default="10:30")
    parser.add_argument("--exit-policy", default="break_even_1r", choices=sorted(VALID_EXIT_POLICIES))
    parser.add_argument("--output-dir", default="outputs/research/regime_diagnostics")

    args = parser.parse_args()

    report = run_regime_diagnostics(
        csv_path=Path(args.csv),
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        setup_combo=args.setup_combo,
        session_start=args.session_start,
        session_end=args.session_end,
        exit_policy=args.exit_policy,
        output_dir=Path(args.output_dir),
    )

    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()
