from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SYMBOLS = ["SPY", "QQQ", "NVDA", "AAPL", "MSFT"]
DEFAULT_OUTPUT_DIR = Path("outputs/research/data_expansion")
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


@dataclass
class ProviderProfile:
    provider: str
    role: str
    recommended_use: str
    strengths: list[str]
    weaknesses: list[str]
    status: str
    priority: int


def clean_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_for_json(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
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


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def years_back(end_date: date, years: int) -> date:
    try:
        return end_date.replace(year=end_date.year - years)
    except ValueError:
        return end_date.replace(month=2, day=28, year=end_date.year - years)


def chunk_by_year(start: date, end: date) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(date(cursor.year, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def estimate_rth_rows(years: int, timeframe: str) -> int:
    trading_days = int(252 * years)
    bars_per_day = {
        "1m": 391,
        "2m": 196,
        "3m": 131,
        "5m": 79,
        "10m": 40,
        "15m": 27,
        "30m": 14,
        "1h": 7,
    }.get(timeframe, 391)
    return trading_days * bars_per_day


def audit_existing_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "rows": 0, "start": None, "end": None, "size_mb": 0.0}

    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        df = pd.read_csv(path, usecols=["timestamp"])
        ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    except Exception:
        return {
            "exists": True,
            "rows": None,
            "start": None,
            "end": None,
            "size_mb": round(size_mb, 3),
            "read_error": True,
        }

    if ts.empty:
        return {"exists": True, "rows": 0, "start": None, "end": None, "size_mb": round(size_mb, 3)}

    return {
        "exists": True,
        "rows": int(len(ts)),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "trading_days": int(ts.dt.date.nunique()),
        "size_mb": round(size_mb, 3),
    }


def audit_current_data(symbols: list[str], timeframes: list[str]) -> list[dict[str, Any]]:
    rows = []
    for symbol in symbols:
        for timeframe in timeframes:
            raw_matches = sorted((DEFAULT_RAW_DIR / symbol).glob(f"{symbol}_{timeframe}_*.csv"))
            processed_matches = sorted((DEFAULT_PROCESSED_DIR / symbol).glob(f"{symbol}_{timeframe}_RTH_*.csv"))
            raw_latest = raw_matches[-1] if raw_matches else None
            processed_latest = processed_matches[-1] if processed_matches else None
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "raw_latest": str(raw_latest) if raw_latest else "",
                    "raw_audit": audit_existing_file(raw_latest) if raw_latest else audit_existing_file(Path("__missing__")),
                    "processed_latest": str(processed_latest) if processed_latest else "",
                    "processed_audit": audit_existing_file(processed_latest) if processed_latest else audit_existing_file(Path("__missing__")),
                }
            )
    return rows


def provider_profiles() -> list[ProviderProfile]:
    return [
        ProviderProfile(
            provider="Alpaca",
            role="Immediate downloader / current integration",
            recommended_use="Use first because keys and scripts already work. Try 5-year 1m expansion for equities/ETFs if plan access allows.",
            strengths=[
                "Already integrated in this repo.",
                "Historical stock bars endpoint fits current OHLCV pipeline.",
                "Fastest path to 5-year test if access/limits allow.",
            ],
            weaknesses=[
                "May require paid plan/feed for longer/deeper historical intraday data.",
                "IEX feed can differ from consolidated SIP feed.",
                "Not ideal for long-term institutional tick/order-book research.",
            ],
            status="primary_now",
            priority=1,
        ),
        ProviderProfile(
            provider="Databento",
            role="Professional historical market data",
            recommended_use="Evaluate for institutional-grade equities, futures, tick, trades, quotes, and bar data if budget allows.",
            strengths=[
                "Professional historical API.",
                "Supports multiple normalized schemas including OHLCV and tick-style data.",
                "Good candidate for futures/market microstructure expansion later.",
            ],
            weaknesses=[
                "Likely more expensive than basic retail APIs.",
                "Requires new integration module.",
            ],
            status="evaluate_next",
            priority=2,
        ),
        ProviderProfile(
            provider="Massive/Polygon-style Stocks API",
            role="Retail/prosumer historical aggregates",
            recommended_use="Evaluate as alternative for long-range stock bars and later options data.",
            strengths=[
                "Stocks aggregate/custom bars are directly aligned with OHLCV needs.",
                "Potentially useful for options/futures/forex expansion depending plan.",
            ],
            weaknesses=[
                "Requires separate subscription and integration.",
                "Need to verify minute-depth availability and rate limits for 5-10 year intraday.",
            ],
            status="evaluate_next",
            priority=3,
        ),
        ProviderProfile(
            provider="QuantConnect / LEAN data ecosystem",
            role="Research/backtesting platform option",
            recommended_use="Evaluate if we want to run broader LEAN backtests with long historical equities data instead of only local CSV pipeline.",
            strengths=[
                "Research/backtest ecosystem built for quant workflows.",
                "Can be useful for cross-checking our local results.",
            ],
            weaknesses=[
                "Different execution/backtest environment from our local scripts.",
                "Exporting/reconciling data into our current repo may require extra work.",
            ],
            status="evaluate_parallel",
            priority=4,
        ),
    ]


def build_download_plan(
    symbols: list[str],
    years: int,
    stress_years: int,
    end_date: date,
    timeframes: list[str],
    provider: str,
) -> dict[str, Any]:
    start_5y = years_back(end_date, years)
    start_10y = years_back(end_date, stress_years)
    chunks_5y = chunk_by_year(start_5y, end_date)
    chunks_10y = chunk_by_year(start_10y, end_date)
    rows = []

    for symbol in symbols:
        for chunk_start, chunk_end in chunks_5y:
            rows.append(
                {
                    "phase": "primary_5y",
                    "provider": provider,
                    "symbol": symbol,
                    "download_timeframe": "1m",
                    "derived_timeframes": ",".join(timeframes),
                    "start_date": str(chunk_start),
                    "end_date": str(chunk_end),
                    "start_utc": f"{chunk_start}T00:00:00Z",
                    "end_utc": f"{chunk_end}T23:59:59Z",
                    "estimated_rth_1m_rows": estimate_rth_rows(1, "1m"),
                    "status": "planned",
                }
            )

        for chunk_start, chunk_end in chunks_10y:
            if chunk_start >= start_5y:
                continue
            rows.append(
                {
                    "phase": "stress_10y_extension",
                    "provider": provider,
                    "symbol": symbol,
                    "download_timeframe": "1m",
                    "derived_timeframes": ",".join(timeframes),
                    "start_date": str(chunk_start),
                    "end_date": str(chunk_end),
                    "start_utc": f"{chunk_start}T00:00:00Z",
                    "end_utc": f"{chunk_end}T23:59:59Z",
                    "estimated_rth_1m_rows": estimate_rth_rows(1, "1m"),
                    "status": "planned_after_5y",
                }
            )

    return {
        "primary_start_date": str(start_5y),
        "stress_start_date": str(start_10y),
        "end_date": str(end_date),
        "download_rows": rows,
    }


def build_alpaca_batch_script(plan_rows: list[dict[str, Any]], output_path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'echo "Checking Alpaca environment variables..."',
        'test -n "${ALPACA_API_KEY_ID:-}" || (echo "Missing ALPACA_API_KEY_ID" && exit 1)',
        'test -n "${ALPACA_API_SECRET_KEY:-}" || (echo "Missing ALPACA_API_SECRET_KEY" && exit 1)',
        "",
        "# Downloads 1m bars and lets scripts/data/download_alpaca_bars.py resample to 2m/3m/5m.",
        "# Run primary_5y first. Run stress_10y_extension only after storage/provider limits are confirmed.",
        "",
    ]

    for row in plan_rows:
        if row["provider"].lower() != "alpaca":
            continue
        if row["phase"] != "primary_5y":
            continue
        symbol = row["symbol"]
        start_utc = row["start_utc"]
        end_utc = row["end_utc"]
        lines.extend(
            [
                f'echo "Downloading {symbol} {start_utc} to {end_utc}"',
                "python scripts/data/download_alpaca_bars.py \\",
                f"  --symbols {symbol} \\",
                f"  --start {start_utc} \\",
                f"  --end {end_utc} \\",
                "  --feed iex",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.chmod(0o755)


def build_rth_processing_script(symbols: list[str], output_path: Path) -> None:
    symbols_literal = "[" + ", ".join([repr(s) for s in symbols]) + "]"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "python - <<'PY'",
        "from pathlib import Path",
        "from datetime import time",
        "import pandas as pd",
        "",
        f"symbols = {symbols_literal}",
        'timeframes = ["1m", "2m", "3m", "5m"]',
        "",
        "for symbol in symbols:",
        '    raw_dir = Path(f"data/raw/{symbol}")',
        '    out_dir = Path(f"data/processed/{symbol}")',
        "    out_dir.mkdir(parents=True, exist_ok=True)",
        "    for timeframe in timeframes:",
        '        files = sorted(raw_dir.glob(f"{symbol}_{timeframe}_*.csv"))',
        "        if not files:",
        '            print(f"MISSING raw files for {symbol} {timeframe}")',
        "            continue",
        "        frames = []",
        "        for src in files:",
        "            df = pd.read_csv(src)",
        '            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")',
        '            for col in ["open", "high", "low", "close", "volume"]:',
        '                df[col] = pd.to_numeric(df[col], errors="coerce")',
        '            df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])',
        "            frames.append(df)",
        "        if not frames:",
        '            print(f"NO VALID DATA: {symbol} {timeframe}")',
        "            continue",
        "        all_df = pd.concat(frames, ignore_index=True)",
        '        all_df = all_df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)',
        "        rth = all_df[",
        '            (all_df["timestamp"].dt.time >= time(9, 30))',
        '            & (all_df["timestamp"].dt.time <= time(16, 0))',
        "        ].copy()",
        '        start = rth["timestamp"].min().date()',
        '        end = rth["timestamp"].max().date()',
        '        dst = out_dir / f"{symbol}_{timeframe}_RTH_{start}_{end}.csv"',
        "        rth.to_csv(dst, index=False)",
        '        print(f"{symbol} {timeframe} raw={len(all_df)} rth={len(rth)} file={dst}")',
        "PY",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.chmod(0o755)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    end_date = parse_date(args.end_date) if args.end_date else date.today()
    symbols = [symbol.upper() for symbol in args.symbols]
    timeframes = args.timeframes

    plan = build_download_plan(
        symbols=symbols,
        years=args.years,
        stress_years=args.stress_years,
        end_date=end_date,
        timeframes=timeframes,
        provider=args.provider,
    )

    audit = audit_current_data(symbols=symbols, timeframes=timeframes)
    providers = [asdict(profile) for profile in provider_profiles()]

    estimated = {
        "symbols": symbols,
        "timeframes": timeframes,
        "primary_years": args.years,
        "stress_years": args.stress_years,
        "estimated_1m_rth_rows_per_symbol_5y": estimate_rth_rows(args.years, "1m"),
        "estimated_5m_rth_rows_per_symbol_5y": estimate_rth_rows(args.years, "5m"),
        "estimated_1m_rth_rows_all_symbols_5y": estimate_rth_rows(args.years, "1m") * len(symbols),
        "estimated_5m_rth_rows_all_symbols_5y": estimate_rth_rows(args.years, "5m") * len(symbols),
    }

    recommendations = [
        "Use Alpaca first because current repo integration already works; treat it as the fastest 5-year feasibility test.",
        "If Alpaca cannot provide full 5-year 1m coverage, evaluate Databento and Massive/Polygon-style historical aggregates next.",
        "Use 5 years as the new research baseline; use 10 years as stress/regime robustness, not as the only selection judge.",
        "Always resample from 1m into 2m/3m/5m so all timeframes share the same source data and avoid provider aggregation differences.",
        "After data expansion, rerun the full pipeline: setup combos -> ranking -> IS/OOS -> paper watch selection -> forward paper execution.",
    ]

    return {
        "ok": True,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "objective": "Expand intraday historical data from discovery sample to 5-year research baseline and 10-year stress target.",
        "provider_selected_for_initial_plan": args.provider,
        "symbols": symbols,
        "timeframes": timeframes,
        "plan": plan,
        "current_data_audit": audit,
        "provider_profiles": providers,
        "estimated_scale": estimated,
        "quality_requirements": {
            "source_timeframe": "1m raw bars",
            "derived_timeframes": timeframes,
            "rth_filter": "09:30-16:00 ET",
            "duplicates": "drop duplicate timestamps",
            "numeric_validation": "open/high/low/close/volume must parse numeric",
            "corporate_actions": "verify adjustment mode and split/dividend behavior per provider before final production tests",
            "costs": "retest cost_1x, cost_2x, cost_3x after expansion",
        },
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 5-year/10-year intraday data expansion plan.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=["1m", "2m", "3m", "5m"])
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--stress-years", type=int, default=10)
    parser.add_argument("--end-date", default="")
    parser.add_argument("--provider", default="alpaca", choices=["alpaca", "databento", "massive_polygon", "quantconnect"])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(args)
    plan_rows = report["plan"]["download_rows"]

    plan_csv = output_dir / "data_expansion_download_plan.csv"
    audit_csv = output_dir / "current_data_audit.csv"
    provider_csv = output_dir / "provider_evaluation.csv"
    report_json = output_dir / "data_expansion_plan_report.json"
    alpaca_script = output_dir / "run_alpaca_5y_download_batches.sh"
    rth_script = output_dir / "process_rth_after_expansion.sh"

    pd.DataFrame(plan_rows).to_csv(plan_csv, index=False)
    pd.DataFrame(report["current_data_audit"]).to_csv(audit_csv, index=False)
    pd.DataFrame(report["provider_profiles"]).to_csv(provider_csv, index=False)

    build_alpaca_batch_script(plan_rows, alpaca_script)
    build_rth_processing_script(report["symbols"], rth_script)

    report["output_files"] = {
        "plan_csv": str(plan_csv),
        "audit_csv": str(audit_csv),
        "provider_csv": str(provider_csv),
        "report_json": str(report_json),
        "alpaca_batch_script": str(alpaca_script),
        "rth_processing_script": str(rth_script),
    }

    report_json.write_text(json.dumps(clean_for_json(report), indent=2), encoding="utf-8")
    print(json.dumps(clean_for_json(report), indent=2))


if __name__ == "__main__":
    main()
