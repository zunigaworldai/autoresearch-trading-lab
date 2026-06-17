from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TZ = "America/New_York"
RTH_START = pd.to_datetime("09:30").time()
RTH_END = pd.to_datetime("16:00").time()
RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed_standard"
STANDARD_TIMEFRAMES = ["5m", "15m", "30m", "1h", "1d"]
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def fail(reason: str, details: dict | None = None) -> None:
    payload = {"ok": False, "reason": reason, "details": details or {}}
    print(json.dumps(payload, indent=2))
    raise SystemExit(1)


def success(details: dict) -> None:
    payload = {"ok": True, "details": details}
    print(json.dumps(payload, indent=2))


def list_symbol_files(symbol: str, root: Path) -> list[Path]:
    symbol_path = root / symbol
    if not symbol_path.exists():
        return []

    files = sorted(symbol_path.glob(f"{symbol}_1m_*.csv"))
    return files


def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        fail("csv_read_error", {"path": str(path), "error": str(exc)})

    return df


def validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        fail("missing_required_columns", {"path": str(path), "missing": missing, "found": list(df.columns)})


def normalize_timestamp(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    bad = df["timestamp"].isna().sum()
    if bad > 0:
        fail("invalid_timestamps", {"path": str(path), "bad_rows": int(bad)})

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(TZ)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(TZ)

    return df


def validate_numeric_columns(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["open", "high", "low", "close", "volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = df[numeric_cols].isna().sum()
    bad = {k: int(v) for k, v in bad.items() if v > 0}
    if bad:
        fail("invalid_numeric_values", {"path": str(path), "invalid_counts": bad})

    return df


def validate_price_logic(df: pd.DataFrame, path: Path) -> None:
    bad_high = df[df["high"] < df[["open", "close", "low"]].max(axis=1)]
    bad_low = df[df["low"] > df[["open", "close", "high"]].min(axis=1)]

    if not bad_high.empty:
        fail("invalid_high_price", {"path": str(path), "bad_rows": int(len(bad_high))})
    if not bad_low.empty:
        fail("invalid_low_price", {"path": str(path), "bad_rows": int(len(bad_low))})

    non_positive = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    if not non_positive.empty:
        fail("non_positive_prices", {"path": str(path), "bad_rows": int(len(non_positive))})


def validate_volume(df: pd.DataFrame, path: Path) -> None:
    bad_volume = df[df["volume"] < 0]
    if not bad_volume.empty:
        fail("negative_volume", {"path": str(path), "bad_rows": int(len(bad_volume))})


def validate_order_and_duplicates(df: pd.DataFrame, path: Path) -> None:
    if not df["timestamp"].is_monotonic_increasing:
        fail("timestamps_not_sorted", {"path": str(path)})

    duplicates = int(df["timestamp"].duplicated().sum())
    if duplicates > 0:
        fail("duplicate_timestamps", {"path": str(path), "duplicates": duplicates})


def validate_rth_timestamps(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"missing_rth_minutes": 0, "rth_days": 0}

    df = df.set_index("timestamp").sort_index()
    df = df.between_time(RTH_START, RTH_END, inclusive="both")
    if df.empty:
        return {"missing_rth_minutes": 0, "rth_days": 0}

    missing_total = 0
    business_days = 0

    for date, group in df.groupby(df.index.date):
        business_days += 1
        expected = pd.date_range(
            start=pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=9, minute=30, tz=TZ),
            end=pd.Timestamp(year=date.year, month=date.month, day=date.day, hour=16, minute=0, tz=TZ),
            freq="1min",
        )
        missing = expected.difference(group.index)
        missing_total += len(missing)

    return {"missing_rth_minutes": int(missing_total), "rth_days": int(business_days)}


def load_and_validate_1m(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    validate_columns(df, path)
    df = normalize_timestamp(df, path)
    df = validate_numeric_columns(df, path)
    validate_price_logic(df, path)
    validate_volume(df, path)
    validate_order_and_duplicates(df, path)
    return df


def filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.set_index("timestamp").sort_index()
    df = df.between_time(RTH_START, RTH_END, inclusive="both")
    return df.reset_index()


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out = out.set_index("timestamp").sort_index()

    if timeframe.endswith("m"):
        rule = timeframe.replace("m", "min")
    elif timeframe == "1h":
        rule = "60min"
    elif timeframe == "1d":
        rule = "1D"
    else:
        fail("unsupported_timeframe", {"timeframe": timeframe})

    offset = pd.Timedelta(minutes=30) if timeframe == "1h" else None
    resampled = out.resample(rule, label="left", closed="left", offset=offset).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    resampled = resampled.dropna().reset_index()
    return resampled


def validate_resampled(df_resampled: pd.DataFrame, df_source: pd.DataFrame, timeframe: str, symbol: str) -> dict:
    if df_resampled.empty:
        return {"ok": True, "message": "empty_resampled"}

    source = df_source.copy()
    source = source.set_index("timestamp").sort_index()
    if timeframe.endswith("m"):
        rule = timeframe.replace("m", "min")
    elif timeframe == "1h":
        rule = "60min"
    elif timeframe == "1d":
        rule = "1D"
    else:
        return {"ok": False, "reason": "unsupported_timeframe", "timeframe": timeframe}

    offset = pd.Timedelta(minutes=30) if timeframe == "1h" else None
    expected = source.resample(rule, label="left", closed="left", offset=offset).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    expected = expected.dropna()
    expected = expected.reset_index()

    if len(expected) != len(df_resampled):
        return {
            "ok": False,
            "reason": "resample_length_mismatch",
            "symbol": symbol,
            "timeframe": timeframe,
            "expected_rows": int(len(expected)),
            "actual_rows": int(len(df_resampled)),
        }

    compare_cols = ["open", "high", "low", "close", "volume"]
    for col in compare_cols:
        if not expected[col].equals(df_resampled[col]):
            mismatches = int((expected[col] != df_resampled[col]).sum())
            return {
                "ok": False,
                "reason": "resample_value_mismatch",
                "symbol": symbol,
                "timeframe": timeframe,
                "column": col,
                "mismatches": mismatches,
            }

    return {"ok": True}


def save_ohlcv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["timestamp"] = out["timestamp"].dt.tz_convert(TZ).dt.tz_localize(None)
    out.to_csv(output_path, index=False, date_format="%Y-%m-%d %H:%M:%S")


def summarize_csv(df: pd.DataFrame, path: Path) -> dict:
    return {
        "path": str(path),
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "total_volume": float(df["volume"].sum()),
    }


def build_symbol(symbol: str, output_root: Path) -> dict:
    files = list_symbol_files(symbol, RAW_ROOT)
    if not files:
        return {"symbol": symbol, "ok": False, "reason": "no_raw_files_found"}

    source_dfs = []
    file_summaries = []

    for path in files:
        df = load_and_validate_1m(path)
        source_dfs.append(df)
        file_summaries.append({"path": str(path), "rows": int(len(df)), "start": str(df["timestamp"].min()), "end": str(df["timestamp"].max())})

    df_all = pd.concat(source_dfs, ignore_index=True)
    df_all = df_all.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    if df_all.empty:
        return {"symbol": symbol, "ok": False, "reason": "empty_concatenated_source"}

    rth_df = filter_rth(df_all)
    missing_info = validate_rth_timestamps(rth_df)

    symbol_summary = {
        "symbol": symbol,
        "ok": True,
        "files": file_summaries,
        "source_rows": int(len(df_all)),
        "rth_rows": int(len(rth_df)),
        "missing_rth_minutes": missing_info["missing_rth_minutes"],
        "rth_days": missing_info["rth_days"],
        "timeframes": {},
    }

    for timeframe in STANDARD_TIMEFRAMES:
        resampled = resample_ohlcv(rth_df, timeframe)
        if resampled.empty:
            symbol_summary["timeframes"][timeframe] = {"ok": False, "reason": "empty_resample"}
            continue

        start_str = resampled["timestamp"].dt.strftime("%Y-%m-%d").iloc[0]
        end_str = resampled["timestamp"].dt.strftime("%Y-%m-%d").iloc[-1]
        output_path = output_root / symbol / f"{symbol}_{timeframe}_{start_str}_{end_str}.csv"
        save_ohlcv(resampled, output_path)

        validation = validate_resampled(resampled, rth_df, timeframe, symbol)
        symbol_summary["timeframes"][timeframe] = {
            "ok": validation["ok"],
            "output_path": str(output_path),
            "rows": int(len(resampled)),
            **({"reason": validation["reason"]} if not validation["ok"] else {}),
        }

    return symbol_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standard OHLCV timeframes from raw 1m data")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA"], help="Symbols to process")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for processed CSVs")

    args = parser.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {"symbols": []}
    for symbol in args.symbols:
        summary["symbols"].append(build_symbol(symbol, output_root))

    success(summary)


if __name__ == "__main__":
    main()
