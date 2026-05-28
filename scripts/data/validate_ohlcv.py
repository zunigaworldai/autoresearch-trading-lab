from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def fail(reason: str, details: dict | None = None) -> None:
    payload = {
        "ok": False,
        "reason": reason,
        "details": details or {},
    }
    print(json.dumps(payload, indent=2))
    raise SystemExit(1)


def success(details: dict) -> None:
    payload = {
        "ok": True,
        "details": details,
    }
    print(json.dumps(payload, indent=2))


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail("file_not_found", {"path": str(path)})

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        fail("csv_read_error", {"error": str(exc)})

    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        fail("missing_required_columns", {"missing": missing, "found": list(df.columns)})


def validate_not_empty(df: pd.DataFrame) -> None:
    if df.empty:
        fail("empty_file")


def normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception as exc:
        fail("timestamp_parse_error", {"error": str(exc)})

    bad_timestamps = df["timestamp"].isna().sum()
    if bad_timestamps > 0:
        fail("invalid_timestamps", {"bad_rows": int(bad_timestamps)})

    return df


def validate_no_missing_values(df: pd.DataFrame) -> None:
    missing = df[REQUIRED_COLUMNS].isna().sum()
    bad = {k: int(v) for k, v in missing.items() if v > 0}

    if bad:
        fail("missing_values", bad)


def validate_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = ["open", "high", "low", "close", "volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = df[numeric_cols].isna().sum()
    bad = {k: int(v) for k, v in bad.items() if v > 0}

    if bad:
        fail("invalid_numeric_values", bad)

    return df


def validate_price_logic(df: pd.DataFrame) -> None:
    bad_high = df[df["high"] < df[["open", "close", "low"]].max(axis=1)]
    bad_low = df[df["low"] > df[["open", "close", "high"]].min(axis=1)]

    if not bad_high.empty:
        fail("invalid_high_price", {"bad_rows": int(len(bad_high))})

    if not bad_low.empty:
        fail("invalid_low_price", {"bad_rows": int(len(bad_low))})

    non_positive_prices = df[
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
    ]

    if not non_positive_prices.empty:
        fail("non_positive_prices", {"bad_rows": int(len(non_positive_prices))})


def validate_volume(df: pd.DataFrame) -> None:
    bad_volume = df[df["volume"] < 0]

    if not bad_volume.empty:
        fail("negative_volume", {"bad_rows": int(len(bad_volume))})


def validate_order_and_duplicates(df: pd.DataFrame) -> None:
    if not df["timestamp"].is_monotonic_increasing:
        fail("timestamps_not_sorted")

    duplicates = df["timestamp"].duplicated().sum()

    if duplicates > 0:
        fail("duplicate_timestamps", {"duplicates": int(duplicates)})


def validate_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> None:
    if start:
        start_dt = pd.to_datetime(start)
        if df["timestamp"].min() < start_dt:
            fail(
                "data_starts_before_expected_start",
                {
                    "expected_start": str(start_dt),
                    "actual_start": str(df["timestamp"].min()),
                },
            )

    if end:
        end_dt = pd.to_datetime(end)
        if df["timestamp"].max() > end_dt:
            fail(
                "data_ends_after_expected_end",
                {
                    "expected_end": str(end_dt),
                    "actual_end": str(df["timestamp"].max()),
                },
            )


def summarize(df: pd.DataFrame, path: Path) -> dict:
    return {
        "path": str(path),
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "columns": list(df.columns),
        "first_close": float(df["close"].iloc[0]),
        "last_close": float(df["close"].iloc[-1]),
        "min_close": float(df["close"].min()),
        "max_close": float(df["close"].max()),
        "total_volume": float(df["volume"].sum()),
    }


def validate_file(path: Path, start: str | None = None, end: str | None = None) -> dict:
    df = load_csv(path)

    validate_not_empty(df)
    validate_columns(df)

    df = normalize_timestamp(df)
    df = validate_numeric_columns(df)

    validate_no_missing_values(df)
    validate_price_logic(df)
    validate_volume(df)
    validate_order_and_duplicates(df)
    validate_date_range(df, start=start, end=end)

    return summarize(df, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OHLCV historical CSV data")
    parser.add_argument("path", help="Path to OHLCV CSV file")
    parser.add_argument("--start", default=None, help="Expected start date, example: 2026-01-01")
    parser.add_argument("--end", default=None, help="Expected end date, example: 2026-05-27")

    args = parser.parse_args()

    result = validate_file(
        path=Path(args.path),
        start=args.start,
        end=args.end,
    )

    success(result)


if __name__ == "__main__":
    main()