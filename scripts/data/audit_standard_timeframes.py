"""Audit all files under data/processed_standard/ for data quality."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TZ = "America/New_York"
RTH_START = pd.Timestamp("09:30").time()
RTH_END = pd.Timestamp("16:00").time()

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed_standard"
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "research"
    / "data_standardization_audit"
)
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "1d"]
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

INTRADAY_TIMEFRAMES = {"5m", "15m", "30m", "1h"}


def find_file(symbol: str, timeframe: str) -> Path | None:
    symbol_dir = DATA_ROOT / symbol
    if not symbol_dir.exists():
        return None
    matches = sorted(symbol_dir.glob(f"{symbol}_{timeframe}_*.csv"))
    return matches[0] if matches else None


def check_columns(df: pd.DataFrame) -> tuple[bool, str]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "ok"


def check_timestamps(ts: pd.Series) -> tuple[bool, str, int]:
    bad = int(ts.isna().sum())
    if bad > 0:
        return False, f"{bad} unparseable", bad
    return True, "ok", 0


def check_monotonic(ts: pd.Series) -> tuple[bool, str]:
    if not ts.is_monotonic_increasing:
        violations = int((ts.diff().dt.total_seconds().dropna() <= 0).sum())
        return False, f"{violations} non-increasing steps"
    return True, "ok"


def check_duplicates(ts: pd.Series) -> tuple[bool, str, int]:
    dupes = int(ts.duplicated().sum())
    if dupes > 0:
        return False, f"{dupes} duplicates", dupes
    return True, "ok", 0


def check_ohlc(df: pd.DataFrame) -> tuple[bool, str, int]:
    issues = 0
    msgs = []

    bad_high = df["high"] < df[["open", "close", "low"]].max(axis=1)
    if bad_high.any():
        n = int(bad_high.sum())
        issues += n
        msgs.append(f"high<max(O,C,L):{n}")

    bad_low = df["low"] > df[["open", "close", "high"]].min(axis=1)
    if bad_low.any():
        n = int(bad_low.sum())
        issues += n
        msgs.append(f"low>min(O,C,H):{n}")

    non_pos = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if non_pos.any():
        n = int(non_pos.sum())
        issues += n
        msgs.append(f"price<=0:{n}")

    if issues > 0:
        return False, "; ".join(msgs), issues
    return True, "ok", 0


def check_volume(df: pd.DataFrame) -> tuple[bool, str, int]:
    neg = int((df["volume"] < 0).sum())
    if neg > 0:
        return False, f"{neg} negative", neg
    return True, "ok", 0


def check_rth(df: pd.DataFrame) -> tuple[bool, str, int]:
    times = df["timestamp"].dt.time
    before_open = times < RTH_START
    after_close = times > RTH_END
    outside = before_open | after_close
    n = int(outside.sum())
    if n > 0:
        examples = df.loc[outside, "timestamp"].head(3).dt.strftime("%Y-%m-%d %H:%M").tolist()
        return False, f"{n} bars outside RTH (e.g. {', '.join(examples)})", n
    return True, "ok", 0


def check_1h_anchor(df: pd.DataFrame) -> tuple[bool, str, int]:
    first_bars = df.groupby(df["timestamp"].dt.date)["timestamp"].first()
    bad = first_bars[first_bars.dt.minute != 30]
    n = int(len(bad))
    if n > 0:
        examples = bad.head(3).dt.strftime("%Y-%m-%d %H:%M").tolist()
        return (
            False,
            f"{n}/{len(first_bars)} days not anchored to :30 (e.g. {', '.join(examples)})",
            n,
        )
    return True, "ok", 0


def check_daily_one_per_date(df: pd.DataFrame) -> tuple[bool, str, int]:
    dates = df["timestamp"].dt.date
    dupes = int(dates.duplicated().sum())
    if dupes > 0:
        return False, f"{dupes} duplicate dates", dupes
    return True, "ok", 0


def detect_gaps(df: pd.DataFrame, timeframe: str) -> tuple[str, list[dict]]:
    """Flag missing trading days or abnormally large intraday gaps.

    Overnight gaps between RTH sessions are expected and excluded.
    """
    if len(df) < 2:
        return "too few bars", []

    dates = df["timestamp"].dt.date
    unique_dates = dates.drop_duplicates().reset_index(drop=True)

    if timeframe == "1d":
        deltas = pd.Series(unique_dates).diff().apply(lambda d: d.days if pd.notna(d) else 0)
        # Weekends = 3 days (Fri->Mon). Flag gaps > 4 calendar days.
        threshold = 4
        large = deltas[deltas > threshold]
        gaps = []
        for idx in large.index:
            gaps.append(
                {
                    "from": str(unique_dates.iloc[idx - 1]),
                    "to": str(unique_dates.iloc[idx]),
                    "gap": f"{int(deltas[idx])} calendar days",
                }
            )
    else:
        # For intraday: check date-to-date gaps (skip normal overnights).
        date_deltas = pd.Series(unique_dates).diff().apply(
            lambda d: d.days if pd.notna(d) else 0
        )
        large_date = date_deltas[date_deltas > 4]
        gaps = []
        for idx in large_date.index:
            gaps.append(
                {
                    "from": str(unique_dates.iloc[idx - 1]),
                    "to": str(unique_dates.iloc[idx]),
                    "gap": f"{int(date_deltas[idx])} calendar days",
                }
            )

        # Also check for abnormally large intra-day gaps within each session.
        expected_minutes = {"5m": 5, "15m": 15, "30m": 30, "1h": 60}
        max_intraday_gap = expected_minutes[timeframe] * 6
        for date_val, group in df.groupby(dates):
            ts = group["timestamp"].reset_index(drop=True)
            if len(ts) < 2:
                continue
            mins = ts.diff().dt.total_seconds().dropna() / 60
            big = mins[mins > max_intraday_gap]
            for i in big.index:
                gaps.append(
                    {
                        "from": str(ts.iloc[i - 1]),
                        "to": str(ts.iloc[i]),
                        "gap": f"{int(mins[i])}m intraday",
                    }
                )

    if gaps:
        return f"{len(gaps)} gaps found", gaps
    return "ok", []


def audit_file(symbol: str, timeframe: str) -> dict:
    row = {
        "symbol": symbol,
        "timeframe": timeframe,
        "file": "",
        "rows": 0,
        "start": "",
        "end": "",
        "columns": "",
        "timestamp_parse": "",
        "monotonic": "",
        "duplicates": "",
        "ohlc_integrity": "",
        "volume": "",
        "rth_check": "",
        "anchor_check": "",
        "daily_unique": "",
        "gaps": "",
        "gap_details": "",
        "overall": "PASS",
    }

    path = find_file(symbol, timeframe)
    if path is None:
        row["file"] = "NOT FOUND"
        row["overall"] = "FAIL"
        return row

    row["file"] = path.name

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        row["file"] = f"READ ERROR: {exc}"
        row["overall"] = "FAIL"
        return row

    row["rows"] = len(df)

    ok, msg = check_columns(df)
    row["columns"] = msg
    if not ok:
        row["overall"] = "FAIL"
        return row

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    ok, msg, _ = check_timestamps(df["timestamp"])
    row["timestamp_parse"] = msg
    if not ok:
        row["overall"] = "FAIL"

    bad_ts = df["timestamp"].isna()
    if bad_ts.any():
        df = df[~bad_ts].copy()

    if df.empty:
        row["overall"] = "FAIL"
        return row

    row["start"] = str(df["timestamp"].iloc[0])
    row["end"] = str(df["timestamp"].iloc[-1])

    ok, msg = check_monotonic(df["timestamp"])
    row["monotonic"] = msg
    if not ok:
        row["overall"] = "FAIL"

    ok, msg, _ = check_duplicates(df["timestamp"])
    row["duplicates"] = msg
    if not ok:
        row["overall"] = "FAIL"

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    ok, msg, _ = check_ohlc(df)
    row["ohlc_integrity"] = msg
    if not ok:
        row["overall"] = "FAIL"

    ok, msg, _ = check_volume(df)
    row["volume"] = msg
    if not ok:
        row["overall"] = "FAIL"

    if timeframe in INTRADAY_TIMEFRAMES:
        ok, msg, _ = check_rth(df)
        row["rth_check"] = msg
        if not ok:
            row["overall"] = "FAIL"
    else:
        row["rth_check"] = "n/a"

    if timeframe == "1h":
        ok, msg, _ = check_1h_anchor(df)
        row["anchor_check"] = msg
        if not ok:
            row["overall"] = "WARN"
    else:
        row["anchor_check"] = "n/a"

    if timeframe == "1d":
        ok, msg, _ = check_daily_one_per_date(df)
        row["daily_unique"] = msg
        if not ok:
            row["overall"] = "FAIL"
    else:
        row["daily_unique"] = "n/a"

    gap_msg, gap_list = detect_gaps(df, timeframe)
    row["gaps"] = gap_msg
    row["gap_details"] = "; ".join(
        f"{g['from']} -> {g['to']} ({g['gap']})" for g in gap_list[:10]
    )

    return row


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("STANDARD TIMEFRAME AUDIT SUMMARY")
    print("=" * 90)

    header = f"{'Symbol':<8} {'TF':<6} {'Rows':>8}  {'Start':<12} {'End':<12} {'Status':<6}  Notes"
    print(header)
    print("-" * 90)

    for r in results:
        start = r["start"][:10] if r["start"] else ""
        end = r["end"][:10] if r["end"] else ""
        notes = []
        for key in [
            "columns",
            "timestamp_parse",
            "monotonic",
            "duplicates",
            "ohlc_integrity",
            "volume",
            "rth_check",
            "anchor_check",
            "daily_unique",
            "gaps",
        ]:
            val = r.get(key, "")
            if val and val not in ("ok", "n/a", ""):
                notes.append(f"{key}={val}")

        note_str = "; ".join(notes) if notes else ""
        print(
            f"{r['symbol']:<8} {r['timeframe']:<6} {r['rows']:>8}  {start:<12} {end:<12} {r['overall']:<6}  {note_str}"
        )

    print("=" * 90)

    total = len(results)
    passed = sum(1 for r in results if r["overall"] == "PASS")
    warned = sum(1 for r in results if r["overall"] == "WARN")
    failed = sum(1 for r in results if r["overall"] == "FAIL")
    print(f"\nTotal: {total}  |  PASS: {passed}  |  WARN: {warned}  |  FAIL: {failed}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit standard OHLCV timeframe files for data quality"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=SYMBOLS,
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=TIMEFRAMES,
    )
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    results = []
    for symbol in args.symbols:
        for tf in args.timeframes:
            print(f"Auditing {symbol} {tf} ...")
            results.append(audit_file(symbol, tf))

    print_summary(results)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "standard_timeframe_audit.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
