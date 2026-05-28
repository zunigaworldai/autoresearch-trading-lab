from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "raw"

ALPACA_BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


def require_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise SystemExit(f"ERROR: missing required environment variable: {name}")

    return value


def request_json(url: str, api_key: str, api_secret: str) -> dict:
    req = Request(
        url,
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        },
    )

    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def alpaca_bar_to_row(symbol: str, bar: dict) -> dict:
    return {
        "timestamp": bar["t"],
        "open": bar["o"],
        "high": bar["h"],
        "low": bar["l"],
        "close": bar["c"],
        "volume": bar["v"],
        "symbol": symbol,
    }


def fetch_symbol_1min(
    symbol: str,
    start: str,
    end: str,
    feed: str,
    api_key: str,
    api_secret: str,
    limit: int = 10000,
) -> pd.DataFrame:
    rows = []
    page_token = None

    while True:
        params = {
            "symbols": symbol,
            "timeframe": "1Min",
            "start": start,
            "end": end,
            "limit": limit,
            "adjustment": "raw",
            "feed": feed,
        }

        if page_token:
            params["page_token"] = page_token

        url = f"{ALPACA_BARS_URL}?{urlencode(params)}"
        payload = request_json(url, api_key=api_key, api_secret=api_secret)

        bars = payload.get("bars", {}).get(symbol, [])
        for bar in bars:
            rows.append(alpaca_bar_to_row(symbol, bar))

        page_token = payload.get("next_page_token")

        if not page_token:
            break

        time.sleep(0.25)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    return df


def save_ohlcv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    out = df[columns].copy()

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = (
        out["timestamp"]
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    out.to_csv(path, index=False)


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp"] = out["timestamp"].dt.tz_convert("America/New_York")
    out = out.set_index("timestamp")
    out = out.sort_index()

    rule = timeframe.replace("m", "min")

    resampled = out.resample(
        rule,
        label="left",
        closed="left",
    ).agg(
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


def download_and_resample(
    symbols: list[str],
    start: str,
    end: str,
    feed: str,
    output_root: Path,
) -> dict:
    api_key = require_env("ALPACA_API_KEY_ID")
    api_secret = require_env("ALPACA_API_SECRET_KEY")

    summary = {
        "start": start,
        "end": end,
        "feed": feed,
        "symbols": {},
    }

    for symbol in symbols:
        print(f"Downloading {symbol} 1Min from Alpaca...")

        df_1m = fetch_symbol_1min(
            symbol=symbol,
            start=start,
            end=end,
            feed=feed,
            api_key=api_key,
            api_secret=api_secret,
        )

        symbol_dir = output_root / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)

        file_start = start[:10]
        file_end = end[:10]

        raw_1m_path = symbol_dir / f"{symbol}_1m_{file_start}_{file_end}.csv"
        save_ohlcv(df_1m, raw_1m_path)

        symbol_summary = {
            "rows_1m": int(len(df_1m)),
            "files": {
                "1m": str(raw_1m_path),
            },
        }

        for timeframe in ["2m", "3m", "5m"]:
            resampled = resample_ohlcv(df_1m, timeframe)
            out_path = symbol_dir / f"{symbol}_{timeframe}_{file_start}_{file_end}.csv"
            save_ohlcv(resampled, out_path)

            symbol_summary[f"rows_{timeframe}"] = int(len(resampled))
            symbol_summary["files"][timeframe] = str(out_path)

        summary["symbols"][symbol] = symbol_summary

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Alpaca historical 1Min bars and resample")
    parser.add_argument("--symbols", nargs="+", required=True, help="Symbols, example: SPY QQQ")
    parser.add_argument("--start", required=True, help="UTC start, example: 2026-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="UTC end, example: 2026-05-28T00:00:00Z")
    parser.add_argument("--feed", default="iex", help="Alpaca feed, example: iex or sip")
    parser.add_argument("--output-root", default="data/raw", help="Output root folder")

    args = parser.parse_args()

    summary = download_and_resample(
        symbols=[s.upper() for s in args.symbols],
        start=args.start,
        end=args.end,
        feed=args.feed,
        output_root=Path(args.output_root),
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()