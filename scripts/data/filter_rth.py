from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def filter_rth(input_path: Path, output_path: Path) -> dict:
    df = pd.read_csv(input_path)

    if "timestamp" not in df.columns:
        raise SystemExit("ERROR: missing timestamp column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    before_rows = len(df)

    df = df.dropna(subset=["timestamp"])
    df = df[
        (df["timestamp"].dt.time >= pd.to_datetime("09:30").time())
        & (df["timestamp"].dt.time <= pd.to_datetime("16:00").time())
    ].copy()

    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return {
        "ok": True,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "before_rows": int(before_rows),
        "after_rows": int(len(df)),
        "start": str(df["timestamp"].min()) if len(df) else None,
        "end": str(df["timestamp"].max()) if len(df) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter OHLCV CSV to regular trading hours")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    result = filter_rth(
        input_path=Path(args.input),
        output_path=Path(args.output),
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()