from __future__ import annotations

import pandas as pd
import numpy as np


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",

    # Opening range definition.
    "or_start": "09:30",
    "or_minutes": 15,

    # Entry window after the opening range has completed.
    "entry_start": None,
    "entry_end": "11:00",

    # Direction: "long", "short", or "both".
    "direction": "both",

    # Risk/target.
    "min_rr": 2.0,
    "stop_mode": "or_opposite",  # "or_opposite" or "atr"
    "atr_len": 14,
    "atr_stop_mult": 1.0,
    "stop_buffer_atr": 0.05,

    # Filters.
    "require_volume_confirm": False,
    "vol_len": 20,
    "vol_mult": 1.2,
    "require_trend_filter": False,
    "trend_len": 50,
    "one_trade_per_day": True,
}


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    out = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif "timestamp" in out.columns:
        out["datetime"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out["datetime"] = pd.to_datetime(out.index, errors="coerce")
    else:
        raise ValueError("DataFrame must contain datetime, timestamp, or DatetimeIndex")

    out = out.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    return out.sort_values("datetime").reset_index(drop=True)


def _time_series(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["datetime"], errors="coerce").dt.time


def _time_value(value: str):
    return pd.to_datetime(value).time()


def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1 / length, adjust=False).mean()


def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return _rma(tr, length)


def _opening_range_features(out: pd.DataFrame, params: dict) -> pd.DataFrame:
    dt = pd.to_datetime(out["datetime"], errors="coerce")
    out["date"] = dt.dt.date
    out["time"] = dt.dt.time

    or_start = _time_value(params["or_start"])
    or_end_dt = pd.to_datetime(params["or_start"]) + pd.Timedelta(minutes=int(params["or_minutes"]))
    or_end = or_end_dt.time()

    in_or = (out["time"] >= or_start) & (out["time"] < or_end)

    or_high_by_day = out.loc[in_or].groupby("date")["high"].max()
    or_low_by_day = out.loc[in_or].groupby("date")["low"].min()

    out["or_high"] = out["date"].map(or_high_by_day)
    out["or_low"] = out["date"].map(or_low_by_day)
    out["or_range"] = out["or_high"] - out["or_low"]
    out["or_complete"] = out["time"] >= or_end

    return out


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    Opening Range Breakout strategy.

    Output is compatible with engine.trade_simulator.run_trade_simulation:
    - entry_signal: 1 long, -1 short, 0 no entry
    - exit_stop
    - exit_tp
    - setup

    The signal enters on close after price breaks the completed opening range.
    """

    params = {**DEFAULT_PARAMS, **(params or {})}
    out = _ensure_ohlcv(df)

    out = _opening_range_features(out, params)

    t = _time_series(out)
    session_start = _time_value(params["session_start"])
    session_end = _time_value(params["session_end"])
    entry_start = params.get("entry_start")
    entry_end = params.get("entry_end") or params["session_end"]

    or_end_dt = pd.to_datetime(params["or_start"]) + pd.Timedelta(minutes=int(params["or_minutes"]))
    default_entry_start = or_end_dt.strftime("%H:%M")

    entry_start_t = _time_value(entry_start or default_entry_start)
    entry_end_t = _time_value(entry_end)

    in_session = (t >= session_start) & (t <= session_end)
    in_entry_window = (t >= entry_start_t) & (t <= entry_end_t)

    atr_len = int(params["atr_len"])
    out["atr"] = _atr(out, atr_len)
    out["vol_sma"] = out["volume"].rolling(int(params["vol_len"])).mean()
    out["trend_ma"] = out["close"].rolling(int(params["trend_len"])).mean()

    volume_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_volume_confirm", False)):
        volume_ok = out["volume"] > out["vol_sma"] * float(params["vol_mult"])

    trend_long_ok = pd.Series(True, index=out.index)
    trend_short_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_trend_filter", False)):
        trend_long_ok = out["close"] > out["trend_ma"]
        trend_short_ok = out["close"] < out["trend_ma"]

    valid_or = out["or_high"].notna() & out["or_low"].notna() & (out["or_range"] > 0)

    prev_close = out["close"].shift(1)
    breakout_long = (prev_close <= out["or_high"].shift(1)) & (out["close"] > out["or_high"])
    breakout_short = (prev_close >= out["or_low"].shift(1)) & (out["close"] < out["or_low"])

    direction = str(params.get("direction", "both")).lower()
    allow_long = direction in {"long", "both"}
    allow_short = direction in {"short", "both"}

    base = (
        in_session
        & in_entry_window
        & out["or_complete"]
        & valid_or
        & volume_ok
    )

    out["long_signal_raw"] = allow_long & base & breakout_long & trend_long_ok
    out["short_signal_raw"] = allow_short & base & breakout_short & trend_short_ok

    if bool(params.get("one_trade_per_day", True)):
        raw_any = out["long_signal_raw"] | out["short_signal_raw"]
        already_had_signal = raw_any.groupby(out["date"]).cumsum().shift(fill_value=0) > 0
        out["long_signal"] = out["long_signal_raw"] & (~already_had_signal)
        out["short_signal"] = out["short_signal_raw"] & (~already_had_signal)
    else:
        out["long_signal"] = out["long_signal_raw"]
        out["short_signal"] = out["short_signal_raw"]

    rr = float(params["min_rr"])
    stop_buffer = out["atr"] * float(params["stop_buffer_atr"])
    stop_mode = str(params.get("stop_mode", "or_opposite")).lower()

    if stop_mode == "atr":
        long_stop = out["close"] - out["atr"] * float(params["atr_stop_mult"])
        short_stop = out["close"] + out["atr"] * float(params["atr_stop_mult"])
    elif stop_mode == "or_opposite":
        long_stop = out["or_low"] - stop_buffer
        short_stop = out["or_high"] + stop_buffer
    else:
        raise ValueError("stop_mode must be 'or_opposite' or 'atr'")

    long_risk = out["close"] - long_stop
    short_risk = short_stop - out["close"]

    long_ok = out["long_signal"] & (long_risk > 0)
    short_ok = out["short_signal"] & (short_risk > 0)

    out["entry_signal"] = 0
    out.loc[long_ok, "entry_signal"] = 1
    out.loc[short_ok, "entry_signal"] = -1

    out["exit_stop"] = np.nan
    out["exit_tp"] = np.nan
    out["setup"] = ""

    out.loc[long_ok, "exit_stop"] = long_stop[long_ok]
    out.loc[long_ok, "exit_tp"] = out.loc[long_ok, "close"] + long_risk[long_ok] * rr
    out.loc[long_ok, "setup"] = "ORB_LONG"

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = out.loc[short_ok, "close"] - short_risk[short_ok] * rr
    out.loc[short_ok, "setup"] = "ORB_SHORT"

    return out
