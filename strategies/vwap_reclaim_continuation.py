from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_PARAMS = {
    "use_rth": True,
    "session_start": "09:30",
    "session_end": "16:00",
    "entry_start": "09:30",
    "entry_end": "16:00",

    # Direction: "long", "short", "both"
    "direction": "both",

    # VWAP reclaim / continuation core
    "reclaim_mode": "close_cross",  # "close_cross" or "two_bar_hold"
    "reclaim_buffer_atr": 0.05,

    # Trend / momentum filters
    "ema_fast_len": 9,
    "ema_slow_len": 21,
    "trend_len": 50,
    "require_trend_alignment": True,
    "require_ema_slope": True,
    "slope_lookback": 5,
    "min_slope_atr": 0.02,

    # Volume / volatility filters
    "require_volume_confirm": True,
    "vol_len": 20,
    "vol_mult": 1.10,
    "atr_len": 14,
    "min_atr_pct": 0.0010,
    "max_atr_pct": 0.0400,

    # Risk / target
    "stop_mode": "atr",       # "atr", "vwap", or "swing"
    "atr_stop_mult": 1.0,
    "swing_lookback": 8,
    "stop_buffer_atr": 0.10,
    "target_mode": "rr",      # "rr" only for now
    "min_rr": 1.5,

    "one_trade_per_day": True,
}


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    out = df.copy()

    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    elif "timestamp" in out.columns:
        out["datetime"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out["datetime"] = pd.to_datetime(out.index, errors="coerce")
    else:
        raise ValueError("DataFrame must contain datetime, timestamp, or DatetimeIndex")

    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    return out.sort_values("datetime").reset_index(drop=True)


def _time_value(value: str):
    return pd.to_datetime(value).time()


def _in_time_window(out: pd.DataFrame, start: str, end: str) -> pd.Series:
    t = pd.to_datetime(out["datetime"]).dt.time
    return (t >= _time_value(start)) & (t <= _time_value(end))


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


def _session_vwap(df: pd.DataFrame) -> pd.Series:
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = hlc3 * df["volume"]
    session_key = pd.to_datetime(df["datetime"]).dt.date
    cum_pv = pv.groupby(session_key).cumsum()
    cum_vol = df["volume"].groupby(session_key).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


def generate_signals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """
    VWAP Reclaim / Continuation strategy.

    Hypothesis:
    A reclaim of session VWAP in the direction of short-term trend can mark continuation
    when confirmed by volume, slope, and controlled volatility.

    Compatible with engine.trade_simulator.run_trade_simulation:
    - entry_signal: 1 long, -1 short, 0 no entry
    - exit_stop
    - exit_tp
    - setup
    """

    params = {**DEFAULT_PARAMS, **(params or {})}
    out = _ensure_ohlcv(df)

    out["date"] = pd.to_datetime(out["datetime"]).dt.date
    out["in_session"] = _in_time_window(out, params["session_start"], params["session_end"])
    out["in_entry_window"] = _in_time_window(out, params["entry_start"], params["entry_end"])

    out["vwap"] = _session_vwap(out)
    out["atr"] = _atr(out, int(params["atr_len"]))
    out["atr_pct"] = out["atr"] / out["close"].replace(0, np.nan)

    out["ema_fast"] = out["close"].ewm(span=int(params["ema_fast_len"]), adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=int(params["ema_slow_len"]), adjust=False).mean()
    out["trend_ma"] = out["close"].rolling(int(params["trend_len"])).mean()

    slope_lookback = int(params["slope_lookback"])
    out["ema_fast_slope"] = out["ema_fast"] - out["ema_fast"].shift(slope_lookback)

    out["vol_sma"] = out["volume"].rolling(int(params["vol_len"])).mean()

    volume_ok = pd.Series(True, index=out.index)
    if bool(params.get("require_volume_confirm", True)):
        volume_ok = out["volume"] >= out["vol_sma"] * float(params["vol_mult"])

    volatility_ok = (
        (out["atr_pct"] >= float(params["min_atr_pct"]))
        & (out["atr_pct"] <= float(params["max_atr_pct"]))
    )

    trend_long_ok = pd.Series(True, index=out.index)
    trend_short_ok = pd.Series(True, index=out.index)

    if bool(params.get("require_trend_alignment", True)):
        trend_long_ok &= out["close"] > out["trend_ma"]
        trend_long_ok &= out["ema_fast"] > out["ema_slow"]
        trend_short_ok &= out["close"] < out["trend_ma"]
        trend_short_ok &= out["ema_fast"] < out["ema_slow"]

    if bool(params.get("require_ema_slope", True)):
        min_slope = out["atr"] * float(params["min_slope_atr"])
        trend_long_ok &= out["ema_fast_slope"] > min_slope
        trend_short_ok &= out["ema_fast_slope"] < -min_slope

    buffer = out["atr"] * float(params["reclaim_buffer_atr"])
    reclaim_mode = str(params["reclaim_mode"]).lower()

    if reclaim_mode == "close_cross":
        long_reclaim = (out["close"].shift(1) <= out["vwap"].shift(1)) & (out["close"] > out["vwap"] + buffer)
        short_reclaim = (out["close"].shift(1) >= out["vwap"].shift(1)) & (out["close"] < out["vwap"] - buffer)
    elif reclaim_mode == "two_bar_hold":
        long_reclaim = (
            (out["close"].shift(2) <= out["vwap"].shift(2))
            & (out["close"].shift(1) > out["vwap"].shift(1))
            & (out["close"] > out["vwap"] + buffer)
        )
        short_reclaim = (
            (out["close"].shift(2) >= out["vwap"].shift(2))
            & (out["close"].shift(1) < out["vwap"].shift(1))
            & (out["close"] < out["vwap"] - buffer)
        )
    else:
        raise ValueError("reclaim_mode must be 'close_cross' or 'two_bar_hold'")

    base = (
        out["in_session"]
        & out["in_entry_window"]
        & out["vwap"].notna()
        & out["atr"].notna()
        & out["trend_ma"].notna()
        & out["ema_fast"].notna()
        & out["ema_slow"].notna()
        & volume_ok
        & volatility_ok
    )

    direction = str(params.get("direction", "both")).lower()
    allow_long = direction in {"long", "both"}
    allow_short = direction in {"short", "both"}

    out["long_signal_raw"] = allow_long & base & long_reclaim & trend_long_ok
    out["short_signal_raw"] = allow_short & base & short_reclaim & trend_short_ok

    if bool(params.get("one_trade_per_day", True)):
        raw_any = out["long_signal_raw"] | out["short_signal_raw"]
        already_had_signal = raw_any.groupby(out["date"]).cumsum().shift(fill_value=0) > 0
        out["long_signal"] = out["long_signal_raw"] & (~already_had_signal)
        out["short_signal"] = out["short_signal_raw"] & (~already_had_signal)
    else:
        out["long_signal"] = out["long_signal_raw"]
        out["short_signal"] = out["short_signal_raw"]

    stop_mode = str(params["stop_mode"]).lower()
    stop_buffer = out["atr"] * float(params["stop_buffer_atr"])

    if stop_mode == "atr":
        long_stop = out["close"] - out["atr"] * float(params["atr_stop_mult"])
        short_stop = out["close"] + out["atr"] * float(params["atr_stop_mult"])
    elif stop_mode == "vwap":
        long_stop = out["vwap"] - stop_buffer
        short_stop = out["vwap"] + stop_buffer
    elif stop_mode == "swing":
        long_stop = out["low"].rolling(int(params["swing_lookback"])).min() - stop_buffer
        short_stop = out["high"].rolling(int(params["swing_lookback"])).max() + stop_buffer
    else:
        raise ValueError("stop_mode must be 'atr', 'vwap', or 'swing'")

    long_risk = out["close"] - long_stop
    short_risk = short_stop - out["close"]
    rr = float(params["min_rr"])

    long_tp = out["close"] + long_risk * rr
    short_tp = out["close"] - short_risk * rr

    long_ok = out["long_signal"] & (long_risk > 0)
    short_ok = out["short_signal"] & (short_risk > 0)

    out["entry_signal"] = 0
    out.loc[long_ok, "entry_signal"] = 1
    out.loc[short_ok, "entry_signal"] = -1

    out["exit_stop"] = np.nan
    out["exit_tp"] = np.nan
    out["setup"] = ""

    out.loc[long_ok, "exit_stop"] = long_stop[long_ok]
    out.loc[long_ok, "exit_tp"] = long_tp[long_ok]
    out.loc[long_ok, "setup"] = "VWAP_RECLAIM_CONT_LONG"

    out.loc[short_ok, "exit_stop"] = short_stop[short_ok]
    out.loc[short_ok, "exit_tp"] = short_tp[short_ok]
    out.loc[short_ok, "setup"] = "VWAP_RECLAIM_CONT_SHORT"

    return out
